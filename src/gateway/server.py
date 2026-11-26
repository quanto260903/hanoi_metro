"""
Hanoi Metro AI Agent - One-pass Chat Gateway.
API tương thích OpenAI (/v1/models, /v1/chat/completions) để LibreChat khai báo như một "mô hình".
Mỗi tin nhắn: tìm kiếm 2 nhánh → gọi LLM ĐÚNG MỘT LẦN → phát trực tiếp câu trả lời + nguồn.
So với Agent + công cụ MCP (mô hình quyết định gọi công cụ, công cụ lại gọi LLM, mô hình chép lại kết quả)
cách này bớt một lượt LLM và không còn khâu mô hình viết lại kết quả.

Mô hình cung cấp:
  hanoi-metro-hoi-dap    hỏi đáp có trích dẫn (hiểu câu hỏi nối tiếp); lệnh nhanh: /tomtat, /baocao, /lietke
  hanoi-metro-tom-tat    tin nhắn là số hiệu / mô tả văn bản cần tóm tắt
  hanoi-metro-soan-thao  tin nhắn là chủ đề; tự nhận loại (báo cáo, tờ trình, công văn, tổng hợp) và số hiệu căn cứ

Chạy: python -m src.gateway.server   (GATEWAY_HOST:GATEWAY_PORT, mặc định 127.0.0.1:8766)
"""

import argparse
import copy
import hmac
import threading
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Iterator, List, Optional, Tuple, Union

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from src.access.policy import AccessPolicy, PolicySource, Principal
from src.access.registration import CHO_DUYET, COMMANDS, Registrations, handle_command, status_text
from src.agent.assistant import DocumentAssistant, Plan
from src.agent.errors import user_message
from src.config import settings
from src.gateway.documents import DocumentLinks, media_type
from src.store.base import Filters, access_allows
from src.text.codes import extract_codes
from src.text.normalize import fold

logger = logging.getLogger("hanoi_metro.gateway")

MODELS = {
    "hanoi-metro-hoi-dap": "Hỏi đáp tài liệu nội bộ (có trích dẫn)",
    "hanoi-metro-tom-tat": "Tóm tắt một văn bản",
    "hanoi-metro-soan-thao": "Soạn dự thảo báo cáo / tờ trình / công văn",
}

HELP = """**Trợ lý tài liệu Hanoi Metro** – cách dùng:
- Đặt câu hỏi bình thường, ví dụ: *Thiết bị nào tại DC Duy Tân bị hỏng?*
- `/tomtat <số hiệu hoặc mô tả văn bản>` – tóm tắt (thêm *ngắn* hoặc *gạch đầu dòng* để đổi kiểu)
- `/baocao <chủ đề>` – soạn dự thảo; ghi *tờ trình*, *công văn*, *tổng hợp* để đổi loại; ghi số hiệu để dùng làm căn cứ
- `/lietke <điều kiện>` – liệt kê đầy đủ, ví dụ: */lietke quyết định năm 2024 về bảo hiểm*
- `/phongban` – xem / khai báo đơn vị công tác (`/phongban KHDA Nguyễn Văn A`); trưởng phòng: `/choduyet`, `/duyet <email>`"""


@dataclass
class Turn:
    role: str
    text: str


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # [{type: text, text: ...}, {type: image_url...}]
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def parse_turns(messages: List[dict]) -> List[Turn]:
    return [Turn(m.get("role", ""), _text_of(m.get("content")).strip()) for m in messages if m.get("role") in ("user", "assistant")]


def history_text(turns: List[Turn], max_chars: int = 1500) -> str:
    """Hai lượt gần nhất trước câu hỏi hiện tại (bỏ phần danh sách nguồn) để LLM hiểu câu hỏi nối tiếp."""
    lines = []
    for t in turns[-5:-1]:
        body = t.text.split("**Nguồn:**")[0].strip()
        lines.append(f"{'Người dùng' if t.role == 'user' else 'Trợ lý'}: {body[:600]}")
    return "\n".join(lines)[-max_chars:]


def search_query_for(turns: List[Turn]) -> str:
    """Câu hỏi ngắn kiểu "còn thiết bị nào khác?" được ghép với câu hỏi trước để tìm đúng tài liệu."""
    current = turns[-1].text
    previous = [t.text for t in turns[:-1] if t.role == "user"]
    if previous and len(current.split()) < 8 and not extract_codes(current):
        return f"{previous[-1]} {current}"
    return current


def summary_args(text: str) -> Tuple[str, str]:
    f = fold(text)
    style = "ngan" if re.search(r"\bngan\b", f) else ("gach_dau_dong" if "gach dau dong" in f else "chi_tiet")
    ref = re.sub(r"(?i)\b(tóm tắt|ngắn gọn|ngắn|gạch đầu dòng|chi tiết|văn bản|giúp tôi|hãy)\b", " ", text)
    return " ".join(ref.split()) or text, style


def report_args(text: str) -> Tuple[str, str, List[str]]:
    f = fold(text)
    if "to trinh" in f:
        kind = "to_trinh"
    elif "cong van" in f:
        kind = "cong_van"
    elif "tong hop" in f:
        kind = "tong_hop"
    else:
        kind = "bao_cao"
    refs = [c for c in extract_codes(text) if "/" in c]
    return text, kind, refs


FEATURE_OF_MODEL = {"hanoi-metro-hoi-dap": "hoi_dap", "hanoi-metro-tom-tat": "tom_tat", "hanoi-metro-soan-thao": "soan_thao"}
FEATURE_NAMES = {"hoi_dap": "Hỏi đáp", "tom_tat": "Tóm tắt văn bản", "soan_thao": "Soạn thảo văn bản", "liet_ke": "Liệt kê văn bản"}


def build_plan(assistant: DocumentAssistant, model: str, turns: List[Turn], filters: Optional[Filters] = None,
               features: Optional[FrozenSet[str]] = None) -> Plan:
    """features=None: không giới hạn chức năng (tắt phân quyền); filters mang phạm vi xem của người dùng."""
    text = turns[-1].text
    feature = FEATURE_OF_MODEL.get(model, "hoi_dap")
    command = re.match(r"^/(\w+)\s*(.*)$", text, re.S)
    if command:
        name, rest = command.group(1).lower(), command.group(2).strip()
        if name in ("tomtat", "tt"):
            model, text, feature = "hanoi-metro-tom-tat", rest, "tom_tat"
        elif name in ("baocao", "soanthao", "bc"):
            model, text, feature = "hanoi-metro-soan-thao", rest, "soan_thao"
        elif name in ("lietke", "lk"):
            if features is not None and "liet_ke" not in features:
                return Plan(direct=_denied("liet_ke"))
            return Plan(direct=assistant.list_documents(rest or "tất cả", filters).text)
        else:
            return Plan(direct=HELP)
    if not text or fold(text) in ("help", "tro giup", "huong dan", "?"):
        return Plan(direct=HELP)
    if features is not None and feature not in features:
        return Plan(direct=_denied(feature))

    if model == "hanoi-metro-tom-tat":
        ref, style = summary_args(text)
        return assistant.plan_summarize(ref, style=style, filters=filters)
    if model == "hanoi-metro-soan-thao":
        topic, kind, refs = report_args(text)
        return assistant.plan_report(topic, kind, doc_refs=refs or None, filters=filters)
    return assistant.plan_ask(text, filters, history=history_text(turns), search_query=search_query_for(turns))


def _denied(feature: str) -> str:
    return (f"Tài khoản của bạn chưa được cấp quyền dùng chức năng **{FEATURE_NAMES.get(feature, feature)}**. "
            "Liên hệ quản trị hệ thống nếu cần.")


NOT_REGISTERED = ("Tài khoản của bạn chưa được cấp quyền tra cứu kho tài liệu nội bộ. "
                  "Liên hệ quản trị hệ thống để được thêm vào danh sách người dùng.")


class AccessLog:
    """Nhật ký truy cập dạng JSON Lines: thời điểm, người dùng, chức năng, văn bản được trả (không ghi câu hỏi)."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, principal: Principal, model: str, doc_ids: List[str], outcome: str) -> None:
        if self.path is None:
            return
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "email": principal.email, "phong": principal.phong,
               "vai_tro": principal.vai_tro, "model": model, "outcome": outcome, "doc_ids": doc_ids}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"Không ghi được nhật ký truy cập: {e}")


def _chunk(cid: str, model: str, content: Optional[str] = None, finish: Optional[str] = None) -> str:
    delta = {"role": "assistant", "content": content} if content is not None else {}
    payload = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(assistant_factory, api_key: str, links: Optional[DocumentLinks] = None,
               policy: Union[AccessPolicy, PolicySource, None] = None, access_log: Optional[AccessLog] = None,
               user_header: str = "x-user-email", link_ttl_hours: float = 24,
               registrations: Optional[Registrations] = None) -> Starlette:
    """policy: bảng phân quyền cố định, hoặc PolicySource (tự nạp lại khi file cấu hình đổi, vd sau khi duyệt khai báo).
    registrations: bật lệnh /phongban, /choduyet, /duyet, /tuchoi và hướng dẫn khai báo cho người dùng mới."""
    source = policy if isinstance(policy, PolicySource) else None
    fixed = policy if isinstance(policy, AccessPolicy) else AccessPolicy(enabled=False)
    current_policy = source.get if source else (lambda: fixed)
    access_log = access_log or AccessLog(None)

    def email_of(request: Request) -> str:
        email = request.headers.get(user_header, "").strip().lower()
        return "" if "{{" in email else email   # LibreChat chưa thay biến (cấu hình sai) → coi như không rõ người dùng

    def account_reply(pol: AccessPolicy, email: str, principal: Principal, text: str) -> Optional[Tuple[str, str]]:
        """Trả (nội dung, kết quả ghi nhật ký) khi tin nhắn là lệnh tài khoản hoặc người dùng chưa được duyệt."""
        command = re.match(r"^/(\w+)\s*(.*)$", text, re.S)
        if registrations is not None and command and command.group(1).lower() in COMMANDS:
            res = handle_command(command.group(1).lower(), command.group(2).strip(), email, pol, registrations)
            return res.text, res.outcome
        if not pol.enabled or email in pol.users:
            return None
        if registrations is None or not email:
            return (NOT_REGISTERED, "tu_choi") if not principal.features else None
        req = registrations.get(email)
        if req is None or req["trang_thai"] != CHO_DUYET:
            return status_text(pol, req, email), "chua_khai_bao"
        if not principal.features:                 # đang chờ duyệt, cấu hình không cho xem tài liệu chung
            return status_text(pol, req, email), "cho_duyet"
        return None

    def can_open(user: str, doc_id: str) -> bool:
        principal = current_policy().by_key(user)
        if principal is None or not principal.features:
            return False
        store = assistant_factory().store
        return access_allows(principal.scope, doc_id, store.access_labels([doc_id]).get(doc_id, []))
    def authorized(request: Request) -> bool:
        if not api_key:
            return True
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        return hmac.compare_digest(token.encode(), api_key.encode())

    async def models(request: Request):
        if not authorized(request):
            return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)
        return JSONResponse({"object": "list", "data": [
            {"id": m, "object": "model", "owned_by": "hanoi-metro", "description": d} for m, d in MODELS.items()
        ]})

    async def chat(request: Request):
        if not authorized(request):
            return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse({"error": {"message": "Body phải là JSON UTF-8 hợp lệ"}}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "Body phải là object JSON"}}, status_code=400)
        model = body.get("model") if body.get("model") in MODELS else "hanoi-metro-hoi-dap"
        turns = parse_turns(body.get("messages") or [])
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if not turns or turns[-1].role != "user":
            turns.append(Turn("user", ""))

        pol = current_policy()
        email = email_of(request)
        principal = pol.resolve(email)

        def generate() -> Iterator[str]:
            reply = account_reply(pol, email, principal, turns[-1].text)
            if reply is not None:                   # tra bảng, không gọi LLM
                access_log.write(principal, model, [], reply[1])
                yield reply[0]
                return
            assistant = assistant_factory()
            if not pol.enabled:
                filters, features = None, None
            else:
                filters, features = Filters(access=principal.scope), principal.features
                # bản sao nhẹ cho riêng yêu cầu này: liên kết mở file gắn với người hỏi
                assistant = copy.copy(assistant)
                if links is not None:
                    ttl = int(link_ttl_hours * 3600)
                    assistant.linker = lambda doc_id, name, pages: links.url(doc_id, name, pages, principal.user_key, ttl)
            try:
                plan = build_plan(assistant, model, turns, filters, features)
                if pol.enabled:
                    access_log.write(principal, model, plan.doc_ids, "ok")
                yield from assistant.run_stream(plan)
            except Exception as e:  # lỗi giữa chừng: báo bằng lời, không lộ chi tiết kỹ thuật
                yield "\n\n" + user_message(e)

        if body.get("stream"):
            async def events():
                async for piece in iterate_in_threadpool(generate()):
                    if piece:
                        yield _chunk(cid, model, piece)
                yield _chunk(cid, model, finish="stop")
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        text = await run_in_threadpool(lambda: "".join(generate()))
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    async def document(request: Request):
        """Mở file gốc từ liên kết trong danh sách nguồn (trình duyệt tự nhảy tới trang nhờ #page=N)."""
        if links is None:
            return PlainTextResponse("Chức năng mở tài liệu gốc đang tắt.", status_code=404)
        doc_id = request.path_params["doc_id"].lower()
        q = request.query_params
        status, path, name = await run_in_threadpool(
            links.resolve, assistant_factory().store, doc_id, q.get("k", ""), q.get("u", ""), q.get("e", ""),
            can_open if current_policy().enabled else None)
        if status == 403:
            return PlainTextResponse("Liên kết không hợp lệ hoặc tài liệu không được phép mở.", status_code=403)
        if status != 200 or path is None:
            return PlainTextResponse("Không tìm thấy file gốc của tài liệu này trên máy chủ.", status_code=404)
        return FileResponse(path, media_type=media_type(path), filename=name, content_disposition_type="inline",
                            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

    return Starlette(routes=[
        Route("/v1/models", models, methods=["GET"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/tai-lieu/{doc_id}/{name}", document, methods=["GET"]),
    ])


def document_links(blocklist, host: str, port: int) -> Optional[DocumentLinks]:
    cfg = settings.agent
    roots = [Path(r) if Path(r).is_absolute() else settings.project_root / r for r in cfg.doc_roots.split(";") if r.strip()]
    base = cfg.doc_link_base_url or f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    return DocumentLinks.create(cfg.doc_link_secret, base, roots, blocklist.is_blocked)


def main() -> None:
    from src.runtime import build_runtime
    parser = argparse.ArgumentParser(description="Hanoi Metro one-pass chat gateway")
    parser.add_argument("--host", default=settings.agent.gateway_host)
    parser.add_argument("--port", type=int, default=settings.agent.gateway_port)
    args = parser.parse_args()
    if not settings.agent.gateway_api_key:
        logger.warning("GATEWAY_API_KEY trống: dịch vụ không kiểm tra khóa. Đặt GATEWAY_API_KEY trong .env.")
    rt = build_runtime()
    assistant = rt.assistant  # nạp sẵn mô hình, kết nối Qdrant trước khi nhận yêu cầu
    if not rt.blocklist.configured and getattr(assistant.llm, "external", False):
        logger.error(f"Danh sách chặn tài liệu mật chưa cấu hình ({rt.blocklist.problem}): mọi câu hỏi cần gọi LLM "
                     "bên ngoài sẽ bị từ chối cho tới khi đặt BLOCKLIST_DIR đúng.")
    cfg = settings.agent
    policy_dir = Path(cfg.access_config_dir)
    policy_dir = policy_dir if policy_dir.is_absolute() else settings.project_root / policy_dir
    policy = PolicySource(policy_dir, cfg.access_control)
    policy.get()                                             # nạp và ghi lỗi cấu hình (nếu có) ngay khi khởi động
    reg_path = Path(cfg.access_registrations)
    registrations = Registrations(reg_path if reg_path.is_absolute() else settings.project_root / reg_path,
                                  policy_dir / "nguoi_dung.csv")
    log_path = Path(cfg.access_log)
    access_log = AccessLog(log_path if log_path.is_absolute() else settings.project_root / log_path)
    if not cfg.access_control:
        logger.info("Phân quyền theo phòng ban đang TẮT (ACCESS_CONTROL=off): mọi người dùng xem toàn bộ kho.")
    links = document_links(rt.blocklist, args.host, args.port)
    if links:
        assistant.linker = links.url
        logger.info(f"Mở tài liệu gốc từ trích dẫn: {links.base_url}/tai-lieu/… (thư mục: {', '.join(map(str, links.roots))})")
    else:
        logger.warning("Không có DOC_LINK_SECRET/GATEWAY_API_KEY: trích dẫn hiển thị dạng chữ, không mở được file gốc.")
    uvicorn.run(create_app(lambda: rt.assistant, settings.agent.gateway_api_key, links, policy, access_log,
                           cfg.access_user_header.lower(), cfg.doc_link_ttl_hours, registrations),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
