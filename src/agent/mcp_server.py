"""
Hanoi Metro AI Agent - MCP Server cho LibreChat.
Đưa các chức năng của trợ lý thành công cụ MCP để agent trong LibreChat gọi:
  search_documents, ask_documents, summarize_document, draft_report, list_documents, read_document.
Chạy:  python -m src.agent.mcp_server              (streamable-http, MCP_HOST:MCP_PORT/mcp)
       python -m src.agent.mcp_server --stdio
Lỗi trả về người dùng là thông báo đã kiểm soát (không lộ exception, đường dẫn, câu truy vấn nội bộ).
"""

import argparse
import logging
from typing import Callable, List, Literal, Optional, TypeVar

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from src.agent.assistant import Answer, format_listing
from src.agent.errors import user_message
from src.config import settings
from src.runtime import build_runtime
from src.store.base import Filters

logger = logging.getLogger("hanoi_metro.mcp")
T = TypeVar("T")

runtime = build_runtime()

server = MCPServer(
    name="hanoi-metro-docs",
    title="Tài liệu nội bộ Đường sắt Hà Nội",
    instructions=(
        "Công cụ tra cứu kho tài liệu nội bộ Công ty TNHH MTV Đường sắt Hà Nội (công văn, quyết định, biên bản, "
        "quy trình vận hành – bảo trì, định mức...). Dùng search_documents để lấy đoạn trích kèm nguồn rồi tự trả lời, "
        "hoặc ask_documents để nhận câu trả lời có trích dẫn. Luôn giữ trích dẫn [n] và danh sách nguồn khi trả lời người dùng. "
        "Dùng list_documents cho câu hỏi liệt kê / thống kê, summarize_document để tóm tắt, draft_report để soạn dự thảo."
    ),
)

async def _run(fn: Callable[[], T]) -> T:
    return await anyio.to_thread.run_sync(fn)


def _filters(loai_vb: Optional[List[str]], year_from: Optional[int], year_to: Optional[int]) -> Filters:
    return Filters(loai_vb=loai_vb or None, year_from=year_from, year_to=year_to)


def _render(answer: Answer) -> str:
    notes = "\n".join(f"> {n}" for n in answer.notes)
    return f"{notes}\n\n{answer.text}".strip() if notes else answer.text


MCP_LOCKED = ("Công cụ tra cứu tài liệu qua MCP đang khóa vì hệ thống đã bật phân quyền theo phòng ban "
              "(MCP chưa nhận diện người dùng). Hãy dùng trợ lý \"Hỏi đáp tài liệu Hanoi Metro\" trong LibreChat.")


async def _guard(fn: Callable[[], str]) -> str:
    if settings.agent.access_control:      # chặn khi chưa kiểm soát được: không biết ai gọi thì không trả tài liệu
        return MCP_LOCKED
    try:
        return await _run(fn)
    except Exception as e:
        return user_message(e)


@server.tool(description=(
    "Tìm đoạn tài liệu nội bộ liên quan (tìm kiếm lai ngữ nghĩa + từ khóa, khớp chính xác số hiệu như 916/CV-VT&QLPT, "
    "mã thiết bị, viết tắt). Trả về các đoạn trích đánh số [n] kèm file và trang để trích dẫn."
))
async def search_documents(query: str, loai_vb: Optional[List[str]] = None, year_from: Optional[int] = None,
                           year_to: Optional[int] = None, limit: int = 8) -> str:
    def work() -> str:
        res = runtime.assistant.search(query, _filters(loai_vb, year_from, year_to), k=max(1, min(limit, 20)))
        if not res.evidence:
            return "Không tìm thấy đoạn tài liệu nào liên quan."
        parts = [
            f"[{ev.rank}] {ev.citation}\n{ev.chunk.context}\n{(ev.parent.text if ev.parent else ev.chunk.text)}"
            for ev in res.evidence
        ]
        warn = "".join(f"> Nhánh '{b}' đang lỗi; kết quả chưa đầy đủ.\n" for b in res.degraded)
        return warn + "\n\n---\n\n".join(parts)
    return await _guard(work)


@server.tool(description="Hỏi đáp trên kho tài liệu nội bộ; trả lời bằng LLM đã cấu hình (hiện qua API, có kiểm tra danh sách chặn tài liệu mật), có trích dẫn [n] và danh sách nguồn (file, trang).")
async def ask_documents(question: str, loai_vb: Optional[List[str]] = None, year_from: Optional[int] = None,
                        year_to: Optional[int] = None) -> str:
    return await _guard(lambda: _render(runtime.assistant.ask(question, _filters(loai_vb, year_from, year_to))))


@server.tool(description=(
    "Tóm tắt một văn bản. `document`: số hiệu (vd 916/CV-VT&QLPT), doc_id hoặc mô tả (vd 'biên bản kiểm đếm thiết bị DC Duy Tân'). "
    "`style`: ngan | gach_dau_dong | chi_tiet. `focus`: trọng tâm cần tóm tắt (tùy chọn)."
))
async def summarize_document(document: str, style: Literal["ngan", "gach_dau_dong", "chi_tiet"] = "chi_tiet", focus: str = "") -> str:
    return await _guard(lambda: _render(runtime.assistant.summarize(document, style, focus)))


@server.tool(description=(
    "Soạn dự thảo văn bản từ tài liệu nội bộ theo thể thức Nghị định 30. `report_type`: bao_cao | to_trinh | cong_van | tong_hop. "
    "`documents`: danh sách số hiệu/doc_id làm căn cứ (bỏ trống để tự tìm theo chủ đề). Chỗ thiếu thông tin ghi [cần bổ sung]."
))
async def draft_report(topic: str, report_type: Literal["bao_cao", "to_trinh", "cong_van", "tong_hop"] = "bao_cao",
                       documents: Optional[List[str]] = None, recipient: str = "", unit: str = "", instructions: str = "") -> str:
    return await _guard(lambda: _render(runtime.assistant.draft_report(topic, report_type, documents, recipient, unit, instructions)))


@server.tool(description=(
    "Liệt kê ĐẦY ĐỦ văn bản theo điều kiện (không giới hạn top-k): loại văn bản (CV, QĐ, TB, BC, TTr, BB, KH...), "
    "khoảng năm, từ khóa trong số hiệu / trích yếu / tên file."
))
async def list_documents(keyword: str = "", loai_vb: Optional[List[str]] = None, year_from: Optional[int] = None,
                         year_to: Optional[int] = None, limit: int = 50) -> str:
    def work() -> str:
        f = _filters(loai_vb, year_from, year_to)
        docs, total = runtime.store.list_documents(f, keyword=keyword, limit=max(1, min(limit, 500)))
        return format_listing(docs, total)
    return await _guard(work)


@server.tool(description="Đọc nguyên văn một văn bản (metadata + nội dung theo trang). `document`: số hiệu, doc_id hoặc mô tả.")
async def read_document(document: str, page_from: Optional[int] = None, page_to: Optional[int] = None) -> str:
    return await _guard(lambda: _render(runtime.assistant.read_document(document, page_from, page_to)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hanoi Metro MCP server")
    parser.add_argument("--stdio", action="store_true", help="Chạy qua stdio thay vì streamable-http")
    parser.add_argument("--host", default=settings.agent.mcp_host)
    parser.add_argument("--port", type=int, default=settings.agent.mcp_port)
    args = parser.parse_args()
    if args.stdio:
        server.run("stdio")
    else:
        hosts = [h.strip() for h in settings.agent.mcp_allowed_hosts.split(",") if h.strip()]
        security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts)
        logger.info(f"MCP server: http://{args.host}:{args.port}/mcp (Host hợp lệ: {', '.join(hosts)})")
        server.run("streamable-http", host=args.host, port=args.port, transport_security=security)


if __name__ == "__main__":
    main()
