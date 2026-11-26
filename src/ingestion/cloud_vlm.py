"""
Hanoi Metro AI Agent - Cloud VLM Parser (Luồng 3 qua API: Gemini 2.5 Flash, tắt thinking).
Kiểm soát theo Mục 6 Báo cáo dự toán chi phí VLM:
- Tài liệu có dấu chỉ độ mật: kiểm tra lại danh sách chặn ngay trước MỖI lần gửi (không chỉ lúc lập danh sách).
- Mỗi trang chỉ gọi một lần: kết quả lưu theo (md5 file, số trang, model, phiên bản prompt); gián đoạn mạng hay
  chạy lại đều không gọi lại trang đã có kết quả.
- Hạn mức chặn ngay trong pipeline: cộng dồn token và chi phí từ phản hồi API, dừng TRƯỚC khi gửi trang làm vượt
  hạn mức (ngân sách Google Cloud mặc định chỉ cảnh báo, không tự chặn chi tiêu).
- Chỉ gửi ảnh từng trang Luồng 3, không gửi cả file.
Số token thực tế ghi lại để thay định mức giả định 4.300 vào / 600 ra trong báo cáo (`python -m src.ingestion.cloud_vlm`).
"""

import base64
import re
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional

import fitz

from src.config import settings
from src.agent.llm import token_limit_kwargs
from src.indexing.indexer import Blocklist
from src.text.normalize import collapse_repetitions

logger = logging.getLogger("hanoi_metro.cloud_vlm")

PROMPT_VERSION = "cloud-v3"

CLOUD_VLM_SYSTEM_PROMPT = """Bạn là chuyên gia số hóa tài liệu kỹ thuật và hành chính của Công ty TNHH MTV Đường sắt Hà Nội (Hanoi Metro).
Chuyển ảnh MỘT trang tài liệu thành văn bản, tuân thủ bắt buộc:
1. Chép trung thực toàn bộ chữ trên trang theo thứ tự đọc; giữ nguyên số hiệu, ngày tháng, số liệu, tên người, mã thiết bị, số serial. Không sửa, không suy đoán, không tóm tắt.
2. Đề mục dùng Markdown (#, ##, ###). Giữ nguyên đánh số I., 1., 1.1, a), Điều, Chương.
3. Bảng KHÔNG có ô gộp: bảng Markdown (| cột |). Bảng CÓ ô gộp (gộp hàng/cột): xuất HTML <table> với rowspan/colspan đúng cấu trúc. Giữ đủ mọi hàng có dữ liệu, cột, đơn vị, thành tiền. BỎ QUA các hàng trống hoàn toàn (biểu mẫu chưa điền); không bao giờ lặp lại một dòng.
4. Chữ viết tay: [Viết tay: ...]. Chữ ký: [Chữ ký: họ tên nếu đọc được]. Con dấu: [Dấu: nội dung đọc được]. Chữ không đọc được: [không rõ].
5. Thuật ngữ thường gặp: AFC (TVM, AG, BOM, PCA), OCC, CMMS, Tuyến 2A (Cát Linh – Hà Đông), Tuyến 3.1 (Nhổn – Ga Hà Nội), Cisco, HPE, Dell, Fortigate, APC.
6. Chỉ xuất nội dung trang. Không thêm lời chào, giải thích, không bọc trong ```."""


class BudgetExceeded(RuntimeError):
    """Chạm hạn mức chi phí VLM đã duyệt: pipeline phải dừng, không xử lý tiếp."""


class VLMUnavailable(RuntimeError):
    """API VLM lỗi tạm thời (429, 5xx, mạng) sau khi đã thử lại: file phải đánh dấu thất bại để lần sau chạy lại,
    không được thay trang bằng chữ thô rồi coi như xong."""


@dataclass
class PageResult:
    markdown: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    cached: bool


class VLMLedger:
    """Sổ ghi kết quả và chi phí từng trang (SQLite). Là bộ nhớ đệm 'mỗi trang gọi một lần' và căn cứ hạn mức."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS vlm_pages (
        md5 TEXT NOT NULL, page INTEGER NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
        markdown TEXT NOT NULL, in_tokens INTEGER, out_tokens INTEGER, cost_usd REAL, created_at TEXT,
        PRIMARY KEY (md5, page, model, prompt_version)
    );
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(self._SCHEMA)
            if "raw" not in {r[1] for r in c.execute("PRAGMA table_info(vlm_pages)")}:
                c.execute("ALTER TABLE vlm_pages ADD COLUMN raw TEXT")   # đầu ra gốc của API, trước hậu xử lý

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, md5: str, page: int, model: str) -> Optional[tuple]:
        """(markdown, token vào, token ra, chi phí) đã ghi cho trang, hoặc None."""
        with self._conn() as c:
            return c.execute("SELECT markdown, in_tokens, out_tokens, cost_usd FROM vlm_pages "
                             "WHERE md5=? AND page=? AND model=? AND prompt_version=?",
                             (md5, page, model, PROMPT_VERSION)).fetchone()

    def record(self, md5: str, page: int, model: str, markdown: str, in_tokens: int, out_tokens: int, cost: float,
               raw: Optional[str] = None) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO vlm_pages (md5, page, model, prompt_version, markdown, in_tokens, out_tokens, "
                      "cost_usd, created_at, raw) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (md5, page, model, PROMPT_VERSION, markdown, in_tokens, out_tokens, cost,
                       datetime.now().isoformat(timespec="seconds"), raw))

    def reprocess(self, fn) -> int:
        """Áp lại hậu xử lý lên đầu ra gốc đã lưu (không gọi API). Trả về số trang thay đổi."""
        changed = 0
        with self._lock, self._conn() as c:
            for rowid, raw, md in c.execute("SELECT rowid, raw, markdown FROM vlm_pages WHERE raw IS NOT NULL").fetchall():
                new = fn(raw)
                if new != md:
                    c.execute("UPDATE vlm_pages SET markdown=? WHERE rowid=?", (new, rowid))
                    changed += 1
        return changed

    def done_pages(self, model: str) -> set:
        """Mọi (md5, trang) đã có kết quả với model + prompt hiện tại (một truy vấn thay cho hàng chục nghìn lần get)."""
        with self._conn() as c:
            return {(m, p) for m, p in c.execute("SELECT md5, page FROM vlm_pages WHERE model=? AND prompt_version=?",
                                                  (model, PROMPT_VERSION))}

    def spent_usd(self) -> float:
        with self._conn() as c:
            return float(c.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM vlm_pages").fetchone()[0])

    def stats(self, current_prompt_only: bool = False) -> dict:
        """Tổng chi (mọi lần chạy) dùng cho hạn mức; current_prompt_only=True để đo token/trang của prompt đang dùng."""
        where, args = ("WHERE prompt_version = ?", (PROMPT_VERSION,)) if current_prompt_only else ("", ())
        with self._conn() as c:
            n, tin, tout, cost = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(in_tokens),0), COALESCE(SUM(out_tokens),0), COALESCE(SUM(cost_usd),0) FROM vlm_pages " + where, args
            ).fetchone()
        return {"pages": n, "in_tokens": tin, "out_tokens": tout, "cost_usd": cost,
                "avg_in": tin / n if n else 0, "avg_out": tout / n if n else 0}


class CloudVLMParser:
    def __init__(self, client=None, blocklist: Optional[Blocklist] = None, ledger: Optional[VLMLedger] = None):
        cfg = settings.vlm
        self.model = cfg.model_name
        self.max_tokens = cfg.max_tokens
        self.timeout = float(cfg.timeout_seconds)
        self.dpi = settings.ingestion.image_dpi_vlm
        self.strips = max(1, cfg.strips)
        self.image_detail = cfg.image_detail
        self.base_url = cfg.base_url
        self.max_attempts = 6
        self.retry_base_seconds = 10.0
        self._sleep = time.sleep
        self.budget = cfg.budget_usd
        self.price_in = cfg.price_in_per_m
        self.price_out = cfg.price_out_per_m
        self.reasoning_effort = cfg.reasoning_effort
        self.pending_usd = 0.0        # chi phí ước tính của các trang đang nằm trong Batch chưa có kết quả
        self.client = client or settings.get_vlm_client()
        self.blocklist = blocklist or Blocklist.from_dir(settings.retrieval.blocklist_dir)
        self.ledger = ledger or VLMLedger(settings.ingestion.state_db_path.with_name("vlm_ledger.db"))
        if not getattr(self.blocklist, "configured", True):
            logger.error(f"Danh sách chặn tài liệu mật chưa cấu hình ({self.blocklist.problem}): mọi trang gửi API sẽ bị từ chối.")
        elif self.blocklist.blocked == set():
            logger.warning("Danh sách chặn tài liệu mật đang rỗng: kiểm tra lại kết quả page_router trước khi gửi trang ra API.")

    def _cost(self, in_tokens: int, out_tokens: int) -> float:
        return in_tokens / 1e6 * self.price_in + out_tokens / 1e6 * self.price_out

    def _image_url(self, b64: str) -> dict:
        url = {"url": f"data:image/png;base64,{b64}"}
        if self.image_detail:
            url["detail"] = self.image_detail   # OpenAI: "high" giữ độ phân giải để đọc chữ viết tay
        return url

    def _create_with_retry(self, **kwargs):
        """Thử lại khi 429 / 5xx / lỗi mạng, giãn cách tăng dần (ưu tiên Retry-After của máy chủ)."""
        delay = self.retry_base_seconds
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                status = getattr(e, "status_code", None)
                transient = status in (408, 429, 500, 502, 503, 504) or type(e).__name__ in ("APIConnectionError", "APITimeoutError")
                if not transient:
                    raise
                if attempt == self.max_attempts:
                    raise VLMUnavailable(f"API VLM lỗi {status or type(e).__name__} sau {attempt} lần thử") from e
                headers = getattr(getattr(e, "response", None), "headers", None) or {}
                wait = float(headers.get("retry-after", delay)) if str(headers.get("retry-after", "")).replace(".", "").isdigit() else delay
                logger.warning(f"API VLM lỗi {status or type(e).__name__}, thử lại sau {wait:.0f}s (lần {attempt}/{self.max_attempts}).")
                self._sleep(wait)
                delay = min(delay * 2, 120)

    def reserve_cost(self) -> float:
        """Chi phí dự phòng cho một trang sắp gửi: lớn hơn giữa định mức báo cáo (4.300 vào / 1.200 ra) và TB đã đo."""
        stats = self.ledger.stats()
        return max(self._cost(4300, 1200), stats["cost_usd"] / stats["pages"] if stats["pages"] else 0.0)

    def check_budget(self, n_pages: int, discount: float = 1.0) -> None:
        """Ném BudgetExceeded nếu gửi thêm n_pages trang có thể vượt VLM_BUDGET_USD (tính cả trang đang chờ Batch)."""
        spent = self.ledger.spent_usd() + self.pending_usd
        need = n_pages * self.reserve_cost() * discount
        if spent + need > self.budget:
            raise BudgetExceeded(f"Đã chi/đang chờ {spent:.4f} USD; gửi thêm {n_pages} trang cần ~{need:.4f} USD "
                                 f"vượt hạn mức VLM_BUDGET_USD={self.budget} USD.")

    def request_body(self, page: Any, page_num: int) -> dict:
        """Tham số một yêu cầu bóc tách trang (dùng chung cho gọi trực tiếp và Batch API)."""
        images = render_strips(page, self.strips, self.dpi)
        instruction = (f"Trang {page_num}. Chép toàn bộ nội dung trang theo quy tắc." if len(images) == 1 else
                       f"Trang {page_num} được cắt thành {len(images)} dải ngang theo thứ tự từ trên xuống, các dải chồng lấn nhẹ. "
                       "Ghép lại thành MỘT trang liền mạch, không lặp phần chồng lấn. Chép toàn bộ nội dung theo quy tắc.")
        body = {
            "model": self.model,
            "temperature": 0.0,
            **token_limit_kwargs(self.base_url, self.max_tokens),
            "messages": [
                {"role": "system", "content": CLOUD_VLM_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": instruction}] + [
                    {"type": "image_url", "image_url": self._image_url(img)} for img in images
                ]},
            ],
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def record_external(self, doc_md5: str, page_num: int, raw_text: str, in_tokens: int, out_tokens: int, discount: float) -> PageResult:
        """Ghi kết quả nhận từ Batch API vào sổ (giá Batch = giá thường × discount)."""
        text = clean_vlm_output(raw_text)
        cost = self._cost(in_tokens, out_tokens) * discount
        self.ledger.record(doc_md5, page_num, self.model, text, in_tokens, out_tokens, cost, raw=raw_text)
        return PageResult(text, in_tokens, out_tokens, cost, False)

    def parse(self, page: Any, page_num: int, pdf_path: Optional[str] = None, doc_md5: Optional[str] = None) -> str:
        return self.parse_page(page, page_num, doc_md5).markdown

    def parse_page(self, page: Any, page_num: int, doc_md5: Optional[str]) -> PageResult:
        if not doc_md5:
            raise ValueError("CloudVLMParser cần md5 của file để kiểm tra danh sách chặn và lưu kết quả")
        self.blocklist.assert_allowed([doc_md5])          # kiểm tra lại ngay trước khi gửi

        cached = self.ledger.get(doc_md5, page_num, self.model)
        if cached is not None:
            return PageResult(cached[0], cached[1] or 0, cached[2] or 0, cached[3] or 0.0, True)

        self.check_budget(1)
        body = self.request_body(page, page_num)
        n_images = len(body["messages"][1]["content"]) - 1
        stream = self._create_with_retry(timeout=self.timeout, stream=True, stream_options={"include_usage": True}, **body)
        parts: List[str] = []
        usage, finish, looped = None, None, False
        for event in stream:
            if getattr(event, "usage", None):
                usage = event.usage
            if event.choices:
                choice = event.choices[0]
                finish = choice.finish_reason or finish
                if choice.delta and choice.delta.content:
                    parts.append(choice.delta.content)
                    if len(parts) % 20 == 0 and is_looping("".join(parts)):
                        looped = True
                        break  # ngắt kết nối: không trả tiền cho phần lặp vô ích
        if looped and hasattr(stream, "close"):
            stream.close()
        text = clean_vlm_output("".join(parts))
        if usage and not looped:
            tin = int(usage.prompt_tokens or 0)
            tout = int(usage.completion_tokens or 0)
        else:
            # bị ngắt giữa chừng: API không gửi usage, ước tính theo ảnh + độ dài đã sinh (tính dư cho an toàn hạn mức)
            tin = 400 + 260 * n_images
            tout = int(len("".join(parts)) / 2.5)
        cost = self._cost(tin, tout)
        if looped:
            logger.warning(f"Trang {page_num}: VLM lặp dòng, đã ngắt sớm và gộp phần lặp (ước tính {tout} token ra).")
        elif finish == "length":
            logger.warning(f"Trang {page_num}: đầu ra chạm VLM_MAX_TOKENS={self.max_tokens}, có thể bị cụt.")
        self.ledger.record(doc_md5, page_num, self.model, text, tin, tout, cost, raw="".join(parts))
        logger.info(f"[CloudVLM] {doc_md5[:8]} trang {page_num}: {tin} token vào, {tout} token ra, {cost:.5f} USD")
        return PageResult(text, tin, tout, cost, False)


_RX_INLINE_REPEAT = re.compile(r"(.{3,300}?)\1{4,}", re.S)   # câu lặp trong ô bảng có thể dài > 100 ký tự
_RX_SPACE_RUN = re.compile(r"[ \t]{200,}$")
_RX_MARKUP = re.compile(r"<[^>]*>|[|\-:=*_#\s]")
TABLE_LOOP_REPEATS = 20     # bảng thật có thể có 5–6 ô "BT" liền nhau; vòng lặp thật dài hàng trăm lần


def _is_table_line(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("|") or "<td" in s or "<th" in s or "<tr" in s


def _meaningful(unit: str) -> bool:
    """Cụm lặp có ít nhất 2 chữ/số thật (không tính "|---", "<td></td>", khoảng trắng): mới có thể là vòng lặp."""
    return len(_RX_MARKUP.sub("", unit)) >= 2


def _loop_runs(line: str, min_repeats: int):
    for m in _RX_INLINE_REPEAT.finditer(line):
        unit = m.group(1)
        if _meaningful(unit) and len(m.group(0)) // len(unit) >= min_repeats:
            yield m


def collapse_inline_repeats(text: str) -> str:
    """
    Gộp cụm lặp trong một dòng ("quận Bắc Từ Liêm, " × 40 -> một lần): ngoài bảng khi lặp >= 5 lần, trong dòng bảng
    khi lặp >= 20 lần. Không đụng tới dòng phân cách "|---|---|", ô trống "<td></td>" lặp lại, khoảng trắng canh lề.
    """
    out = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]{200,}", " ", line)
        need = TABLE_LOOP_REPEATS if _is_table_line(line) else 5
        for m in sorted(_loop_runs(line, need), key=lambda m: -m.start()):
            line = line[:m.start()] + m.group(1) + line[m.end():]
        out.append(line)
    return "\n".join(out)


def clean_vlm_output(raw: str) -> str:
    """Hậu xử lý chung: gộp dòng lặp, gộp cụm lặp trong dòng, bỏ ``` bao ngoài."""
    return _strip_fences(collapse_inline_repeats("\n".join(collapse_repetitions(raw.split("\n"), min_repeats=3))))


def is_looping(text: str, repeats: int = 12) -> bool:
    """
    Dấu hiệu VLM rơi vào vòng lặp: đuôi văn bản là một dòng (hoặc nhóm 2–3 dòng) lặp >= `repeats` lần, hoặc một cụm
    3–80 ký tự lặp >= 8 lần liền trong cùng dòng, hoặc chuỗi khoảng trắng rất dài (đo trên 200 trang thật: 4/200 trang
    lặp kiểu này tới hết trần token mà không xuống dòng).
    """
    tail = text[-2400:]
    if _RX_SPACE_RUN.search(tail):
        return True
    last = tail.rstrip().split("\n")[-1]
    need = TABLE_LOOP_REPEATS if _is_table_line(last) else 8
    if any(m.end() >= len(last) - len(m.group(1)) for m in _loop_runs(last, need)):
        return True
    lines = [ln.strip() for ln in text.rstrip().split("\n")][-repeats * 3:]
    for size in (1, 2, 3):
        tail = lines[-size * repeats:]
        if len(tail) == size * repeats and all(tail[i] == tail[i % size] for i in range(len(tail))):
            return True
    return False


def render_strips(page: Any, n: int, dpi: int, overlap: float = 0.04) -> List[str]:
    """
    Cắt trang thành n dải ngang chồng lấn nhẹ. Gemini tính mỗi ảnh khoảng 258 token (ảnh bị thu nhỏ), nên một ảnh
    cả trang làm chữ số viết tay dễ đọc sai; đo trên biên bản viết tay: 1 ảnh đọc sai ngày 2/2 lần, 3 dải đọc đúng 2/2.
    """
    r = page.rect
    if n <= 1:
        return [base64.b64encode(page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")).decode()]
    h = r.height / n
    out = []
    for i in range(n):
        y0 = max(0.0, i * h - overlap * r.height)
        y1 = min(r.height, (i + 1) * h + overlap * r.height)
        pix = page.get_pixmap(dpi=dpi, alpha=False, clip=fitz.Rect(r.x0, r.y0 + y0, r.x1, r.y0 + y1))
        out.append(base64.b64encode(pix.tobytes("png")).decode())
    return out


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ledger = VLMLedger(settings.ingestion.state_db_path.with_name("vlm_ledger.db"))
    total = ledger.stats()
    s = ledger.stats(current_prompt_only=True)
    cfg = settings.vlm
    print(f"Đã chi tổng cộng {total['cost_usd']:.4f} USD / hạn mức {cfg.budget_usd} USD ({total['pages']} lượt gọi API, mọi phiên bản prompt)")
    print(f"Prompt hiện tại ({PROMPT_VERSION}): {s['pages']} trang | token vào TB {s['avg_in']:.0f}, ra TB {s['avg_out']:.0f}")
    if s["pages"]:
        for label, pages in (("Kịch bản 1", 52661), ("Kịch bản 2", 64000), ("Kịch bản 3", 111011)):
            usd = pages * (s["avg_in"] / 1e6 * cfg.price_in_per_m + s["avg_out"] / 1e6 * cfg.price_out_per_m)
            print(f"  {label} ({pages:,} trang) theo token thực đo: ≈ {usd:,.2f} USD (Batch API ≈ {usd / 2:,.2f} USD)")


if __name__ == "__main__":
    main()
