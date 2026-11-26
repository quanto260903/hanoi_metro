"""
Hanoi Metro AI Agent - Corpus Ingestion theo kết quả page_router.
Bóc tách toàn kho bằng chính danh sách đầu vào và luồng từng trang mà page_router đã lập:
  - ingest_manifest.csv: file được phép (đã loại trùng, đã loại tài liệu có dấu chỉ độ mật)
  - pages_routed.csv:    luồng từng trang (DIRECT_TEXT / STANDARD_OCR / VLM / BLANK), số trang tính từ 0
Luồng xử lý:
  - DIRECT_TEXT: PyMuPDF, bảng điện tử lấy bằng find_tables (text nằm trong bảng không bị chép hai lần).
  - VLM và STANDARD_OCR: VLM API (gọi trực tiếp hoặc Batch API giảm 50%). RapidOCR mặc định làm mất dấu tiếng Việt
    ("CONGTYTNHHMTVDUONGSATHANOI") và mất ~6 giây/trang trên CPU, nên trang OCR mặc định cũng đi VLM
    (CORPUS_OCR_VIA=rapidocr để dùng lại RapidOCR).
  - BLANK: bỏ qua.
Kết quả lưu theo từng trang (data/corpus.db cho trang text, sổ VLM cho trang API); file chỉ được ghép thành Markdown
khi đủ mọi trang, nên dừng giữa chừng lúc nào cũng chạy tiếp được, trang đã xong không bị tính tiền lại.

  python -m src.ingestion.corpus status
  python -m src.ingestion.corpus local        [--filter "VĂN BẢN SCAN"] [--limit-files 50]
  python -m src.ingestion.corpus vlm-sync     [...]
  python -m src.ingestion.corpus batch-submit [--batch-pages 500] [--max-batches 1]
  python -m src.ingestion.corpus batch-collect
  python -m src.ingestion.corpus assemble
  python -m src.cli index --dir data/corpus_md
"""

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import fitz

from src.config import settings
from src.ingestion.cloud_vlm import BudgetExceeded, CloudVLMParser
from src.indexing.indexer import BlocklistNotConfigured

logger = logging.getLogger("hanoi_metro.corpus")

MAX_BATCH_BYTES = 180 * 1024 * 1024   # giới hạn file đầu vào Batch API là 200 MB


@dataclass
class FilePlan:
    md5: str
    path: str                                   # tương đối với thư mục gốc kho
    total_pages: int
    routes: Dict[int, str] = field(default_factory=dict)   # trang (0-based) -> luồng


def load_plan(results_dir: Path, path_filter: str = "", limit: Optional[int] = None) -> Dict[str, FilePlan]:
    plans: Dict[str, FilePlan] = {}
    rows = sorted(csv.DictReader(open(results_dir / "ingest_manifest.csv", encoding="utf-8-sig")), key=lambda r: r["path"])
    for r in rows:
        if path_filter and path_filter.lower() not in r["path"].lower():
            continue
        plans[r["md5"]] = FilePlan(r["md5"], r["path"], int(r["pages"] or 0))
        if limit and len(plans) >= limit:
            break
    for r in csv.DictReader(open(results_dir / "pages_routed.csv", encoding="utf-8-sig")):
        plan = plans.get(r["md5"])
        if plan is not None:
            plan.routes[int(r["page"])] = r["route"]
    return plans


# ------------------------------------------------------------------------------
# Trang text gốc
# ------------------------------------------------------------------------------
_RX_STRUCT_START = re.compile(r"^(?:Điều|Chương|Mục|Khoản|Phụ lục|PHỤ LỤC|[IVX]{1,5}\.|\d{1,2}(?:\.\d{1,2})*[.)]|[a-zđ]\)|[-–•+])\s")


def _join_wrapped_lines(text: str) -> str:
    """
    Nối dòng bị ngắt giữa câu (dòng sau bắt đầu chữ thường, dòng trước không kết thúc bằng dấu câu), giữ xuống dòng
    trước "Điều 2.", "1.", "a)", "-"... để bộ chunking còn nhận ra đề mục và khoản.
    """
    out: List[str] = []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if out and not _RX_STRUCT_START.match(line) and not out[-1].endswith((".", ":", ";", "!", "?")) \
                and not line[0].isupper() and not line[0].isdigit():
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def direct_text_markdown(page, tables: bool = True) -> str:
    """
    Text theo thứ tự đọc; bảng điện tử thành Markdown đúng vị trí; bỏ text nằm trong vùng bảng.
    tables=False: bỏ bước dò bảng (dùng cho trang từng làm PyMuPDF sập ở bước này), chỉ lấy text.
    """
    items: List[Tuple[float, float, str]] = []
    boxes: List[fitz.Rect] = []
    try:
        for t in (page.find_tables().tables if tables else []):
            md = t.to_markdown().strip()
            if md:
                boxes.append(fitz.Rect(t.bbox))
                items.append((t.bbox[1], t.bbox[0], md))
    except Exception as e:  # find_tables lỗi trên vài PDF hỏng: vẫn lấy text
        logger.debug(f"find_tables lỗi: {e}")
    for x0, y0, x1, y1, text, _, btype in page.get_text("blocks", sort=True):
        if btype != 0 or not text.strip():
            continue
        center = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(center in b for b in boxes):
            continue
        items.append((y0, x0, _join_wrapped_lines(text)))
    items.sort(key=lambda it: (round(it[0]), it[1]))
    return "\n\n".join(text for _, _, text in items).strip()


# ------------------------------------------------------------------------------
# Lưu trạng thái
# ------------------------------------------------------------------------------
LOCAL_FAILED = "LOCAL_FAILED"     # route lưu cho trang trích trên máy thất bại (chờ API)


class CorpusStore:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS local_pages (md5 TEXT, page INTEGER, route TEXT, markdown TEXT, PRIMARY KEY (md5, page));
    CREATE TABLE IF NOT EXISTS files (md5 TEXT PRIMARY KEY, path TEXT, status TEXT, out_path TEXT, error TEXT, updated TEXT);
    CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, input_file_id TEXT, status TEXT, n INTEGER,
                                        est_usd REAL, created TEXT, collected INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS batch_items (custom_id TEXT PRIMARY KEY, batch_id TEXT, md5 TEXT, page INTEGER, status TEXT);
    CREATE TABLE IF NOT EXISTS local_inflight (md5 TEXT, page INTEGER, mode TEXT, PRIMARY KEY (md5, page));
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(self._SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=60)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def local_pages(self, md5: str) -> Dict[int, str]:
        with self._conn() as c:
            return {p: md for p, md in c.execute("SELECT page, markdown FROM local_pages WHERE md5=?", (md5,))}

    def save_local(self, md5: str, page: int, route: str, markdown: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO local_pages VALUES (?,?,?,?)", (md5, page, route, markdown))
            c.execute("DELETE FROM local_inflight WHERE md5=? AND page=?", (md5, page))

    def mark_inflight(self, md5: str, page: int, mode: str) -> None:
        """Ghi trang sắp xử lý; nếu tiến trình sập (lỗi C trong thư viện PDF), lần chạy sau biết trang nào gây sập."""
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO local_inflight VALUES (?,?,?)", (md5, page, mode))

    def inflight(self) -> Dict[Tuple[str, int], str]:
        with self._conn() as c:
            return {(m, p): mode for m, p, mode in c.execute("SELECT md5, page, mode FROM local_inflight")}

    def failed_local_pages(self) -> Set[Tuple[str, int]]:
        """Trang không trích được trên máy (sập cả khi bỏ dò bảng) → chuyển sang API. page tính từ 0."""
        with self._conn() as c:
            return {(m, p) for m, p in c.execute("SELECT md5, page FROM local_pages WHERE route = ?", (LOCAL_FAILED,))}

    def set_file(self, md5: str, path: str, status: str, out_path: str = "", error: str = "") -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)",
                      (md5, path, status, out_path, error, datetime.now().isoformat(timespec="seconds")))

    def file_status(self) -> Dict[str, str]:
        with self._conn() as c:
            return dict(c.execute("SELECT md5, status FROM files"))

    def active_batch_pages(self) -> Set[Tuple[str, int]]:
        with self._conn() as c:
            return {(m, p) for m, p in c.execute(
                "SELECT i.md5, i.page FROM batch_items i JOIN batches b USING (batch_id) WHERE b.collected = 0")}

    def pending_batch_usd(self) -> float:
        with self._conn() as c:
            return float(c.execute("SELECT COALESCE(SUM(est_usd),0) FROM batches WHERE collected = 0").fetchone()[0])

    def add_batch(self, batch_id: str, input_file_id: str, items: List[Tuple[str, str, int]], est_usd: float) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO batches VALUES (?,?,?,?,?,?,0)",
                      (batch_id, input_file_id, "submitted", len(items), est_usd, datetime.now().isoformat(timespec="seconds")))
            c.executemany("INSERT OR REPLACE INTO batch_items VALUES (?,?,?,?,'pending')", [(cid, batch_id, m, p) for cid, m, p in items])

    def open_batches(self) -> List[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT batch_id FROM batches WHERE collected = 0 ORDER BY created")]

    def update_batch(self, batch_id: str, status: str, collected: bool = False) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE batches SET status=?, collected=? WHERE batch_id=?", (status, int(collected), batch_id))

    def set_item(self, custom_id: str, status: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE batch_items SET status=? WHERE custom_id=?", (status, custom_id))


# ------------------------------------------------------------------------------
# Điều phối
# ------------------------------------------------------------------------------
class Corpus:
    def __init__(self, root: Path, store: CorpusStore, parser: CloudVLMParser, out_dir: Path, ocr_via: str = "vlm"):
        self.root = Path(root)
        self.store = store
        self.parser = parser
        self.out_dir = Path(out_dir)
        self.ocr_via = ocr_via
        self._ocr = None

    def is_api_route(self, route: str) -> bool:
        return route == "VLM" or (route == "STANDARD_OCR" and self.ocr_via == "vlm")

    def _allowed(self, plan: FilePlan) -> bool:
        if self.parser.blocklist.is_blocked(plan.md5):
            self.store.set_file(plan.md5, plan.path, "blocked")
            return False
        return True

    # ---- trang xử lý trên máy
    def run_local(self, plans: Dict[str, FilePlan], workers: int = 4) -> Dict[str, int]:
        stats = {"pages": 0, "files": 0, "errors": 0}
        stop = threading.Event()
        # trang đang xử lý dở khi tiến trình trước sập: lần này hạ cấp (full → bỏ dò bảng → chuyển API)
        suspects = self.store.inflight()
        if suspects:
            logger.warning(f"{len(suspects)} trang đang xử lý dở khi lần chạy trước bị sập; xử lý lại ở chế độ an toàn")

        def work(plan: FilePlan) -> int:
            if stop.is_set():
                return 0
            done = self.store.local_pages(plan.md5)
            todo = [(p, r) for p, r in sorted(plan.routes.items())
                    if p not in done and not self.is_api_route(r)]
            if not todo:
                return 0
            doc = fitz.open(self.root / plan.path)
            try:
                for p, route in todo:
                    if stop.is_set():
                        break
                    mode = {None: "full", "full": "safe"}.get(suspects.get((plan.md5, p)), "failed")
                    if mode == "failed":
                        logger.warning(f"Bỏ trích trên máy '{plan.path}' tr.{p + 1} (làm thư viện PDF sập 2 lần) → chuyển API")
                        self.store.save_local(plan.md5, p, LOCAL_FAILED, "")
                        continue
                    self.store.mark_inflight(plan.md5, p, mode)
                    if route == "BLANK":
                        md = ""
                    elif route == "STANDARD_OCR":
                        md = self._rapidocr(doc[p], p + 1)
                    else:
                        md = direct_text_markdown(doc[p], tables=mode == "full")
                    self.store.save_local(plan.md5, p, route, md)
            finally:
                doc.close()
            return len(todo)

        targets = [pl for pl in plans.values() if self._allowed(pl)]
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(work, pl): pl for pl in targets}
        try:
            for fut in as_completed(futures):
                try:
                    stats["pages"] += fut.result()
                    stats["files"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    pl = futures[fut]
                    self.store.set_file(pl.md5, pl.path, "error", error=f"local: {type(e).__name__}: {e}"[:300])
                    logger.error(f"Lỗi trích text '{pl.path}': {e}")
        except KeyboardInterrupt:
            # Ctrl+C: bỏ các file chưa bắt đầu, file đang làm dừng sau trang hiện tại; trang đã lưu giữ nguyên
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)
            print(f"Đã dừng theo yêu cầu. Trang đã lưu được giữ lại; chạy lại đúng lệnh này để làm tiếp. ({stats})")
            raise SystemExit(130)
        pool.shutdown(wait=True)
        return stats

    def _rapidocr(self, page, page_num: int) -> str:
        if self._ocr is None:
            from src.ingestion.parsers import StandardOCRParser
            self._ocr = StandardOCRParser()
        return self._ocr.parse(page, page_num)

    # ---- trang gửi API
    def pending_api_pages(self, plans: Dict[str, FilePlan]) -> List[Tuple[FilePlan, int]]:
        active = self.store.active_batch_pages()
        done = self.parser.ledger.done_pages(self.parser.model)
        failed = self.store.failed_local_pages()
        out = []
        for plan in plans.values():
            if self.parser.blocklist.is_blocked(plan.md5):
                continue
            for p, route in sorted(plan.routes.items()):
                key = (plan.md5, p + 1)
                needs_api = self.is_api_route(route) or (plan.md5, p) in failed
                if needs_api and key not in active and key not in done:
                    out.append((plan, p))
        return out

    def _require_blocklist(self) -> None:
        """Fail-closed: không có danh sách chặn hợp lệ thì không gửi trang nào ra API."""
        bl = self.parser.blocklist
        if not getattr(bl, "configured", True):
            raise BlocklistNotConfigured(f"Không gửi trang ra API: danh sách chặn chưa cấu hình ({bl.problem})")

    def run_vlm_sync(self, plans: Dict[str, FilePlan], workers: int = 4) -> Dict[str, int]:
        self._require_blocklist()
        todo = self.pending_api_pages(plans)
        stats = {"pages": 0, "errors": 0, "stopped_budget": 0}
        stop = threading.Event()

        def work(plan: FilePlan, p: int) -> None:
            if stop.is_set():
                return
            doc = fitz.open(self.root / plan.path)
            try:
                self.parser.parse_page(doc[p], p + 1, plan.md5)
            finally:
                doc.close()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, plan, p) for plan, p in todo]
            for fut in as_completed(futures):
                try:
                    fut.result()
                    stats["pages"] += 1
                except BudgetExceeded as e:
                    if not stop.is_set():
                        logger.error(f"DỪNG – chạm hạn mức: {e}")
                    stop.set()
                    stats["stopped_budget"] = 1
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Lỗi VLM: {type(e).__name__}: {e}")
        return stats

    def submit_batches(self, plans: Dict[str, FilePlan], client, batch_pages: int = 500, max_batches: int = 1,
                       work_dir: Optional[Path] = None) -> List[str]:
        """Dựng file JSONL, kiểm tra danh sách chặn + hạn mức (tính cả Batch đang chờ), tải lên và tạo Batch."""
        self._require_blocklist()
        work_dir = Path(work_dir or settings.project_root / "data" / "batches")
        work_dir.mkdir(parents=True, exist_ok=True)
        todo = self.pending_api_pages(plans)
        discount = settings.vlm.batch_discount
        created: List[str] = []
        i = 0
        while i < len(todo) and len(created) < max_batches:
            chunk = todo[i:i + batch_pages]
            self.parser.pending_usd = self.store.pending_batch_usd()
            self.parser.check_budget(len(chunk), discount)
            path = work_dir / f"batch_{uuid.uuid4().hex[:12]}.jsonl"
            items: List[Tuple[str, str, int]] = []
            size = 0
            with open(path, "w", encoding="utf-8") as f:
                for plan, p in chunk:
                    self.parser.blocklist.assert_allowed([plan.md5])     # kiểm tra lại ngay trước khi gửi
                    doc = fitz.open(self.root / plan.path)
                    try:
                        body = self.parser.request_body(doc[p], p + 1)
                    finally:
                        doc.close()
                    custom_id = f"{plan.md5}:{p + 1}"
                    line = json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body},
                                      ensure_ascii=False) + "\n"
                    if size + len(line.encode()) > MAX_BATCH_BYTES:
                        break
                    f.write(line)
                    size += len(line.encode())
                    items.append((custom_id, plan.md5, p + 1))
            est = len(items) * self.parser.reserve_cost() * discount
            with open(path, "rb") as fh:
                uploaded = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h",
                                          metadata={"project": "hanoi_metro_agent"})
            self.store.add_batch(batch.id, uploaded.id, items, est)
            os.remove(path)                       # ảnh tài liệu nội bộ: không giữ bản sao trên đĩa sau khi gửi
            logger.info(f"Đã tạo Batch {batch.id}: {len(items)} trang, ước tính {est:.4f} USD")
            created.append(batch.id)
            i += len(items)
        return created

    def collect_batches(self, client) -> Dict[str, int]:
        stats = {"done": 0, "failed": 0, "waiting": 0}
        discount = settings.vlm.batch_discount
        for batch_id in self.store.open_batches():
            b = client.batches.retrieve(batch_id)
            if b.status in ("validating", "in_progress", "finalizing", "cancelling"):
                self.store.update_batch(batch_id, b.status)
                stats["waiting"] += 1
                continue
            for file_id in (getattr(b, "output_file_id", None), getattr(b, "error_file_id", None)):
                if not file_id:
                    continue
                for line in client.files.content(file_id).text.splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    custom_id = obj["custom_id"]
                    md5, page = custom_id.rsplit(":", 1)
                    resp = obj.get("response") or {}
                    body = resp.get("body") or {}
                    if resp.get("status_code") == 200 and body.get("choices"):
                        usage = body.get("usage") or {}
                        self.parser.record_external(md5, int(page), body["choices"][0]["message"].get("content") or "",
                                                    int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), discount)
                        self.store.set_item(custom_id, "done")
                        stats["done"] += 1
                    else:
                        self.store.set_item(custom_id, "failed")   # sẽ được gửi lại ở lần batch-submit sau
                        stats["failed"] += 1
            self.store.update_batch(batch_id, b.status, collected=True)
            logger.info(f"Batch {batch_id}: {b.status}")
        return stats

    # ---- ghép Markdown
    def assemble(self, plans: Dict[str, FilePlan], partial: bool = False, min_partial_chars: int = 300) -> Dict[str, int]:
        """
        Ghép Markdown khi đủ mọi trang. partial=True: văn bản đã xong mọi trang xử lý trên máy nhưng còn trang chờ API
        cũng được ghép tạm (trạng thái "partial", ghi số trang còn thiếu) để tìm kiếm sớm phần text gốc; khi API xong,
        lần ghép sau ghi đè file, bộ nạp thấy nội dung đổi nên tự nạp lại.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        status = self.store.file_status()
        failed = self.store.failed_local_pages()
        stats = {"assembled": 0, "partial": 0, "incomplete": 0, "skipped": 0}
        for plan in plans.values():
            if status.get(plan.md5) in ("assembled", "blocked") or not self._allowed(plan):
                stats["skipped"] += 1
                continue
            local = self.store.local_pages(plan.md5)
            pages: List[Tuple[int, str, str]] = []
            missing_api, missing_local = 0, 0
            for p, route in sorted(plan.routes.items()):
                if self.is_api_route(route) or (plan.md5, p) in failed:
                    row = self.parser.ledger.get(plan.md5, p + 1, self.parser.model)
                    md = row[0] if row else None
                    missing_api += md is None
                else:
                    md = local.get(p)
                    missing_local += md is None
                if md is not None and route != "BLANK" and md.strip():
                    pages.append((p + 1, route, md))
            out = self.out_dir / f"{plan.md5}.md"
            if not missing_api and not missing_local:
                out.write_text(self._markdown(plan, pages), encoding="utf-8")
                self.store.set_file(plan.md5, plan.path, "assembled", str(out))
                stats["assembled"] += 1
            elif partial and not missing_local and sum(len(md) for _, _, md in pages) >= min_partial_chars:
                out.write_text(self._markdown(plan, pages, pending=missing_api), encoding="utf-8")
                self.store.set_file(plan.md5, plan.path, "partial", str(out))
                stats["partial"] += 1
            else:
                stats["incomplete"] += 1
        return stats

    def _markdown(self, plan: FilePlan, pages: List[Tuple[int, str, str]], pending: int = 0) -> str:
        name = Path(plan.path.replace("\\", "/")).name
        counts: Dict[str, int] = {}
        for r in plan.routes.values():
            counts[r] = counts.get(r, 0) + 1
        head = [
            "---",
            f"title: {json.dumps(Path(name).stem, ensure_ascii=False)}",
            f"source_file: {json.dumps(name, ensure_ascii=False)}",
            f"source_path: {json.dumps((self.root / plan.path).as_posix(), ensure_ascii=False)}",
            f"md5_hash: \"{plan.md5}\"",
            f"total_pages: {plan.total_pages}",
            f"processed_at: \"{datetime.now().isoformat(timespec='seconds')}\"",
            f"vlm_model: {json.dumps(self.parser.model)}",
            *([f"pending_api_pages: {pending}"] if pending else []),
            "classification_summary:",
            *[f"  {k}: {v}" for k, v in sorted(counts.items())],
            "---",
        ]
        body = [f"<!-- TRANG {n} [{route}] -->\n\n{md}" for n, route, md in pages]
        return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"

    def status(self, plans: Dict[str, FilePlan]) -> Dict[str, int]:
        routes: Dict[str, int] = {}
        for plan in plans.values():
            for r in plan.routes.values():
                routes[r] = routes.get(r, 0) + 1
        fs = self.store.file_status()
        out = {"files": len(plans), **{f"pages_{k}": v for k, v in sorted(routes.items())}}
        for st in ("assembled", "partial", "blocked", "error"):
            out[f"files_{st}"] = sum(1 for m in plans if fs.get(m) == st)
        out["api_pages_pending"] = len(self.pending_api_pages(plans))
        out["batches_open"] = len(self.store.open_batches())
        return out


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Bóc tách toàn kho theo kết quả page_router")
    ap.add_argument("command", choices=["status", "local", "vlm-sync", "batch-submit", "batch-collect", "assemble", "run"])
    ap.add_argument("--root", type=Path, default=Path(os.getenv("CORPUS_ROOT", r"D:\40GB")))
    ap.add_argument("--results", type=Path, default=Path(os.getenv("CORPUS_RESULTS_DIR", r"D:\40GB\_phan_tich_chi_phi_VLM\ket_qua")))
    ap.add_argument("--out", type=Path, default=settings.project_root / os.getenv("CORPUS_MD_DIR", "data/corpus_md"))
    ap.add_argument("--filter", default="", help="Chỉ xử lý file có đường dẫn chứa chuỗi này")
    ap.add_argument("--limit-files", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-pages", type=int, default=500)
    ap.add_argument("--max-batches", type=int, default=1)
    ap.add_argument("--partial", action="store_true", help="assemble: ghép tạm cả văn bản còn trang chờ API (chỉ phần text gốc)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.command == "local" and not args.child:
        return _supervise(list(argv if argv is not None else sys.argv[1:]))

    plans = load_plan(args.results, args.filter, args.limit_files)
    store = CorpusStore(settings.project_root / "data" / "corpus.db")
    corpus = Corpus(args.root, store, CloudVLMParser(), args.out, os.getenv("CORPUS_OCR_VIA", "vlm"))
    cmd = args.command
    try:
        _dispatch(cmd, args, plans, corpus)
    except BlocklistNotConfigured as e:
        print(f"DỪNG: {e}. Đặt BLOCKLIST_DIR trong .env trỏ tới thư mục kết quả page_router (có restricted_md5.txt).")
        return 2
    print("Trạng thái:", json.dumps(corpus.status(plans), ensure_ascii=False))
    return 0


def _supervise(argv: List[str], max_restarts: int = 500) -> int:
    """
    Chạy bước trích trên máy trong tiến trình con và canh nó. Một số PDF làm PyMuPDF sập ở tầng C (không bắt được
    bằng try/except, cả tiến trình chết) hoặc treo vô hạn ở bước dò bảng. Khi con sập, hoặc một trang xử lý quá
    LOCAL_PAGE_TIMEOUT giây (mặc định 180), dừng con và khởi động lại; con đọc bảng local_inflight để hạ cấp đúng
    các trang đó (bỏ dò bảng, lần sau nữa chuyển API) nên mỗi lần khởi động lại đều tiến lên.
    """
    timeout = float(os.getenv("LOCAL_PAGE_TIMEOUT", "180"))
    store = CorpusStore(settings.project_root / "data" / "corpus.db")
    cmd = [sys.executable, "-m", "src.ingestion.corpus", *argv, "--child"]
    for attempt in range(max_restarts + 1):
        child = subprocess.Popen(cmd, cwd=settings.project_root)
        first_seen: Dict[Tuple[str, int, str], float] = {}
        hung = None
        try:
            while child.poll() is None:
                time.sleep(5)
                now = time.monotonic()
                current = {(m, p, mode) for (m, p), mode in store.inflight().items()}
                first_seen = {k: first_seen.get(k, now) for k in current}
                hung = next((k for k, t in first_seen.items() if now - t > timeout), None)
                if hung:
                    child.kill()
                    child.wait()
                    break
        except KeyboardInterrupt:
            try:
                child.wait(timeout=60)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                child.kill()
            return 130
        code = child.returncode
        if hung is None and code in (0, 2, 130):
            return code
        why = f"trang {hung[1] + 1} xử lý quá {timeout:.0f}s" if hung else f"tiến trình sập (mã {code})"
        print(f"Trích text: {why}; khởi động lại, trang gây lỗi được xử lý ở chế độ an toàn "
              f"(lần {attempt + 1}/{max_restarts}).", flush=True)
    print("Dừng: khởi động lại quá nhiều lần, cần kiểm tra thủ công.")
    return 1


def _dispatch(cmd: str, args, plans, corpus: "Corpus") -> None:
    if cmd in ("local", "run"):
        print("Trang xử lý trên máy:", corpus.run_local(plans, args.workers))
    if cmd in ("vlm-sync", "run"):
        print("Trang gửi API:", corpus.run_vlm_sync(plans, args.workers))
    if cmd == "batch-submit":
        print("Batch đã tạo:", corpus.submit_batches(plans, settings.get_vlm_client(), args.batch_pages, args.max_batches))
    if cmd == "batch-collect":
        print("Kết quả Batch:", corpus.collect_batches(settings.get_vlm_client()))
    if cmd in ("assemble", "run"):
        print("Ghép Markdown:", corpus.assemble(plans, partial=getattr(args, "partial", False)))


if __name__ == "__main__":
    sys.exit(main())
