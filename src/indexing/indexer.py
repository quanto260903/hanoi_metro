"""
Hanoi Metro AI Agent - Two-branch Indexer.
Nạp tăng dần từ data/processed_md: kiểm tra danh sách chặn tài liệu mật TRƯỚC mọi bước, bỏ qua văn bản
không đổi (cùng phiên bản chunker + cùng nội dung), thay chunk cũ ở cả hai nhánh trong cùng một lần chạy.
Ghi theo chunk_id trong một tiến trình nạp để Qdrant và kho từ khóa luôn đồng bộ.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from src.indexing.document import load_markdown
from src.indexing.models import CHUNKER_VERSION
from src.indexing.structured import StructuredChunker
from src.store.base import DocumentStore, SearchBranch

logger = logging.getLogger("hanoi_metro.indexer")


def _read_md5_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^\s*([0-9a-fA-F]{32})\b", line)
        if m:
            out.add(m.group(1).lower())
    return out


class Blocklist:
    """md5 bị chặn = tự phát hiện (restricted_md5.txt) + thủ công (blocklist_md5.txt) − đã xác nhận không mật (allowlist_md5.txt)."""

    def __init__(self, blocked: Set[str], configured: bool = True, problem: str = ""):
        self.blocked = blocked
        # configured=False: thiếu BLOCKLIST_DIR hoặc thiếu restricted_md5.txt -> mọi lần gửi ra ngoài đều bị từ chối
        self.configured = configured
        self.problem = problem

    @classmethod
    def from_dir(cls, directory: Optional[Path]) -> "Blocklist":
        if directory is None:
            problem = "chưa đặt BLOCKLIST_DIR"
            logger.error(f"{problem.upper()}: mọi lần gửi tài liệu ra dịch vụ bên ngoài (VLM/LLM API) sẽ bị từ chối. "
                         "Đặt BLOCKLIST_DIR trong .env (thư mục kết quả page_router).")
            return cls(set(), configured=False, problem=problem)
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"BLOCKLIST_DIR không tồn tại: {d}")
        blocked = (_read_md5_list(d / "restricted_md5.txt") | _read_md5_list(d / "blocklist_md5.txt")) - _read_md5_list(d / "allowlist_md5.txt")
        if not (d / "restricted_md5.txt").is_file():
            problem = f"thiếu restricted_md5.txt trong {d} (chưa chạy page_router restricted?)"
            logger.error(f"Danh sách chặn chưa đầy đủ: {problem}. Mọi lần gửi tài liệu ra dịch vụ bên ngoài sẽ bị từ chối.")
            return cls(blocked, configured=False, problem=problem)
        logger.info(f"Đã nạp danh sách chặn tài liệu mật: {len(blocked)} md5 từ {d}")
        return cls(blocked)

    def is_blocked(self, md5: str) -> bool:
        return md5.lower() in self.blocked

    def assert_allowed(self, md5s: Sequence[str]) -> None:
        """Gọi ngay trước khi gửi nội dung tài liệu ra dịch vụ bên ngoài (LLM/VLM API); có tài liệu bị chặn thì ném lỗi."""
        if not self.configured:
            raise BlocklistNotConfigured(f"Danh sách chặn tài liệu mật chưa được cấu hình ({self.problem}); không gửi dữ liệu ra ngoài")
        hits = [m for m in md5s if self.is_blocked(m)]
        if hits:
            raise BlockedDocumentError(f"{len(hits)} tài liệu có dấu chỉ độ mật, cấm gửi ra ngoài")


class BlockedDocumentError(RuntimeError):
    pass


class BlocklistNotConfigured(BlockedDocumentError):
    """Thiếu danh sách chặn: từ chối gửi mọi tài liệu ra ngoài (fail-closed) cho tới khi cấu hình BLOCKLIST_DIR đúng."""


@dataclass
class IndexReport:
    indexed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    chunks: int = 0

    def summary(self) -> str:
        return (f"nạp {len(self.indexed)} văn bản ({self.chunks} chunk), không đổi {len(self.unchanged)}, "
                f"bị chặn {len(self.blocked)}, lỗi {len(self.failed)}")


class DocumentIndexer:
    def __init__(self, chunker: StructuredChunker, store: DocumentStore, vector: Optional[SearchBranch], blocklist: Blocklist):
        self.chunker = chunker
        self.store = store
        self.vector = vector
        self.blocklist = blocklist

    def index_file(self, path: Path, report: IndexReport, force: bool = False) -> None:
        path = Path(path)
        raw = path.read_bytes()
        src = load_markdown(path)
        if self.blocklist.is_blocked(src.doc_id):
            # Tài liệu mật không được có trong bất kỳ chỉ mục nào: xóa luôn nếu trước đây đã lỡ nạp
            self._remove(src.doc_id)
            report.blocked.append(src.source_file)
            logger.warning(f"BỎ QUA tài liệu nằm trong danh sách chặn: {src.doc_id}")
            return

        version = f"{CHUNKER_VERSION}:{hashlib.md5(raw).hexdigest()[:12]}"
        if not force and self.store.indexed_version(src.doc_id) == version:
            report.unchanged.append(src.source_file)
            return

        doc, parents, chunks = self.chunker.chunk_document(src)
        doc.chunker_version = version
        try:
            if self.vector is not None:
                self.vector.delete(doc.doc_id)
                labels = self.store.access_labels([doc.doc_id]).get(doc.doc_id, []) if hasattr(self.store, "access_labels") else []
                self.vector.index(doc, chunks, phong=labels)   # nạp lại không làm mất nhãn phòng đã duyệt
            self.store.upsert_document(doc, parents, chunks)
        except Exception as e:
            try:
                self._remove(doc.doc_id)
            except Exception as cleanup_error:
                logger.error(f"Không gỡ được '{doc.doc_id}' sau lỗi nạp: {cleanup_error}")
            report.failed[src.source_file] = str(e)
            logger.error(f"Lỗi nạp '{src.source_file}', đã gỡ khỏi cả hai nhánh: {e}")
            return
        report.indexed.append(src.source_file)
        report.chunks += len(chunks)

    def index_paths(self, paths: Sequence[Path], force: bool = False) -> IndexReport:
        report = IndexReport()
        for p in paths:
            self.index_file(p, report, force=force)
        logger.info(f"Hoàn tất: {report.summary()}")
        return report

    def index_directory(self, directory: Path, force: bool = False) -> IndexReport:
        return self.index_paths(sorted(Path(directory).glob("*.md")), force=force)

    def _remove(self, doc_id: str) -> None:
        if self.vector is not None:
            self.vector.delete(doc_id)
        self.store.delete_document(doc_id)
