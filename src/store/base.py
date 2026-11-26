"""
Hanoi Metro AI Agent - Search & Storage Contracts.
Thiết kế adapter dùng chung cho đánh giá thu nhỏ và triển khai thật (Mục 5.2 đề xuất):
Qdrant, PostgreSQL, SQLite (thử nghiệm) đều cài đặt cùng một giao diện, trả về [(chunk_id, thứ hạng)],
nên gộp RRF giống nhau cho mọi hệ và đổi nhánh từ khóa chỉ là thay một module.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Protocol, Sequence, Tuple

from src.indexing.models import Chunk, DocRecord, ParentSection
from src.text.abbrev import Expansion


@dataclass(frozen=True)
class AccessScope:
    """
    Phạm vi người dùng được xem: văn bản có nhãn phòng thuộc `phong`, hoặc doc_id được cấp riêng.
    Hai tập rỗng = không xem được văn bản nào (mặc định chặn). Filters.access = None nghĩa là không giới hạn.
    """
    phong: FrozenSet[str] = frozenset()
    doc_ids: FrozenSet[str] = frozenset()

    @property
    def empty(self) -> bool:
        return not self.phong and not self.doc_ids


@dataclass
class Filters:
    """Bộ lọc metadata và quyền, áp ở mọi nhánh TRƯỚC khi gộp kết quả."""
    doc_ids: Optional[List[str]] = None
    loai_vb: Optional[List[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    don_vi: Optional[List[str]] = None        # đơn vị người dùng được xem; None = không giới hạn
    max_muc_mat: int = 0                      # tài liệu mật không có trong chỉ mục; giữ để mở rộng
    lang: Optional[str] = None
    access: Optional[AccessScope] = None      # phân quyền theo phòng ban (src.access); None = không giới hạn
    huong_vb: Optional[List[str]] = None      # di / den / noi_bo / khong_ro (bảng doc_fields, src.indexing.enrich)


@dataclass
class ParsedQuery:
    """Câu hỏi đã phân tích bằng quy tắc (không cần LLM)."""
    text: str
    terms: List[str] = field(default_factory=list)          # từ nội dung (đã bỏ từ dừng)
    codes: List[str] = field(default_factory=list)          # số hiệu / mã đã chuẩn hóa
    expansions: List[Expansion] = field(default_factory=list)
    loai_vb: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    listing: bool = False                                   # "tất cả / liệt kê / danh sách"
    huong_vb: List[str] = field(default_factory=list)       # "công văn đến", "văn bản đi"


Ranked = List[Tuple[str, int]]

# Nguồn của một giá trị metadata bổ sung. Phân quyền chỉ dùng giá trị NGUON_DUYET; NGUON_AI chỉ để hiển thị.
NGUON_LUAT, NGUON_AI, NGUON_DUYET = "luat", "ai", "duyet"
_NGUON_RANK = {NGUON_AI: 0, NGUON_LUAT: 1, NGUON_DUYET: 2}


def nguon_rank(nguon: str) -> int:
    """Giá trị nguồn thấp hơn không được ghi đè giá trị nguồn cao hơn (AI < luật < người duyệt)."""
    return _NGUON_RANK.get(nguon, -1)


@dataclass
class DocField:
    """Một giá trị metadata bổ sung (bảng doc_fields): không nằm trong DocRecord nên không phải nhúng lại."""
    truong: str
    gia_tri: str
    nguon: str = NGUON_LUAT
    trang: Optional[int] = None               # trang chứng cứ (tính từ 1)
    do_tin_cay: str = ""
    phien_ban_bo_trich: str = ""
    ghi_chu: str = ""
    nguoi_duyet: str = ""


@dataclass
class DocRelation:
    """Quan hệ văn bản → văn bản (bảng doc_relations); lưu theo số hiệu đích, có doc_id_dich nếu đích có trong kho."""
    loai: str                                 # can_cu, tra_loi, thay_the, bai_bo, sua_doi, kem_theo, lien_quan
    so_hieu_dich: str
    doc_id_dich: str = ""
    trang: Optional[int] = None
    nguon: str = NGUON_LUAT
    doc_id: str = ""                          # văn bản nguồn (điền khi đọc ra)


@dataclass
class DocTask:
    """Nhiệm vụ được giao trong văn bản (bảng doc_tasks); hiện chỉ do AI trích."""
    phong: str
    nguoi: str
    noi_dung: str
    han: str = ""                             # ISO yyyy-mm-dd hoặc rỗng
    loai_han: str = ""                        # ngay | dinh_ky | khong_co
    trang: Optional[int] = None
    nguon: str = NGUON_AI
    doc_id: str = ""


def access_allows(scope: Optional[AccessScope], doc_id: str, labels: Sequence[str]) -> bool:
    """Một văn bản cụ thể có trong phạm vi xem không (dùng khi tra thẳng theo doc_id/số hiệu, mở file gốc)."""
    return scope is None or doc_id in scope.doc_ids or bool(scope.phong & set(labels))


class SearchBranch(Protocol):
    """Một nhánh tìm kiếm (vector hoặc từ khóa)."""
    name: str

    def index(self, doc: DocRecord, chunks: Sequence[Chunk]) -> None: ...

    def delete(self, doc_id: str) -> None: ...

    def search(self, query: ParsedQuery, filters: Filters, k: int) -> Ranked: ...


class DocumentStore(Protocol):
    """Kho metadata văn bản, mục cha và nội dung chunk (PostgreSQL khi triển khai, SQLite khi thử nghiệm)."""

    def upsert_document(self, doc: DocRecord, parents: Sequence[ParentSection], chunks: Sequence[Chunk]) -> None: ...

    def delete_document(self, doc_id: str) -> None: ...

    def indexed_version(self, doc_id: str) -> Optional[str]: ...

    def get_document(self, doc_id: str) -> Optional[DocRecord]: ...

    def get_documents(self, doc_ids: Sequence[str]) -> Dict[str, DocRecord]: ...

    def find_documents_by_code(self, code: str, limit: int = 5) -> List[DocRecord]: ...

    def list_documents(self, filters: Filters, keyword: str = "", limit: int = 200, offset: int = 0) -> Tuple[List[DocRecord], int]: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, Chunk]: ...

    def get_document_chunks(self, doc_id: str) -> List[Chunk]: ...

    def get_parents(self, parent_ids: Sequence[str]) -> Dict[str, ParentSection]: ...

    def set_access_labels(self, doc_id: str, phong: Sequence[str], nguon: str = "", ghi_chu: str = "") -> None: ...

    def access_labels(self, doc_ids: Sequence[str]) -> Dict[str, List[str]]: ...

    # ---- metadata bổ sung (src.indexing.enrich); khóa doc_id, không đụng tới chunk/vector
    def set_doc_fields(self, doc_id: str, fields: Sequence[DocField]) -> List[str]:
        """Ghi các trường; không ghi đè giá trị có nguồn cao hơn (duyet > luat > ai). Trả về tên trường đã đổi."""
        ...

    def doc_fields(self, doc_ids: Sequence[str], truong: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, DocField]]: ...

    def set_doc_relations(self, doc_id: str, relations: Sequence[DocRelation], nguon: str) -> None:
        """Thay toàn bộ quan hệ nguồn `nguon` của văn bản."""
        ...

    def doc_relations(self, doc_ids: Sequence[str]) -> List[DocRelation]:
        """Quan hệ đi ra từ các văn bản."""
        ...

    def relations_to(self, doc_id: str, so_hieu: str = "") -> List[DocRelation]:
        """Quan hệ trỏ tới văn bản (theo doc_id_dich hoặc số hiệu đích)."""
        ...

    def set_doc_tasks(self, doc_id: str, tasks: Sequence[DocTask], nguon: str) -> None: ...

    def doc_tasks(self, doc_ids: Sequence[str]) -> List[DocTask]: ...
