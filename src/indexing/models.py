"""
Hanoi Metro AI Agent - Shared Index Data Model.
Định danh xuyên suốt: doc_id = MD5 file gốc; chunk_id dùng chung cho Qdrant và nhánh từ khóa,
nên gộp kết quả hai nhánh không cần ánh xạ và mọi câu trả lời truy ngược được về file, trang gốc.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

CHUNKER_VERSION = "structured-v4"


@dataclass
class DocRecord:
    """Một văn bản: metadata dùng cho lọc, phân quyền, liệt kê và trích dẫn."""
    doc_id: str
    source_file: str
    source_path: str
    title: str
    total_pages: int
    so_hieu: str = ""
    ngay_ban_hanh: str = ""          # ISO yyyy-mm-dd
    nam: Optional[int] = None
    loai_vb: str = ""                # CV, QĐ, TB, BC, TTr, GM, KH, HĐ, BB...
    co_quan: str = ""
    don_vi: str = ""                 # đơn vị soạn thảo/ban hành (dùng cho phân quyền theo đơn vị)
    trich_yeu: str = ""
    kinh_gui: List[str] = field(default_factory=list)
    nguoi_ky: str = ""
    noi_nhan: List[str] = field(default_factory=list)
    ho_so: str = "H3"
    ho_so_confidence: str = ""
    ho_so_reasons: List[str] = field(default_factory=list)
    lang: str = "vi"
    pair_id: str = ""
    muc_mat: int = 0
    cites: List[str] = field(default_factory=list)
    chunker_version: str = CHUNKER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParentSection:
    """Mục cha (tới ~2.000 token) đưa cho LLM sau khi tìm thấy chunk con."""
    parent_id: str
    doc_id: str
    breadcrumb: str
    text: str
    page_from: int
    page_to: int


@dataclass
class Chunk:
    """Chunk con (250–500 token với tài liệu kỹ thuật) dùng để tìm kiếm."""
    chunk_id: str
    doc_id: str
    parent_id: str
    seq: int
    page_from: int
    page_to: int
    breadcrumb: str
    context: str                     # dòng ngữ cảnh: [số hiệu | ngày | đơn vị | trích yếu] › đề mục
    text: str                        # Markdown hiển thị / đưa vào LLM
    search_text: str                 # context + nội dung (bảng đã làm phẳng) cho embedding & toàn văn
    codes: List[str] = field(default_factory=list)
    is_table: bool = False
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
