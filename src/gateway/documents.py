"""
Hanoi Metro AI Agent - Mở tài liệu gốc từ trích dẫn.
Mỗi nguồn [n] trong câu trả lời là một liên kết tới file PDF gốc, nhảy thẳng tới trang được trích (#page=N).

An toàn:
- Liên kết mang chữ ký HMAC theo doc_id: chỉ văn bản mà dịch vụ đã trích dẫn mới mở được, không đoán/dò được
  doc_id khác. Chữ ký không hết hạn để liên kết trong lịch sử hội thoại vẫn dùng được.
- Kiểm tra danh sách chặn (tài liệu mật) ngay lúc mở, kể cả khi văn bản bị đánh dấu mật sau khi đã được trích.
- Chỉ phục vụ file nằm trong các thư mục gốc cho phép (DOC_ROOTS); đường dẫn lấy từ chỉ mục, không lấy từ URL.
- Khi bật phân quyền: liên kết gắn với người dùng (định danh băm, không chứa email) và có hạn dùng; lúc mở kiểm tra
  LẠI quyền hiện tại của người đó, nên thu hồi quyền có hiệu lực ngay cả với liên kết cũ.
"""

import hashlib
import hmac
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union
from urllib.parse import quote

from src.store.base import DocumentStore

_RX_DOC_ID = re.compile(r"^[0-9a-f]{32}$")
_RX_FIRST_PAGE = re.compile(r"\d+")


def first_page(pages: str) -> int:
    m = _RX_FIRST_PAGE.search(pages or "")
    return max(1, int(m.group(0))) if m else 1


@dataclass
class DocumentLinks:
    secret: bytes
    base_url: str
    roots: Sequence[Path]
    is_blocked: Callable[[str], bool]

    @classmethod
    def create(cls, secret: str, base_url: str, roots: Sequence[Union[str, Path]],
               is_blocked: Callable[[str], bool]) -> Optional["DocumentLinks"]:
        """Không có khóa bí mật thì không tạo liên kết (trích dẫn giữ dạng chữ như trước)."""
        if not secret:
            return None
        key = hmac.new(secret.encode(), b"hanoi-metro/doc-links/v1", hashlib.sha256).digest()
        return cls(key, base_url.rstrip("/"), [Path(r).resolve() for r in roots if str(r).strip()], is_blocked)

    def sign(self, doc_id: str, user: str = "", expires: int = 0) -> str:
        msg = doc_id if not user else f"{doc_id}|{user}|{expires}"
        return hmac.new(self.secret, msg.encode(), hashlib.sha256).hexdigest()[:32]

    def url(self, doc_id: str, source_file: str, pages: str, user: str = "", ttl_seconds: int = 0) -> str:
        name = quote(Path(source_file).name or "tai-lieu", safe="")
        if not user:
            return f"{self.base_url}/tai-lieu/{doc_id}/{name}?k={self.sign(doc_id)}#page={first_page(pages)}"
        expires = int(time.time()) + ttl_seconds if ttl_seconds > 0 else 0
        return (f"{self.base_url}/tai-lieu/{doc_id}/{name}?u={user}&e={expires}&k={self.sign(doc_id, user, expires)}"
                f"#page={first_page(pages)}")

    def _allowed(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.roots)

    def resolve(self, store: DocumentStore, doc_id: str, signature: str, user: str = "", expires: str = "",
                can_open: Optional[Callable[[str, str], bool]] = None) -> Tuple[int, Optional[Path], str]:
        """
        → (mã HTTP, đường dẫn file, tên hiển thị). Lỗi trả mã 403/404, không nói lý do chi tiết ra ngoài.
        can_open(user, doc_id): khi bật phân quyền, bắt buộc liên kết gắn người dùng và người đó còn quyền xem.
        """
        if not _RX_DOC_ID.match(doc_id or ""):
            return 403, None, ""
        try:
            exp = int(expires or 0)
        except ValueError:
            return 403, None, ""
        if not hmac.compare_digest(self.sign(doc_id, user, exp), signature or ""):
            return 403, None, ""
        if user and exp and time.time() > exp:
            return 403, None, ""
        if can_open is not None and (not user or not can_open(user, doc_id)):
            return 403, None, ""
        if self.is_blocked(doc_id):
            return 403, None, ""
        doc = store.get_documents([doc_id]).get(doc_id)
        if not doc or not doc.source_path:
            return 404, None, ""
        path = Path(doc.source_path).resolve()
        if not self._allowed(path) or not path.is_file():
            return 404, None, ""
        return 200, path, doc.source_file or path.name


def media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
