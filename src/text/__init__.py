"""
Hanoi Metro AI Agent - Text Utilities Package.
Chuẩn hóa văn bản tiếng Việt, trích xuất số hiệu/mã và từ điển viết tắt dùng chung
cho chunking, lập chỉ mục và phân tích câu hỏi.
"""

from src.text.normalize import normalize_text, fold, collapse_repetitions, approx_tokens
from src.text.codes import extract_codes, normalize_code
from src.text.abbrev import AbbreviationDictionary

__all__ = [
    "normalize_text",
    "fold",
    "collapse_repetitions",
    "approx_tokens",
    "extract_codes",
    "normalize_code",
    "AbbreviationDictionary",
]
