"""
Hanoi Metro AI Agent - Vietnamese Text Normalization.
Đưa văn bản về Unicode NFC, sửa glyph lỗi font (Ƣ → Ư), tách chữ dính kiểu "SởXây dựng",
gộp các khối dòng lặp do VLM sinh lặp vô hạn, và tạo dạng "không dấu" cho tìm kiếm từ khóa.
"""

import re
import unicodedata
from typing import List

# Glyph lỗi khi xuất PDF từ font TCVN3/VNI sang Unicode (đo được trên ~4/25 tài liệu mẫu)
_GLYPH_FIXES = str.maketrans({
    "Ƣ": "Ư",
    "ƣ": "ư",
    " ": " ",   # non-breaking space
    "​": "",    # zero-width space
    "﻿": "",    # BOM
    "­": "",    # soft hyphen
})

_VI_LOWER = "a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_VI_UPPER = "A-ZÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"

# "SởXây" -> "Sở Xây": chữ thường có dấu đứng liền chữ hoa. Chỉ tách khi bên trái có dấu
# tiếng Việt để không phá các tên riêng/mã kiểu "iPhone", "McAfee", "HPE".
_RX_GLUED = re.compile(
    rf"(?<=[{_VI_UPPER}])([{_VI_LOWER}]*[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ][{_VI_LOWER}]*)(?=[{_VI_UPPER}][{_VI_LOWER}])"
)
_RX_SPACES = re.compile(r"[ \t]+")
_RX_WORD = re.compile(r"[\wđĐ]+")

# Từ dừng giữ nguyên dấu: "bao" trong "bao nhiêu" là từ dừng nhưng "báo" trong "báo cáo" thì không
_STOPWORDS = {
    "và", "của", "là", "có", "các", "những", "được", "cho", "trong", "với", "về", "theo", "tại", "này", "đó",
    "nào", "gì", "như", "thế", "bao", "nhiêu", "không", "hãy", "biết", "tôi", "ai", "khi", "đã", "sẽ", "đang",
    "thì", "mà", "để", "từ", "đến", "ra", "vào", "lên", "một", "rồi", "nhé", "ạ", "hỏi", "tìm", "xem", "giúp",
    "mình", "ở", "bị", "nêu", "cụ", "thể", "gồm", "sao", "diễn",
}


def content_terms(text: str) -> List[str]:
    """Từ nội dung (chữ thường, giữ dấu) sau khi bỏ từ dừng và từ 1 ký tự."""
    return [t for t in _RX_WORD.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def normalize_text(text: str) -> str:
    """Chuẩn hóa NFC, sửa glyph lỗi font, tách chữ dính, gộp khoảng trắng thừa trong dòng."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).translate(_GLYPH_FIXES)
    text = _RX_GLUED.sub(r"\1 ", text)
    return "\n".join(_RX_SPACES.sub(" ", line).rstrip() for line in text.splitlines())


def fold(text: str) -> str:
    """Bỏ dấu tiếng Việt và chuyển chữ thường: "Báo cáo Đường sắt" -> "bao cao duong sat"."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()


def approx_tokens(text: str) -> int:
    """Ước lượng số token (bge-m3/LLM) cho tiếng Việt: ~1,3 token mỗi âm tiết."""
    return int(len(text.split()) * 1.3) + 1


def collapse_repetitions(lines: List[str], max_block: int = 40, min_repeats: int = 3) -> List[str]:
    """
    Gộp khối dòng lặp liên tiếp >= min_repeats lần thành 1 khối.
    VLM cỡ nhỏ hay rơi vào vòng lặp (ví dụ khối chữ ký lặp 35 lần ở cuối biên bản);
    nếu không gộp, chunk bị phình và vector bị nhiễu.
    """
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        collapsed = False
        for size in range(1, min(max_block, (n - i) // min_repeats) + 1):
            block = lines[i:i + size]
            if not any(line.strip() for line in block):
                continue
            repeats = 1
            while lines[i + repeats * size:i + (repeats + 1) * size] == block:
                repeats += 1
            if repeats >= min_repeats:
                out.extend(block)
                i += repeats * size
                collapsed = True
                break
        if not collapsed:
            out.append(lines[i])
            i += 1
    return out
