"""
Hanoi Metro AI Agent - Rule-based Query Parser (bước 1 luồng truy vấn, không cần LLM).
Nhận diện số hiệu, mã, năm, loại văn bản, ý định liệt kê và mở rộng viết tắt bằng từ điển nội bộ.
"""

import re
from typing import List

from src.store.base import ParsedQuery
from src.text.abbrev import AbbreviationDictionary
from src.text.codes import extract_codes
from src.text.normalize import content_terms, fold

_LOAI_WORDS = {
    "quyet dinh": "QĐ", "cong van": "CV", "to trinh": "TTr", "thong bao": "TB", "bien ban": "BB",
    "ke hoach": "KH", "bao cao": "BC", "giay moi": "GM", "hop dong": "HĐ", "quy che": "QC",
}
_RX_LISTING = re.compile(r"\b(tat ca|liet ke|danh sach|toan bo|thong ke|bao nhieu (van ban|quyet dinh|cong van|to trinh|bien ban))\b")
_RX_YEAR = re.compile(r"(?<![\d/.-])((?:19|20)\d{2})(?![\d/-])")
_RX_HUONG = re.compile(r"\b(?:cong van|van ban|cv|vb|thu)\s+(di|den|noi bo)\b")
_HUONG = {"di": "di", "den": "den", "noi bo": "noi_bo"}
_LISTING_WORDS = {"tất", "cả", "liệt", "kê", "danh", "sách", "toàn", "bộ", "năm", "thống"}


def parse_query(text: str, abbrev: AbbreviationDictionary) -> ParsedQuery:
    text = text.strip()
    codes = list(dict.fromkeys(extract_codes(text) + extract_codes(text.upper())))
    folded = fold(text)
    loai = [code for word, code in _LOAI_WORDS.items() if re.search(rf"\b{word}\b", folded)]
    without_codes = text
    for raw in re.findall(r"\S*\d\S*[/-]\S+", text):
        without_codes = without_codes.replace(raw, " ")
    years = [int(y) for y in _RX_YEAR.findall(without_codes)]
    return ParsedQuery(
        text=text,
        terms=content_terms(without_codes),
        codes=codes,
        expansions=abbrev.expand(text),
        loai_vb=loai,
        years=years,
        listing=bool(_RX_LISTING.search(folded)),
        huong_vb=list(dict.fromkeys(_HUONG[m] for m in _RX_HUONG.findall(folded))),
    )


def listing_keywords(q: ParsedQuery) -> str:
    """Từ khóa còn lại của câu hỏi liệt kê sau khi bỏ tên loại văn bản, năm và từ chỉ ý liệt kê."""
    drop = {w for word in _LOAI_WORDS for w in word.split()}
    if q.huong_vb:
        drop |= {"đến", "đi", "nội", "bộ", "văn", "bản"}
    kept: List[str] = [t for t in q.terms if fold(t) not in drop and t not in _LISTING_WORDS and not t.isdigit()]
    return " ".join(kept)
