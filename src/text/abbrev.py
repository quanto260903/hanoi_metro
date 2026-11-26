"""
Hanoi Metro AI Agent - Internal Abbreviation Dictionary.
Từ điển viết tắt nội bộ (data/dictionaries/synonym_dict.json): viết tắt ↔ dạng đầy đủ, lĩnh vực,
trạng thái duyệt. Mở rộng ở thời điểm truy vấn cho cả hai nhánh tìm kiếm, nội dung gốc không bị sửa,
nên khi từ điển cập nhật không phải đánh chỉ mục lại.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.text.normalize import fold


@dataclass
class AbbreviationEntry:
    abbr: str
    expansions: List[str]
    domain: str = ""
    approved: bool = False


@dataclass
class Expansion:
    """Một cụm trong câu hỏi kèm các dạng tương đương (viết tắt hoặc đầy đủ)."""
    matched: str
    alternatives: List[str] = field(default_factory=list)


class AbbreviationDictionary:
    """Tra cứu hai chiều: "BCVT" -> "báo cáo văn thư" và "báo cáo văn thư" -> "BCVT"."""

    # "Hệ thống thu soát vé tự động (AFC)" hoặc "AFC (Hệ thống thu soát vé tự động)"
    _RX_FULL_ABBR = re.compile(r"((?:[A-ZĐÀ-Ỹa-zđà-ỹ]+\s+){1,8}[A-ZĐÀ-Ỹa-zđà-ỹ]+)\s*\(\s*([A-ZĐ][A-ZĐ0-9&\-]{1,12})\s*\)")
    _RX_ABBR_FULL = re.compile(r"\b([A-ZĐ][A-ZĐ0-9&\-]{1,12})\s*\(\s*((?:[A-ZĐÀ-Ỹa-zđà-ỹ]+\s+){1,8}[A-ZĐÀ-Ỹa-zđà-ỹ]+)\s*\)")

    def __init__(self, entries: Optional[List[AbbreviationEntry]] = None, approved_only: bool = True):
        self.entries = [e for e in (entries or []) if e.approved or not approved_only]
        self._by_abbr: Dict[str, List[AbbreviationEntry]] = {}
        for e in self.entries:
            self._by_abbr.setdefault(e.abbr.upper(), []).append(e)

    @classmethod
    def load(cls, path: Path, approved_only: bool = True) -> "AbbreviationDictionary":
        path = Path(path)
        if not path.exists():
            return cls([], approved_only)
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = [AbbreviationEntry(**item) for item in raw.get("entries", [])]
        return cls(entries, approved_only)

    def expand(self, query: str) -> List[Expansion]:
        """Tìm các viết tắt và dạng đầy đủ có trong câu hỏi, trả về cụm tương đương để OR khi tìm kiếm."""
        result: List[Expansion] = []
        folded_query = f" {fold(query)} "
        tokens = set(re.findall(r"[A-ZĐ][A-ZĐ0-9&]+", query))

        for abbr, entries in self._by_abbr.items():
            if abbr in tokens:
                alts = [x for e in entries for x in e.expansions]
                result.append(Expansion(matched=abbr, alternatives=alts))
                continue
            for e in entries:
                for full in e.expansions:
                    if f" {fold(full)} " in folded_query:
                        result.append(Expansion(matched=full, alternatives=[e.abbr]))
                        break
        return result

    @classmethod
    def mine_candidates(cls, text: str) -> List[Tuple[str, str]]:
        """Khai thác ứng viên "Tên đầy đủ (VIẾT TẮT)" từ kho để cán bộ chuyên môn duyệt."""
        found: List[Tuple[str, str]] = []
        for m in cls._RX_FULL_ABBR.finditer(text):
            words, abbr = m.group(1).split(), m.group(2)
            target = fold(abbr).replace("&", "").replace("-", "")
            # không khớp chữ cái đầu (viết tắt tiếng Anh): giữ từ chữ hoa đầu tiên trở đi để người duyệt cắt
            first_cap = next((k for k, w in enumerate(words) if w[0].isupper()), 0)
            full = " ".join(words[first_cap:])
            for n in range(1, len(words) + 1):
                if "".join(fold(w)[0] for w in words[-n:]) == target:
                    full = " ".join(words[-n:])
                    break
            found.append((abbr, full))
        for m in cls._RX_ABBR_FULL.finditer(text):
            found.append((m.group(1), m.group(2).strip()))
        return found
