"""
Hanoi Metro AI Agent - Document Number & Code Extraction.
Trích số hiệu văn bản (916/CV-VT&QLPT), mã tài liệu (HNC-IMP-PPG-ZTO-D00-00015),
mã gói thầu (CP-06), mã thiết bị (DC2-HN01) và chuẩn hóa để khớp chính xác tuyệt đối.
Bộ tách từ toàn văn cắt các mã này thành nhiều mảnh, nên chúng được lưu ở cột riêng.
"""

import re
from typing import List

_U = "A-ZĐÀ-Ỹ"
# Số hiệu văn bản: "916/CV-VT&QLPT", "485/TT-KHDA", "06-QC/TU", "983/ĐA-ĐSĐT-HCTC", "2922/BC-ĐSHN",
# có năm ở giữa (văn bản quy phạm, một số công văn): "30/2020/NĐ-CP", "449/2026/CV-VHM-VSP"
_RX_SO_HIEU = re.compile(
    rf"(?<![\w/])(\d{{1,5}})(?:\s*/\s*(?:19|20)\d{{2}})?\s*[/-]\s*([{_U}][{_U}0-9]*(?:\s*[-/&]\s*[{_U}0-9][{_U}0-9&]*)*)",
)
# Mã tài liệu / thiết bị / gói thầu: có ít nhất 1 dấu "-" và 1 chữ số, ví dụ DC2-HN01, CP-06, UJV-OPM-001
_RX_CODE = re.compile(rf"(?<![\w-])([{_U}]{{1,6}}[0-9]*(?:-[{_U}0-9]{{1,10}}){{1,6}})(?![\w-])")
# Mã serial/model dạng chữ + số liền nhau: N3K93180, R660
_RX_SERIAL = re.compile(r"(?<![\w-])([A-Z]{1,4}\d[A-Z0-9]{3,})(?![\w-])")

_RX_DASH = re.compile(r"\s*([-/&])\s*")


def normalize_code(code: str) -> str:
    """Chuẩn hóa mã: chữ hoa, bỏ khoảng trắng quanh "/", "-", "&"; "916 / cv-vt&qlpt" -> "916/CV-VT&QLPT"."""
    return _RX_DASH.sub(r"\1", code.strip().upper()).replace(" ", "")


def extract_codes(text: str) -> List[str]:
    """Trả về danh sách mã đã chuẩn hóa (giữ thứ tự xuất hiện, không trùng)."""
    found: List[str] = []
    seen = set()

    def add(code: str) -> None:
        norm = normalize_code(code)
        if len(norm) >= 4 and norm not in seen:
            seen.add(norm)
            found.append(norm)

    for m in _RX_SO_HIEU.finditer(text):
        add(m.group(0))
    for rx in (_RX_CODE, _RX_SERIAL):
        for m in rx.finditer(text):
            if any(ch.isdigit() for ch in m.group(1)):
                add(m.group(1))
    return found
