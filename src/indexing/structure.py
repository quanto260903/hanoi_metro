"""
Hanoi Metro AI Agent - Document Structure Analysis (Bước 1 và 3 của quy trình chunking).
- Làm sạch: chuẩn hóa Unicode, gộp vòng lặp VLM, loại header/footer lặp trên >50% số trang,
  số trang, trang mục lục (giữ làm khung đề mục).
- Tách khối thể thức hành chính (Nghị định 30) thành metadata, không đưa vào nội dung chunk.
- Dựng danh sách khối (đề mục / đoạn văn / bảng) kèm số trang; đề mục nhận bằng biểu thức chính quy
  nên chạy được cả trên kết quả OCR không có thẻ #.
"""

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

from src.text.codes import extract_codes, normalize_code
from src.text.normalize import collapse_repetitions, fold, normalize_text


@dataclass
class Line:
    text: str
    page: int


@dataclass
class Block:
    kind: str                 # "heading" | "text" | "table"
    text: str
    page_from: int
    page_to: int
    level: int = 0            # cấp đề mục (heading)
    legal: str = ""           # "chuong" | "dieu" | "phu_luc" với đề mục pháp quy


@dataclass
class AdminInfo:
    """Khối thể thức hành chính tách ra khỏi nội dung."""
    co_quan: str = ""
    don_vi: str = ""
    so_hieu: str = ""
    ngay_ban_hanh: str = ""
    loai_vb: str = ""
    trich_yeu: str = ""
    kinh_gui: List[str] = field(default_factory=list)
    nguoi_ky: str = ""
    noi_nhan: List[str] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)   # quoc_hieu, so, vv, kinh_gui, noi_nhan


# ------------------------------------------------------------------------------
# 1. Làm sạch
# ------------------------------------------------------------------------------
_RX_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*(?:trang|page)?\s*\d{1,4}\s*(?:/\s*\d{1,4})?\s*[-–—]?\s*$", re.I)
_RX_TOC = re.compile(r"^\s*(.{3,}?)\s*(?:\.{4,}|…{2,}|(?:\. ){4,})\s*(\d{1,4})\s*$")
_RX_FENCE = re.compile(r"^\s*```(?:markdown|md|html)?\s*$", re.I)
# VLM đánh dấu chữ viết tay "[Viết tay: 9]": giữ nội dung, bỏ nhãn để nhận được ngày tháng, số hiệu, mã thiết bị
# Watermark ứng dụng quét điện thoại: chen giữa bảng vắt qua trang và gây nhiễu tìm kiếm
_RX_SCANNER_MARK = re.compile(r"^\s*(?:CS\s*)?(?:Được quét bằng|Scanned (?:with|by))\s+(?:CamScanner|Adobe Scan|Microsoft Lens|Genius Scan)\s*$", re.I)
_RX_HANDWRITING = re.compile(r"\[\s*Viết tay\s*:\s*([^\]]*?)\s*\]", re.I)


def clean_pages(pages: List[Tuple[int, str]]) -> Tuple[List[Line], List[str]]:
    """Trả về (các dòng đã làm sạch kèm số trang, các mục lục thu được)."""
    per_page: List[Tuple[int, List[str]]] = []
    toc: List[str] = []
    for number, text in pages:
        lines = [_RX_HANDWRITING.sub(r"\1", ln) for ln in normalize_text(text).splitlines() if not _RX_FENCE.match(ln)]
        lines = collapse_repetitions(lines)
        kept = []
        for ln in lines:
            if (_RX_PAGE_NUMBER.match(ln) and ln.strip()) or _RX_SCANNER_MARK.match(ln):
                continue
            m = _RX_TOC.match(ln)
            if m and re.search(r"[A-Za-zÀ-ỹ]{2,}", m.group(1)):
                toc.append(m.group(1).strip())
                continue
            kept.append(ln)
        per_page.append((number, kept))

    repeated = _repeated_lines(per_page)
    out: List[Line] = []
    for number, lines in per_page:
        for ln in lines:
            if ln.strip() and _line_key(ln) in repeated:
                continue
            out.append(Line(ln, number))
    return out, toc


_RX_PAGE_OF = re.compile(r"(?:trang|page)\s*\d+\s*(?:(?:/|of|tren)\s*\d+|$)", re.I)


def _line_key(line: str) -> str:
    # Chỉ bỏ cụm "Trang 3/50", "Page 3 of 50": thay mọi chữ số sẽ làm đề mục "1.2.1 ..." của các trang trùng nhau
    return _RX_PAGE_OF.sub("#", fold(line.strip()))


def _repeated_lines(per_page: List[Tuple[int, List[str]]]) -> set:
    """Header/footer: dòng ngắn (không phải hàng bảng) xuất hiện trên >50% số trang, tài liệu >= 3 trang."""
    if len(per_page) < 3:
        return set()
    counts: Counter = Counter()
    for _, lines in per_page:
        counts.update({_line_key(ln) for ln in lines if ln.strip() and not ln.lstrip().startswith("|") and len(ln) < 150})
    return {k for k, c in counts.items() if c > len(per_page) / 2}


# ------------------------------------------------------------------------------
# 2. Khối thể thức hành chính
# ------------------------------------------------------------------------------
LOAI_VB_TITLES: Dict[str, str] = {
    "quyet dinh": "QĐ", "cong van": "CV", "thong bao": "TB", "bao cao": "BC", "to trinh": "TTr",
    "giay moi": "GM", "ke hoach": "KH", "hop dong": "HĐ", "bien ban": "BB", "quy che": "QC",
    "quy dinh": "QyĐ", "huong dan": "HD", "chuong trinh": "CTr", "phuong an": "PA", "de an": "ĐA",
    "nghi quyet": "NQ", "chi thi": "CT", "giay phep": "GP", "giay chung nhan": "GCN",
}
_SO_HIEU_TYPES = {"TT": "TTr", "TTR": "TTr", "QD": "QĐ", "HD": "HĐ"}
_RX_QUOC_HIEU = re.compile(r"cong hoa xa hoi chu nghia viet nam|doc lap\s*[-–]\s*tu do\s*[-–]\s*hanh phuc")
_RX_SO = re.compile(r"^\s*(?:\*\*)?s[oố]\s*(?:hiệu)?\s*[:.]\s*(.+)$", re.I)
_RX_VV = re.compile(r"^\s*(?:\*\*)?(?:v/v|v\.v|vv|về việc)\s*[:.]?\s*(.+)$", re.I)
_RX_KINH_GUI = re.compile(r"^\s*(?:\*\*)?kính\s+gửi\s*[:.]?\s*(.*)$", re.I)
_RX_NOI_NHAN = re.compile(r"^\s*(?:\*\*)?nơi\s+nhận\s*[:.]?\s*(.*)$", re.I)
_RX_DATE_VN = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.I)
_RX_DATE_NUM = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")
_RX_SIGNER_ROLE = re.compile(
    r"^(?:KT\.|TL\.|TM\.|Q\.|PHÓ\s+)?\s*(TỔNG GIÁM ĐỐC|GIÁM ĐỐC|TRƯỞNG PHÒNG|PHÓ TRƯỞNG PHÒNG|TRƯỞNG BAN|CHỦ TỊCH|"
    r"PHÓ GIÁM ĐỐC|PHÓ TỔNG GIÁM ĐỐC|TRƯỞNG ĐOÀN|THỦ TRƯỞNG|NGƯỜI LẬP|ĐẠI DIỆN)\b"
)


def _is_upper_line(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.85


def iso_date(d: int, m: int, y: int) -> str:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return ""


def split_loai_vb(text: str) -> Tuple[str, str]:
    """"BIÊN BẢN KIỂM ĐẾM THIẾT BỊ" -> ("BB", "KIỂM ĐẾM THIẾT BỊ"); không phải tên loại văn bản -> ("", "")."""
    plain = text.strip(" #*:")
    key = fold(plain)
    for title, code in LOAI_VB_TITLES.items():
        if key == title or key.startswith(title + " "):
            return code, " ".join(plain.split()[len(title.split()):])
    return "", ""


def loai_vb_from_text(text: str) -> str:
    return split_loai_vb(text)[0]


def loai_vb_from_so_hieu(so_hieu: str) -> str:
    m = re.search(r"[/-]([A-ZĐ]{1,4}[a-z]?)(?:[-/]|$)", so_hieu)
    if not m:
        return ""
    raw = m.group(1).upper()
    return _SO_HIEU_TYPES.get(fold(raw).upper(), raw)


def extract_admin(lines: List[Line], head_lines: int = 30) -> Tuple[AdminInfo, List[Line]]:
    """Tách quốc hiệu, cơ quan, số hiệu, ngày, V/v, Kính gửi (đầu văn bản) và Nơi nhận, người ký (cuối văn bản)."""
    info = AdminInfo()
    drop = set()
    nonblank = [i for i, ln in enumerate(lines) if ln.text.strip()]
    head = nonblank[:head_lines]
    agency: List[str] = []
    first_page = lines[nonblank[0]].page if nonblank else 1

    for idx in head:
        if idx in drop:
            continue
        plain = lines[idx].text.strip().strip("#* ").strip()
        folded = fold(plain)
        if lines[idx].page != first_page:
            break
        if _RX_QUOC_HIEU.search(folded):
            drop.add(idx)
            if "quoc_hieu" not in info.markers:
                info.markers.append("quoc_hieu")
        elif (m := _RX_SO.match(plain)) and not info.so_hieu:
            codes = extract_codes(m.group(1))
            info.so_hieu = codes[0] if codes else normalize_code(m.group(1).split()[0])
            info.markers.append("so")
            drop.add(idx)
        elif (m := _RX_VV.match(plain)) and not info.trich_yeu:
            parts = [m.group(1).strip()]
            drop.add(idx)
            # trích yếu có thể xuống dòng tới khi gặp dòng trống / Kính gửi
            j = idx + 1
            while j < len(lines) and lines[j].text.strip() and not _RX_KINH_GUI.match(lines[j].text) and len(parts) < 4:
                parts.append(lines[j].text.strip())
                drop.add(j)
                j += 1
            info.trich_yeu = " ".join(parts).strip(" .")
            info.markers.append("vv")
        elif (m := _RX_KINH_GUI.match(plain)) and not info.kinh_gui:
            drop.add(idx)
            info.markers.append("kinh_gui")
            if m.group(1).strip():
                info.kinh_gui.append(m.group(1).strip(" ;.-"))
            j = idx + 1
            while j < len(lines) and re.match(r"^\s*[-–•+]\s*\S", lines[j].text):
                info.kinh_gui.append(lines[j].text.strip(" ;.-–•+"))
                drop.add(j)
                j += 1
        elif (m := _RX_DATE_VN.search(plain)) and not info.ngay_ban_hanh and len(plain) < 80:
            info.ngay_ban_hanh = iso_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            drop.add(idx)
        elif not info.so_hieu and not info.trich_yeu and _is_upper_line(plain) and not loai_vb_from_text(plain) and len(plain) < 120:
            agency.append(plain)
            drop.add(idx)
        elif not info.loai_vb and len(plain) < 80 and split_loai_vb(plain)[0]:
            info.loai_vb, rest = split_loai_vb(plain)
            if rest and not info.trich_yeu:
                info.trich_yeu = rest
            drop.add(idx)

    if agency:
        info.co_quan = agency[0]
        info.don_vi = agency[-1] if len(agency) > 1 else agency[0]

    # Cuối văn bản: "Nơi nhận" và chữ ký
    for idx in range(len(lines) - 1, max(-1, len(lines) - 60), -1):
        m = _RX_NOI_NHAN.match(lines[idx].text.strip())
        if not m:
            continue
        info.markers.append("noi_nhan")
        for j in range(idx, len(lines)):
            text = lines[j].text.strip()
            drop.add(j)
            if j > idx and re.match(r"^[-–•+]\s*\S", text):
                info.noi_nhan.append(text.strip(" ;.-–•+"))
        break
    _extract_signer(lines, info, drop)

    if not info.loai_vb and info.so_hieu:
        info.loai_vb = loai_vb_from_so_hieu(info.so_hieu)
    body = [ln for k, ln in enumerate(lines) if k not in drop]
    return info, body


def _extract_signer(lines: List[Line], info: AdminInfo, drop: set) -> None:
    """Người ký: dòng chức danh viết hoa (TRƯỞNG PHÒNG, KT. GIÁM ĐỐC...) và dòng họ tên ngay sau."""
    tail = [k for k in range(max(0, len(lines) - 25), len(lines)) if lines[k].text.strip()]
    for pos, k in enumerate(tail):
        text = lines[k].text.strip().strip("*# ")
        if _RX_SIGNER_ROLE.match(text) and len(text) < 60 and text != "ĐẠI DIỆN":
            drop.add(k)
            for n in tail[pos + 1:pos + 4]:
                cand = lines[n].text.strip().strip("*# ")
                drop.add(n)
                if re.fullmatch(r"[A-ZĐÀ-Ỹ][a-zà-ỹđ]+(?:\s+[A-ZĐÀ-Ỹ][a-zà-ỹđ]+){1,4}", cand):
                    info.nguoi_ky = cand
                    break
            return


def admin_from_filename(name: str) -> Tuple[str, str]:
    """Tên file văn thư thường chứa số hiệu + ngày: "470 TB-UBND Ngày 1452018" -> ("470/TB-UBND", "2018-05-14")."""
    stem = re.sub(r"\.(pdf|md|docx?)$", "", name, flags=re.I)
    so_hieu = ""
    # số hiệu ở bất kỳ đâu trong tên: "470 TB-UBND", "Quyết định số 1963-QĐ-SXD", "Tờ trình số 363-TTr-SXD"
    m = re.search(r"(?<![\d.])(\d{1,5})[\s._/-]*([A-ZĐ][A-ZĐa-z]{0,6}(?:\s*[-&]\s*[A-ZĐ0-9&]+)+)", stem)
    if m:
        so_hieu = normalize_code(f"{m.group(1)}/{m.group(2)}")
    ngay = ""
    d = re.search(r"ngày\s*(\d{1,2})[./-]?(\d{1,2})[./-]?(\d{4})", stem, re.I)
    if d:
        ngay = iso_date(int(d.group(1)), int(d.group(2)), int(d.group(3)))
    else:
        d = re.search(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{2}|\d{4})(?![\d.])", stem)   # "14.10.26", "30.7.2026"
        if d:
            year = int(d.group(3)) + (2000 if len(d.group(3)) == 2 else 0)
            ngay = iso_date(int(d.group(1)), int(d.group(2)), year)
    return so_hieu, ngay


# ------------------------------------------------------------------------------
# 3. Khối đề mục / đoạn văn / bảng
# ------------------------------------------------------------------------------
_RX_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_RX_CHUONG = re.compile(r"^(?:\*\*)?(Chương|CHƯƠNG)\s+[IVXLC\d]+\b", re.U)
_RX_MUC = re.compile(r"^(?:\*\*)?(Mục|MỤC)\s+\d+\b")
_RX_DIEU = re.compile(r"^(?:\*\*)?(Điều|ĐIỀU)\s+\d+[a-z]?\s*[.:]", re.U)
_RX_PHU_LUC = re.compile(r"^(?:\*\*)?(PHỤ LỤC|Phụ lục)\b")
_RX_ROMAN = re.compile(r"^(?:\*\*)?([IVX]{1,5})\.\s+\S")
_RX_NUMBERED = re.compile(r"^(?:\*\*)?(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+[A-ZĐÀ-Ỹa-zđà-ỹ]")
_RX_BUOC = re.compile(r"^(?:\*\*)?(Bước|BƯỚC)\s+\d+")
_RX_BOLD_LINE = re.compile(r"^\*\*([^*]{3,100})\*\*:?$")


def classify_heading(text: str) -> Optional[Tuple[int, str, str]]:
    """Trả về (cấp, tiêu đề, loại pháp quy) nếu dòng là đề mục."""
    s = text.strip()
    if not s or len(s) > 160 or s.startswith("|"):
        return None
    m = _RX_MD_HEADING.match(s)
    if m:
        title = m.group(2).strip("* ")
        legal = "dieu" if _RX_DIEU.match(title) else ("chuong" if _RX_CHUONG.match(title) else "")
        return len(m.group(1)), title, legal
    if _RX_CHUONG.match(s):
        return 1, s.strip("* "), "chuong"
    if _RX_PHU_LUC.match(s) and len(s) < 100:
        return 1, s.strip("* "), "phu_luc"
    if _RX_MUC.match(s):
        return 2, s.strip("* "), ""
    if _RX_DIEU.match(s):
        return 3, s.strip("* "), "dieu"
    if s.endswith((";", ",")) or len(s) > 120:
        return None
    if _RX_ROMAN.match(s):
        return 2, s.strip("* "), ""
    m = _RX_NUMBERED.match(s)
    if m:
        return 2 + m.group(1).count(".") + 1, s.strip("* "), ""
    if _RX_BUOC.match(s):
        return 5, s.strip("* "), ""
    m = _RX_BOLD_LINE.match(s)
    if m:
        return 5, m.group(1).strip(), ""
    return None


def build_blocks(lines: List[Line]) -> List[Block]:
    blocks: List[Block] = []
    buf: List[Line] = []

    def flush_text() -> None:
        if any(ln.text.strip() for ln in buf):
            text = "\n".join(ln.text for ln in buf).strip()
            blocks.append(Block("text", text, buf[0].page, buf[-1].page))
        buf.clear()

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.text.strip()
        if s.startswith("|"):
            flush_text()
            j = i
            rows: List[Line] = []
            while j < len(lines) and (lines[j].text.strip().startswith("|") or not lines[j].text.strip()):
                if lines[j].text.strip():
                    if rows and lines[j].page != rows[-1].page:
                        break  # sang trang mới: để _append_table ghép và bỏ hàng tiêu đề lặp
                    rows.append(lines[j])
                j += 1
            _append_table(blocks, "\n".join(r.text.strip() for r in rows), rows[0].page, rows[-1].page)
            i = j
            continue
        if s.lower().startswith("<table"):
            flush_text()
            j = i
            parts = []
            while j < len(lines):
                parts.append(lines[j].text)
                if "</table>" in lines[j].text.lower():
                    break
                j += 1
            _append_table(blocks, "\n".join(parts).strip(), ln.page, lines[min(j, len(lines) - 1)].page)
            i = j + 1
            continue
        heading = classify_heading(s)
        if heading:
            flush_text()
            level, title, legal = heading
            blocks.append(Block("heading", title, ln.page, ln.page, level=level, legal=legal))
        else:
            buf.append(ln)
        i += 1
    flush_text()
    return blocks


def _append_table(blocks: List[Block], text: str, page_from: int, page_to: int) -> None:
    """Ghép bảng vắt qua trang: bảng liền sau bảng cùng số cột (chỉ cách nhau ranh giới trang) được nối lại."""
    prev = blocks[-1] if blocks else None
    if prev and prev.kind == "table" and page_from > prev.page_to and not text.lower().startswith("<table"):
        prev_header, _ = table_rows(prev.text)
        header, rows = table_rows(text)
        if prev_header and len(header) == len(prev_header):
            body_lines = text.splitlines()
            # tiêu đề lặp ở trang sau có thể lệch vài ký tự do OCR ("Dơn vị" / "Đơn vị")
            same = difflib.SequenceMatcher(None, fold(" ".join(header)), fold(" ".join(prev_header))).ratio() >= 0.85
            if same:
                body_lines = body_lines[2:] if len(body_lines) > 1 and "---" in body_lines[1] else body_lines[1:]
            prev.text = prev.text + "\n" + "\n".join(body_lines)
            prev.page_to = page_to
            return
    blocks.append(Block("table", text, page_from, page_to))


# ------------------------------------------------------------------------------
# 4. Bảng: đọc hàng, làm phẳng, chia nhỏ
# ------------------------------------------------------------------------------
def _split_md_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


class _HTMLTable(HTMLParser):
    """Đọc <table> có rowspan/colspan thành lưới đầy đủ (ô gộp được nhân ra từng ô)."""

    def __init__(self) -> None:
        super().__init__()
        self.grid: List[List[str]] = []
        self._row: Optional[List[Tuple[str, int, int]]] = None
        self._cell: Optional[List[str]] = None
        self._span = (1, 1)
        self._next_row = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._span = (int(a.get("rowspan") or 1), int(a.get("colspan") or 1))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append((" ".join("".join(self._cell).split()), *self._span))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._place(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def _place(self, cells: List[Tuple[str, int, int]]) -> None:
        r = self._next_row
        while len(self.grid) <= r:
            self.grid.append([])
        col = 0
        for text, rowspan, colspan in cells:
            while col < len(self.grid[r]) and self.grid[r][col] is not None:
                col += 1
            for dr in range(rowspan):
                while len(self.grid) <= r + dr:
                    self.grid.append([])
                row = self.grid[r + dr]
                while len(row) < col + colspan:
                    row.append(None)
                for dc in range(colspan):
                    row[col + dc] = text
            col += colspan
        self._next_row += 1


def table_rows(text: str) -> Tuple[List[str], List[List[str]]]:
    """Trả về (hàng tiêu đề, các hàng dữ liệu) cho bảng Markdown hoặc HTML."""
    if text.lstrip().lower().startswith("<table"):
        parser = _HTMLTable()
        parser.feed(text)
        grid = [[c or "" for c in row] for row in parser.grid if row]
        return (grid[0], grid[1:]) if grid else ([], [])
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return [], []
    header = _split_md_row(lines[0])
    rows = [_split_md_row(ln) for ln in lines[1:] if not re.fullmatch(r"\|?[\s:|-]+\|?", ln.strip())]
    return header, rows


def flatten_table(text: str, caption: str = "") -> str:
    """
    Làm phẳng bảng để đánh chỉ mục: mỗi hàng ghi rõ tên cột. Ô trống ở cột đầu (ô gộp dọc trong
    bảng Markdown) được điền từ hàng trên để con số không bị mất ngữ cảnh.
    """
    header, rows = table_rows(text)
    if not header:
        return text
    out = [caption] if caption else []
    prev: List[str] = []
    for row in rows:
        filled = []
        for k, cell in enumerate(row):
            if not cell and k < 2 and k < len(prev):
                cell = prev[k]
            filled.append(cell)
        prev = filled
        pairs = [f"{header[k] if k < len(header) and header[k] else f'Cột {k + 1}'}: {v}" for k, v in enumerate(filled) if v]
        if pairs:
            out.append("; ".join(pairs))
    return "\n".join(out)


def split_table(text: str, rows_per_chunk: int) -> List[str]:
    """Chia bảng lớn theo hàng, lặp lại hàng tiêu đề ở mỗi phần; không cắt giữa hàng."""
    header, rows = table_rows(text)
    if not header or len(rows) <= rows_per_chunk:
        return [text]
    head = "| " + " | ".join(header) + " |\n|" + "---|" * len(header)
    return [
        head + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows[i:i + rows_per_chunk])
        for i in range(0, len(rows), rows_per_chunk)
    ]
