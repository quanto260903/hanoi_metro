"""
Hanoi Metro AI Agent - Bổ sung metadata cho văn bản (đối chiếu bộ 120 tiêu chí, phần ưu tiên 1).

Không sửa DocRecord, không tăng CHUNKER_VERSION: kết quả nằm ở bảng riêng trong kho văn bản
(doc_fields, doc_relations, doc_tasks), khóa doc_id; trường dùng để lọc được ghi thêm vào payload Qdrant
bằng cập nhật payload (không nhúng lại). Mỗi giá trị có nguồn (luat / ai / duyet), trang chứng cứ và phiên bản bộ trích.

Bằng luật (không gọi API):
  huong_vb              di / den / noi_bo / khong_ro    (thư mục "CV ĐI"/"CV ĐẾN", cơ quan ban hành, nơi nhận)
  doc_relations         can_cu, tra_loi, thay_the, bai_bo, sua_doi, kem_theo, lien_quan
  hieu_luc_tu           "có hiệu lực (kể) từ ngày …", "kể từ ngày ký"
  trang_thai_hieu_luc   con_hieu_luc / bi_thay_the / bai_bo / khong_ro (đảo chiều quan hệ), thay_the_boi
  ty_le_trang_scan      pages_routed.csv (page_router)
  ty_le_doc_duoc, trang_can_scan_lai   sổ cái VLM (trang rỗng, chạm trần token, nhiều "[không rõ]")

Bằng AI (mặc định TẮT, cờ --ai): một lượt gọi mỗi văn bản → phòng chủ trì / phối hợp (kèm lý do) và nhiệm vụ
(ai, làm gì, hạn, trang). Gợi ý phòng chỉ ghi vào file duyệt nhãn (cột phong_goi_y), KHÔNG áp vào phân quyền.

  python -m src.indexing.enrich chay [--ai] [--limit N] [--ngau-nhien] [--doc DOC_ID]
  python -m src.indexing.enrich xuat-kiem-tra [--out data/access/kiem_tra_enrich.csv]
  python -m src.indexing.enrich thong-ke
"""

import argparse
import csv
import json
import logging
import re
import sqlite3
import sys
import threading
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from src.indexing.models import DocRecord
from src.store.base import NGUON_AI, NGUON_LUAT, DocField, DocRelation, DocTask, Filters
from src.text.codes import extract_codes, normalize_code
from src.text.normalize import approx_tokens, fold

logger = logging.getLogger("hanoi_metro.enrich")

RULES_VERSION = "luat-v2"   # v2: "đến" cần số hiệu …/… + tên cơ quan thật; bỏ quan hệ trong văn bản góp ý dự thảo
AI_VERSION = "ai-v8"      # v8: hậu tố -2A là phiên bản, mã tài liệu theo tuyến, tuyển dụng → TCNS, môi trường → ATCL; v7: v7: đưa hướng văn bản (đến/đi) vào đầu vào, văn bản đến luôn có phòng xử lý; v6: v6: thêm phong_can_xem (nhiều phòng) + cờ chung, bắt buộc chủ trì khi văn bản liên quan Công ty; v5: nhiệm vụ chỉ là việc được GIAO cho Công ty (không phải bước thao tác kỹ thuật); v4: chỉ MỘT phòng chủ trì; v3: định nghĩa rõ chủ trì (không phải người ký); v2 chưa định nghĩa; v1 không suy luận
# Phiên bản cũ mà kết quả vẫn dùng được với bản hiện tại (cùng định nghĩa; khác biệt đã được chuẩn hóa khi đọc kết quả):
# văn bản đã chạy ở các phiên bản này không bị gọi lại.
AI_COMPATIBLE: tuple = ()   # v6 cần chạy lại mọi văn bản (trường phong_can_xem mới)
AI_PREVIOUS = ("ai-v7", "ai-v6", "ai-v5", "ai-v4", "ai-v3")   # tinh-lai-ai: văn bản chưa có v6 thì đọc lại phản hồi cũ

HUONG_VB = ("di", "den", "noi_bo", "khong_ro")
RELATION_TYPES = ("can_cu", "tra_loi", "thay_the", "bai_bo", "sua_doi", "kem_theo", "lien_quan")
STATUS = ("con_hieu_luc", "bi_thay_the", "bai_bo", "khong_ro")
PAYLOAD_FIELDS = ("huong_vb", "trang_thai_hieu_luc")
UNCLEAR_MARK = "[không rõ]"
UNCLEAR_LIMIT = 3                 # số "[không rõ]" trên một trang để coi là trang đọc kém


# ====================================================================== nội dung theo trang
def document_pages(store, doc_id: str) -> Dict[int, str]:
    """Nội dung văn bản theo trang (từ chunk đã lưu trong kho; số trang tính từ 1)."""
    pages: Dict[int, List[str]] = {}
    for ch in store.get_document_chunks(doc_id):
        pages.setdefault(ch.page_from, []).append(ch.text)
    return {p: "\n".join(parts) for p, parts in sorted(pages.items())}


# ====================================================================== 1. hướng văn bản
_RX_FOLDER = re.compile(r"\b(?:cv|cong van|van ban|vb)\s*(di|den)\b")
_DSHN = ("duong sat ha noi", "dshn")
_EXTERNAL = ("uy ban", "ubnd", "so ", "bo ", "cuc ", "ban quan ly", "tong cong ty", "lien danh", "nha thau",
             "tu van", "chinh phu", "thanh uy", "kho bac", "ngan hang", "cong an", "quan ", "huyen ", "phuong ")
_INTERNAL = ("phong ", "xi nghiep", "cac don vi", "tong giam doc", "hoi dong thanh vien", "chu tich hdtv",
             "doi ", "van phong cong ty", "ban dieu hanh", "chi nhanh")


def issued_by_company(doc: DocRecord) -> bool:
    text = fold(f"{doc.co_quan} {doc.don_vi} {doc.so_hieu}")
    return any(k in text for k in _DSHN)


def _recipient_kind(name: str) -> str:
    f = f" {fold(name)} "
    if any(k in f for k in _DSHN) and "cong ty" in f and "chi nhanh" not in f:
        return "noi_bo"           # gửi chính Công ty (vd văn bản của phòng trình Công ty)
    if any(f" {k}" in f for k in _EXTERNAL) or ("cong ty" in f and not any(k in f for k in _DSHN)):
        return "ngoai"
    if any(k in f for k in _INTERNAL):
        return "noi_bo"
    return ""


_AGENCY = ("bo ", "so ", "ubnd", "uy ban", "ban ", "cong ty", "trung tam", "cuc ", "vien ", "hoi dong", "van phong",
           "chinh phu", "thanh uy", "tap doan", "lien danh", "quoc hoi", "toa an", "kho bac", "ngan hang", "dien luc",
           "cong an", "bao hiem", "phuong ", "xa ", "quan ", "huyen ", "doan ", "hiep hoi", "truong ", "chi cuc",
           "thanh tra", "kiem toan", "tong cuc", "nha xuat ban", "benh vien", "thue ")


def looks_like_agency(name: str) -> bool:
    """Tên cơ quan thật (Bộ, Sở, UBND, Công ty…), không phải tiêu đề trang hay ô bảng bóc nhầm."""
    if any(ch in name for ch in "|*#<>"):
        return False
    f = f" {fold(name)} "
    return any(f" {k}" in f for k in _AGENCY)


def detect_huong_vb(doc: DocRecord) -> DocField:
    folders = [fold(p) for p in re.split(r"[\\/]", doc.source_path)[:-1]]
    for seg in reversed(folders):
        m = _RX_FOLDER.search(seg) or re.fullmatch(r"\s*(di|den)\s*", seg)
        if m:
            return DocField("huong_vb", m.group(1), NGUON_LUAT, None, "cao", RULES_VERSION, f"thư mục '{seg}'")
    if issued_by_company(doc):
        kinds = {_recipient_kind(k) for k in doc.kinh_gui if k.strip()}
        if "ngoai" in kinds:
            return DocField("huong_vb", "di", NGUON_LUAT, 1, "trung_binh", RULES_VERSION, "Công ty ban hành, gửi đơn vị ngoài")
        if kinds == {"noi_bo"}:
            return DocField("huong_vb", "noi_bo", NGUON_LUAT, 1, "trung_binh", RULES_VERSION, "Công ty ban hành, gửi đơn vị nội bộ")
        return DocField("huong_vb", "khong_ro", NGUON_LUAT, None, "", RULES_VERSION, "Công ty ban hành, không rõ nơi nhận")
    issuer = (doc.co_quan or doc.don_vi).strip()
    if issuer and "/" in doc.so_hieu and looks_like_agency(issuer):
        return DocField("huong_vb", "den", NGUON_LUAT, 1, "trung_binh", RULES_VERSION, f"cơ quan ban hành: {issuer}")
    # tài liệu kỹ thuật, sổ tay nhà thầu…: "cơ quan ban hành" bóc tách được thường là tiêu đề trang, không phải cơ quan
    return DocField("huong_vb", "khong_ro", NGUON_LUAT, None, "", RULES_VERSION,
                    "không có số hiệu văn bản hoặc tên cơ quan ban hành" if issuer else "")


# ====================================================================== 2. quan hệ văn bản
_RX_CLAUSE_END = re.compile(r";|\n\s*\n|\.\s*\n|\.\s+(?=[A-ZĐ])")
_PARTIAL = re.compile(r"\b(noi dung|dieu|khoan|diem|muc|phu luc|mot so|cac quy dinh tai)\b")
# thứ tự quan trọng: bãi bỏ / thay thế / sửa đổi trước "kèm theo"
_KEYWORDS = (
    ("bai_bo", re.compile(r"\bbai bo\b")),
    ("thay_the", re.compile(r"\bthay the\b")),
    ("sua_doi", re.compile(r"\bsua doi\b")),
    ("tra_loi", re.compile(r"\b(?:phuc dap|tra loi|phan hoi)\b")),
    ("kem_theo", re.compile(r"\bkem theo\b")),
)
_RX_CAN_CU = re.compile(r"(?m)^[\s>*_#-]*can cu\b")
# văn bản / mệnh đề nói về DỰ THẢO hay xin ý kiến: "thay thế/bãi bỏ" là nội dung của dự thảo, không phải của văn bản này
_RX_DRAFT = re.compile(r"\b(du thao|lay y kien|xin y kien|gop y|tham gia y kien)\b")
_EFFECT_KINDS = ("bai_bo", "thay_the", "sua_doi")


def _aligned_fold(text: str) -> str:
    """Bỏ dấu, chữ thường, GIỮ NGUYÊN độ dài để vị trí trong bản bỏ dấu dùng được cho văn bản gốc."""
    folded = fold(text)
    if len(folded) == len(text):
        return folded
    return "".join((fold(ch) or " ")[0] for ch in text)


def _clause_end(text: str, start: int) -> int:
    """Hết mệnh đề: dấu ;, dòng trống, dấu chấm cuối dòng hoặc trước chữ hoa (tìm trên văn bản gốc, giữ chữ hoa)."""
    m = _RX_CLAUSE_END.search(text, start)
    return m.start() if m else len(text)


def _doc_codes(text: str) -> List[str]:
    """Số hiệu văn bản (có '/') trong một đoạn."""
    return [c for c in extract_codes(text) if "/" in c]


def extract_relations(doc: DocRecord, pages: Dict[int, str]) -> List[DocRelation]:
    own = normalize_code(doc.so_hieu) if doc.so_hieu else ""
    found: Dict[Tuple[str, str], DocRelation] = {}

    def add(loai: str, code: str, page: int) -> None:
        norm = normalize_code(code)
        if norm and norm != own and (loai, norm) not in found:
            found[(loai, norm)] = DocRelation(loai, code, "", page, NGUON_LUAT, doc.doc_id)

    about_draft = bool(_RX_DRAFT.search(fold(f"{doc.trich_yeu} {doc.title}")))
    for page, raw in pages.items():
        text = unicodedata.normalize("NFC", raw)
        folded = _aligned_fold(text)
        used: set = set()                        # số hiệu đã thuộc dòng "Căn cứ" của trang này
        for m in _RX_CAN_CU.finditer(folded):
            for code in _doc_codes(text[m.end():_clause_end(text, m.end())]):
                add("can_cu", code, page)
                used.add(normalize_code(code))
        for loai, rx in _KEYWORDS:
            for m in rx.finditer(folded):
                end = _clause_end(text, m.end())
                targets = [c for c in _doc_codes(text[m.end():end]) if normalize_code(c) not in used]
                if not targets:
                    continue
                if loai in _EFFECT_KINDS:
                    begin = max(text.rfind(";", 0, m.start()), text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start()))
                    if about_draft or _RX_DRAFT.search(folded[begin + 1:m.start()]):
                        continue                 # "dự thảo Quyết định thay thế…": chưa có hiệu lực, không ghi quan hệ
                kind = loai
                lead = re.split(r"\d+\s*/", folded[m.end():end], maxsplit=1)[0]
                if loai in ("bai_bo", "thay_the") and _PARTIAL.search(lead):
                    kind = "sua_doi"             # bãi bỏ/thay thế MỘT PHẦN (điều, khoản…) = sửa đổi, không mất hiệu lực
                for code in targets:
                    add(kind, code, page)
    named = {norm for _, norm in found}
    for code in doc.cites:
        if "/" in code and normalize_code(code) not in named:
            add("lien_quan", code, None)
    return list(found.values())


def link_targets(relations: Sequence[DocRelation], store) -> None:
    """Điền doc_id_dich khi số hiệu đích có trong kho (chỉ khớp chính xác, không khớp gần đúng)."""
    for r in relations:
        norm = normalize_code(r.so_hieu_dich)
        exact = [d for d in store.find_documents_by_code(r.so_hieu_dich) if normalize_code(d.so_hieu) == norm]
        r.doc_id_dich = exact[0].doc_id if len(exact) == 1 else ""


# ====================================================================== 3. ngày hiệu lực, trạng thái
_RX_EFFECT = re.compile(r"co hieu luc(?: thi hanh)?(?: ke)? tu ngay\s+(?:"
                        r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})"
                        r"|(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})"
                        r"|(ky|ban hanh))")


def extract_effective_date(doc: DocRecord, pages: Dict[int, str]) -> Optional[DocField]:
    for page, text in pages.items():
        m = _RX_EFFECT.search(fold(text))
        if not m:
            continue
        if m.group(7):
            if not doc.ngay_ban_hanh:
                continue
            return DocField("hieu_luc_tu", doc.ngay_ban_hanh, NGUON_LUAT, page, "cao", RULES_VERSION, "kể từ ngày ký")
        d, mth, y = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(4), m.group(5), m.group(6))
        try:
            iso = datetime(int(y), int(mth), int(d)).date().isoformat()
        except ValueError:
            continue
        return DocField("hieu_luc_tu", iso, NGUON_LUAT, page, "cao", RULES_VERSION, "")
    return None


# ký hiệu loại văn bản đứng trong số hiệu (không phải mã cơ quan)
_DOC_KINDS = {"QD", "NQ", "TT", "ND", "CV", "TB", "BC", "TTR", "KH", "GM", "BB", "HD", "QC", "CT", "QH", "TTG", "UBTVQH",
              "PL", "L", "HDLT", "TTLT", "CD", "GUQ", "UQ", "PA", "DA", "KL", "BKTT"}


def issuing_agencies(so_hieu: str) -> set:
    """Mã cơ quan trong số hiệu: 2068/QĐ-UBND → {UBND}; 457/TTĐHGT-KTDV → {TTĐHGT, KTDV}; rỗng nếu không đọc được."""
    if "/" not in (so_hieu or ""):
        return set()
    tail = so_hieu.split("/")[-1]
    parts = {fold(x).upper().replace("&", "") for x in re.split(r"[-–]", tail) if x.strip()}
    return {x for x in parts if x and not x.isdigit() and x not in _DOC_KINDS}


def compute_status(doc: DocRecord, incoming: Sequence[DocRelation], sources: Dict[str, DocRecord],
                   has_effective_date: bool) -> List[DocField]:
    """Đảo chiều quan hệ: văn bản khác (mới hơn hoặc cùng ngày) thay thế / bãi bỏ văn bản này."""
    by_kind: Dict[str, List[str]] = {"bai_bo": [], "thay_the": []}
    for r in incoming:
        src = sources.get(r.doc_id)
        if r.loai not in by_kind or src is None or src.doc_id == doc.doc_id:
            continue
        if src.ngay_ban_hanh and doc.ngay_ban_hanh and src.ngay_ban_hanh < doc.ngay_ban_hanh:
            continue                     # văn bản cũ hơn không thể thay thế văn bản mới: nhiều khả năng đọc nhầm
        a_src, a_dst = issuing_agencies(src.so_hieu), issuing_agencies(doc.so_hieu or r.so_hieu_dich)
        if a_src and a_dst and not (a_src & a_dst):
            continue                     # khác cơ quan ban hành (công văn của Trung tâm không thay thế QĐ của UBND)
        by_kind[r.loai].append(src.so_hieu or src.source_file)
    if by_kind["bai_bo"]:
        status, who = "bai_bo", by_kind["bai_bo"]
    elif by_kind["thay_the"]:
        status, who = "bi_thay_the", by_kind["thay_the"]
    else:
        status = "con_hieu_luc" if has_effective_date else "khong_ro"
        note = "không thấy văn bản thay thế / bãi bỏ trong kho" if has_effective_date else ""
        return [DocField("trang_thai_hieu_luc", status, NGUON_LUAT, None, "", RULES_VERSION, note),
                DocField("thay_the_boi", "", NGUON_LUAT, None, "", RULES_VERSION, "")]
    who = list(dict.fromkeys(who))
    return [DocField("trang_thai_hieu_luc", status, NGUON_LUAT, None, "cao", RULES_VERSION, ""),
            DocField("thay_the_boi", ";".join(who), NGUON_LUAT, None, "cao", RULES_VERSION, "")]


# ====================================================================== 4. chất lượng bóc tách
class RoutingTable:
    """Luồng từng trang theo page_router (pages_routed.csv, trang tính từ 0)."""

    SCAN = {"STANDARD_OCR", "VLM"}

    def __init__(self, routes: Dict[str, Dict[int, str]]):
        self.routes = routes

    @classmethod
    def load(cls, results_dir: Path) -> "RoutingTable":
        path = Path(results_dir) / "pages_routed.csv"
        routes: Dict[str, Dict[int, str]] = {}
        if not path.is_file():
            logger.warning(f"Không có {path}: bỏ qua tỷ lệ trang scan.")
            return cls(routes)
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                routes.setdefault(r["md5"].lower(), {})[int(r["page"])] = r["route"]
        return cls(routes)

    def scan_ratio(self, md5: str) -> Optional[float]:
        pages = {p: r for p, r in self.routes.get(md5, {}).items() if r != "BLANK"}
        if not pages:
            return None
        return sum(r in self.SCAN for r in pages.values()) / len(pages)

    def api_pages(self, md5: str) -> List[int]:
        """Trang (tính từ 1) cần nhận dạng qua API."""
        return sorted(p + 1 for p, r in self.routes.get(md5, {}).items() if r == "VLM")


class LedgerQuality:
    """Đọc sổ cái VLM (chỉ đọc): bản ghi mới nhất của từng trang (trang tính từ 1)."""

    def __init__(self, pages: Dict[str, Dict[int, Tuple[str, int]]], max_tokens: int):
        self.pages, self.max_tokens = pages, max_tokens

    @classmethod
    def load(cls, path: Path, max_tokens: int) -> "LedgerQuality":
        pages: Dict[str, Dict[int, Tuple[str, int]]] = {}
        if not Path(path).is_file():
            return cls(pages, max_tokens)
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
        try:
            for md5, page, md, out_tokens in conn.execute(
                    "SELECT md5, page, markdown, out_tokens FROM vlm_pages ORDER BY created_at"):
                pages.setdefault(md5.lower(), {})[int(page)] = (md or "", int(out_tokens or 0))
        finally:
            conn.close()
        return cls(pages, max_tokens)

    def bad_pages(self, md5: str) -> List[int]:
        bad = []
        for page, (md, out_tokens) in self.pages.get(md5, {}).items():
            if not md.strip() or (self.max_tokens and out_tokens >= 0.98 * self.max_tokens) \
                    or md.count(UNCLEAR_MARK) >= UNCLEAR_LIMIT:
                bad.append(page)
        return sorted(bad)

    def done(self, md5: str) -> set:
        return set(self.pages.get(md5, {}))


def quality_fields(doc: DocRecord, routing: RoutingTable, ledger: LedgerQuality) -> List[DocField]:
    out: List[DocField] = []
    ratio = routing.scan_ratio(doc.doc_id)
    if ratio is not None:
        out.append(DocField("ty_le_trang_scan", f"{ratio:.2f}", NGUON_LUAT, None, "", RULES_VERSION, "pages_routed.csv"))
    if ratio is None and doc.doc_id not in ledger.pages:
        return out
    bad = ledger.bad_pages(doc.doc_id)
    unread = [p for p in routing.api_pages(doc.doc_id) if p not in ledger.done(doc.doc_id)]
    total = max(doc.total_pages, 1)
    readable = max(0.0, (total - len(set(bad) | set(unread))) / total)
    note = f"{len(unread)} trang chưa nhận dạng" if unread else ""
    out.append(DocField("ty_le_doc_duoc", f"{readable:.2f}", NGUON_LUAT, None, "", RULES_VERSION, note))
    out.append(DocField("trang_can_scan_lai", ",".join(map(str, bad)), NGUON_LUAT, bad[0] if bad else None, "",
                        RULES_VERSION, "trang rỗng, chạm trần token hoặc nhiều chỗ [không rõ]" if bad else ""))
    return out


# ====================================================================== chạy luật
@dataclass
class EnrichStats:
    counts: Counter = field(default_factory=Counter)

    def __str__(self) -> str:
        return ", ".join(f"{k}: {v}" for k, v in sorted(self.counts.items()))


def run_rules(store, docs: Sequence[DocRecord], routing: RoutingTable, ledger: LedgerQuality,
              vector=None, all_docs: Optional[Sequence[DocRecord]] = None) -> EnrichStats:
    stats = EnrichStats()
    touched: set = set()                  # văn bản có trường lọc (payload Qdrant) thay đổi
    for doc in docs:
        pages = document_pages(store, doc.doc_id)
        fields = [detect_huong_vb(doc)]
        eff = extract_effective_date(doc, pages)
        if eff:
            fields.append(eff)
        fields += quality_fields(doc, routing, ledger)
        relations = extract_relations(doc, pages)
        link_targets(relations, store)
        store.set_doc_relations(doc.doc_id, relations, NGUON_LUAT)
        if set(store.set_doc_fields(doc.doc_id, fields)) & set(PAYLOAD_FIELDS):
            touched.add(doc.doc_id)
        stats.counts[f"huong_{fields[0].gia_tri}"] += 1
        stats.counts["quan_he"] += len(relations)
        stats.counts["co_hieu_luc_tu"] += bool(eff)
    # trạng thái hiệu lực cần quan hệ của MỌI văn bản (đảo chiều) → tính sau, trên toàn bộ phạm vi
    scope = list(all_docs) if all_docs is not None else list(docs)
    stats.counts.update(update_status(store, scope, vector, touched))
    return stats


def update_status(store, docs: Sequence[DocRecord], vector=None, touched: Optional[set] = None) -> Counter:
    """
    Tính trạng thái hiệu lực cho cả phạm vi bằng vài truy vấn hàng loạt (đọc mọi quan hệ một lần, nhóm theo đích),
    rồi chỉ cập nhật payload Qdrant cho văn bản có trường lọc thay đổi.
    """
    counts: Counter = Counter()
    ids = [d.doc_id for d in docs]
    effective = {k: "hieu_luc_tu" in v for k, v in store.doc_fields(ids, ["hieu_luc_tu"]).items()}
    by_target: Dict[str, List[DocRelation]] = {}
    for r in store.doc_relations(ids):
        if r.loai in ("thay_the", "bai_bo"):
            if r.doc_id_dich:
                by_target.setdefault(r.doc_id_dich, []).append(r)
            by_target.setdefault("#" + normalize_code(r.so_hieu_dich), []).append(r)
    known = {d.doc_id: d for d in docs}
    changed = set(touched or ())
    for doc in docs:
        incoming = list(by_target.get(doc.doc_id, []))
        if doc.so_hieu:
            incoming += [r for r in by_target.get("#" + normalize_code(doc.so_hieu), []) if r not in incoming]
        missing = [r.doc_id for r in incoming if r.doc_id not in known]
        sources = {**{r.doc_id: known[r.doc_id] for r in incoming if r.doc_id in known}, **store.get_documents(missing)}
        fields = compute_status(doc, incoming, sources, effective.get(doc.doc_id, False))
        if set(store.set_doc_fields(doc.doc_id, fields)) & set(PAYLOAD_FIELDS):
            changed.add(doc.doc_id)
        counts[f"hieu_luc_{fields[0].gia_tri}"] += 1
    if vector is not None and hasattr(vector, "set_doc_payload") and changed:
        if hasattr(vector, "ensure_enrich_indexes"):
            vector.ensure_enrich_indexes()
        current = store.doc_fields(sorted(changed), PAYLOAD_FIELDS)
        for doc_id in sorted(changed):
            vector.set_doc_payload(doc_id, {k: v.gia_tri for k, v in current.get(doc_id, {}).items()})
        counts["cap_nhat_payload"] = len(changed)
    return counts


# ====================================================================== cảnh báo cho trợ lý
def document_warnings(store, doc_ids: Sequence[str], low_read_ratio: float = 0.8) -> List[str]:
    """Ghi chú kèm câu trả lời: văn bản đã bị thay thế / bãi bỏ, văn bản đọc kém."""
    if not doc_ids or not hasattr(store, "doc_fields"):
        return []
    ids = list(dict.fromkeys(doc_ids))
    fields = store.doc_fields(ids, ["trang_thai_hieu_luc", "thay_the_boi", "ty_le_doc_duoc", "trang_can_scan_lai"])
    docs = store.get_documents(ids)
    notes: List[str] = []
    for doc_id in ids:
        f, doc = fields.get(doc_id, {}), docs.get(doc_id)
        name = (doc.so_hieu or doc.source_file) if doc else doc_id
        status = f.get("trang_thai_hieu_luc")
        if status and status.gia_tri in ("bi_thay_the", "bai_bo"):
            by = f.get("thay_the_boi")
            verb = "đã bị bãi bỏ" if status.gia_tri == "bai_bo" else "đã bị thay thế"
            notes.append(f"Lưu ý: văn bản {name} {verb}" + (f" bởi {by.gia_tri.replace(';', ', ')}" if by and by.gia_tri else "")
                         + "; kiểm tra văn bản đang có hiệu lực trước khi áp dụng.")
        read = f.get("ty_le_doc_duoc")
        try:
            ratio = float(read.gia_tri) if read else 1.0
        except ValueError:
            ratio = 1.0
        if ratio < low_read_ratio:
            pages = f.get("trang_can_scan_lai")
            notes.append(f"Văn bản {name} đọc kém (khoảng {ratio:.0%} số trang đọc được"
                         + (f"; trang cần scan lại: {pages.gia_tri}" if pages and pages.gia_tri else "")
                         + "), nên xem bản gốc.")
    return notes


# ====================================================================== bước AI (mặc định tắt)
AI_SYSTEM = """Bạn hỗ trợ phân loại văn bản nội bộ Công ty TNHH MTV Đường sắt Hà Nội. Chỉ dựa vào nội dung được cung cấp.
Trả về DUY NHẤT một đối tượng JSON, không thêm chữ nào khác, theo lược đồ:
{"phong_chu_tri": [{"ma": "<mã đơn vị>", "ly_do": "<ngắn, dẫn chứng từ văn bản>"}],
 "phong_phoi_hop": [{"ma": "<mã đơn vị>", "ly_do": "..."}],
 "phong_can_xem": [{"ma": "<mã đơn vị>", "ly_do": "..."}],
 "chung": true|false,
 "nhiem_vu": [{"phong": "<mã đơn vị hoặc rỗng>", "nguoi": "<người được giao hoặc rỗng>", "noi_dung": "<việc phải làm>",
               "han": "<yyyy-mm-dd hoặc rỗng>", "loai_han": "ngay|dinh_ky|khong_co", "trang": <số trang>}]}
Định nghĩa:
- "phong_chu_tri": đơn vị trong Công ty tham mưu, soạn thảo, hoặc chịu trách nhiệm chính thực hiện / theo dõi nội dung
  văn bản, đối chiếu với chức năng nhiệm vụ trong danh mục. KHÔNG phải người ký hay cấp phê duyệt: quyết định do
  Chủ tịch HĐTV / Tổng Giám đốc ký thì chủ trì là phòng tham mưu nội dung đó; chỉ chọn BDH khi nội dung thuộc chính
  công việc của Hội đồng thành viên / Ban điều hành (nghị quyết HĐTV, phân công lãnh đạo…).
- Văn bản của cơ quan ngoài gửi đến (thư mời họp, mời tập huấn, đôn đốc, xin ý kiến): chủ trì là đơn vị có chức năng
  phù hợp nhất với nội dung để tham mưu xử lý hoặc cử người tham dự, không phải BDH.
- "phong_phoi_hop": đơn vị được văn bản giao phối hợp hoặc có ý kiến thẩm định, góp ý được nêu trong văn bản.
- Tài liệu kỹ thuật, quy trình vận hành: chủ trì là đơn vị quản lý chuyên môn đó (vận tải – phương tiện, hạ tầng,
  an toàn) hoặc xí nghiệp vận hành đúng tuyến được nêu trong nội dung.
- Chỉ MỘT phòng chủ trì; các đơn vị khác cùng tham gia đưa vào "phong_phoi_hop".
- Văn bản gửi đến Công ty, do Công ty ban hành, hoặc nói về hoạt động, tuyến, tài sản, nhân sự của Công ty thì PHẢI có
  đúng một phòng chủ trì (chọn đơn vị gần nhất với nội dung). Chỉ để rỗng khi văn bản hoàn toàn không liên quan
  đến công việc của Công ty. Văn bản mà dòng "Hướng" ghi là văn bản ĐẾN / ĐI đều nằm trong sổ công văn của Công ty
  → luôn chọn đơn vị xử lý hoặc theo dõi (dự án đường sắt đô thị mới, chủ trương đầu tư → KHDA; vốn, tài chính, xếp loại
  doanh nghiệp → TCKT; nhân sự, cán bộ, thi đua, kiểm soát viên → TCNS; an toàn, trật tự giao thông đường sắt → ATCL).
- Tài liệu bảo dưỡng đoàn tàu (MAINTENANCE TASK ANALYSIS, mã CIDM…, HANOÏ L3, toa xe, cửa tàu, cabin, pin, biến áp
  trên tàu) thuộc phương tiện → VTQLPT, kể cả khi trang rất ít chữ; hệ thống tín hiệu, cấp điện, nhà ga, đường ray,
  thang cuốn thuộc hạ tầng → QLCSHT.
- Hậu tố -1A, -2A, -3A, -4A, -5A, -6A… ở cuối mã tài liệu (vd PIC-OMP-PRC-ZMO-L00-00134-E-2A, UJV-…-V-3A-S200) là SỐ
  PHIÊN BẢN, KHÔNG phải tuyến 2A. Tài liệu dự án tuyến 3 Nhổn – ga Hà Nội có mã PIC-, UJV-, VC2-, PSC-, HPLML, HMC-L3,
  CIDM, ghi "HANOÏ L3", "Hanoi Pilot Light Metro", "Nhổn" → XN3. Tài liệu tuyến 2A Cát Linh – Hà Đông có mã TXWJ-,
  JXGC- hoặc của "Công ty HH Tập đoàn Cục 6 ĐS Trung Quốc" → XN2. Mã có "-RSK-" là đoàn tàu → VTQLPT.
- Tuyển dụng, tiêu chuẩn chức danh, đào tạo, sát hạch nhân sự → có TCNS (cùng phòng chuyên môn liên quan).
  Giấy phép môi trường, quan trắc, xử lý nước thải, chất thải → ATCL và QLCSHT.
- "phong_can_xem": TẤT CẢ đơn vị cần đọc văn bản để làm việc (chủ trì, phối hợp, đơn vị có nghiệp vụ liên quan trực
  tiếp, xí nghiệp của tuyến được nêu). Văn bản có thể thuộc nhiều đơn vị; không thêm đơn vị chỉ vì "nên biết".
  Văn bản có phòng chủ trì thì "phong_can_xem" phải gồm phòng đó.
- "chung": true CHỈ khi mọi nhân viên Công ty đều cần biết (nội quy, quy chế chung, thông báo nghỉ lễ, lịch làm việc
  toàn Công ty, quy định áp dụng cho mọi người lao động). Văn bản nghiệp vụ của một vài phòng → false.
- "nhiem_vu": việc văn bản GIAO hoặc YÊU CẦU Công ty / một đơn vị, một người của Công ty phải làm (thường có "giao",
  "đề nghị", "yêu cầu", "chịu trách nhiệm", "báo cáo… trước ngày…"). KHÔNG đưa: các bước thao tác của quy trình kỹ thuật,
  hướng dẫn bảo dưỡng, SOP, sổ tay sử dụng; việc của cơ quan khác (bên mời họp chuẩn bị phòng họp, Sở tổng hợp báo cáo…).
  Tài liệu kỹ thuật, quy trình thường không có nhiệm vụ → mảng rỗng. Không đưa câu thi hành chung ("… chịu trách
  nhiệm thi hành Quyết định này", "thực hiện Quyết định này"); văn bản góp ý, xin ý kiến gửi cơ quan khác không giao
  việc cho Công ty thì không có nhiệm vụ.
Quy tắc: chỉ dùng mã đơn vị trong danh mục; không đủ căn cứ trong nội dung thì để mảng rỗng; không suy đoán hạn khi văn
bản không ghi; "ly_do" phải dẫn đúng chi tiết có trong văn bản; "trang" là số trong dòng "--- Trang N ---" nơi có nhiệm vụ."""


_HUONG_TEXT = {"den": "văn bản ĐẾN (Công ty là nơi nhận, lưu trong sổ công văn đến)",
               "di": "văn bản ĐI (do Công ty ban hành)", "noi_bo": "văn bản nội bộ Công ty"}


def build_ai_input(doc: DocRecord, pages: Dict[int, str], departments: Dict[str, object], max_chars: int = 6000,
                   huong: str = "") -> str:
    menu = "\n".join(f"- {code}: {getattr(d, 'name', '')} – {getattr(d, 'mo_ta', '')}" for code, d in departments.items())
    keys = sorted(pages)
    chosen = keys[:1] + ([keys[-1]] if len(keys) > 1 else [])
    body = "\n\n".join(f"--- Trang {p} ---\n{pages[p][:max_chars]}" for p in chosen)
    meta = f"Số hiệu: {doc.so_hieu}\nNgày: {doc.ngay_ban_hanh}\nLoại: {doc.loai_vb}\nTrích yếu: {doc.trich_yeu or doc.title}"
    if huong in _HUONG_TEXT:
        meta += f"\nHướng: {_HUONG_TEXT[huong]}"
    return f"DANH MỤC ĐƠN VỊ:\n{menu}\n\nTHÔNG TIN VĂN BẢN:\n{meta}\n\nNỘI DUNG (trang đầu và trang cuối):\n{body}"


@dataclass
class AIResult:
    chu_tri: List[Tuple[str, str]] = field(default_factory=list)
    phoi_hop: List[Tuple[str, str]] = field(default_factory=list)
    tasks: List[DocTask] = field(default_factory=list)
    can_xem: List[Tuple[str, str]] = field(default_factory=list)     # mọi đơn vị cần xem (gồm chủ trì, phối hợp)
    chung: bool = False                                               # mọi nhân viên cần biết

    def view_codes(self) -> List[str]:
        """Nhãn xem đề xuất: phòng cần xem + chủ trì + phối hợp (+ CHUNG); chỉ là GỢI Ý, người duyệt quyết định."""
        out: List[str] = []
        for c, _ in self.chu_tri + self.phoi_hop + self.can_xem:
            if c not in out:
                out.append(c)
        return out + (["CHUNG"] if self.chung else [])


_RX_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_ai_output(text: str, departments: Sequence[str], pages: Sequence[int]) -> AIResult:
    """Đọc JSON của mô hình; bỏ mọi mã đơn vị ngoài danh mục, trang ngoài văn bản, hạn sai định dạng."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    known, valid_pages = set(departments), set(pages)

    def units(key: str) -> List[Tuple[str, str]]:
        out = []
        for item in data.get(key) or []:
            if isinstance(item, dict):
                code = str(item.get("ma", "")).strip().upper()
                if code in known and code not in {c for c, _ in out}:
                    out.append((code, " ".join(str(item.get("ly_do", "")).replace("|", "/").split())[:200]))
        return out

    tasks = []
    for t in data.get("nhiem_vu") or []:
        if not isinstance(t, dict) or not str(t.get("noi_dung", "")).strip():
            continue
        phong = str(t.get("phong", "")).strip().upper()
        han = str(t.get("han", "") or "").strip()
        loai = str(t.get("loai_han", "") or "").strip()
        try:
            page = int(t.get("trang"))
        except (TypeError, ValueError):
            page = None
        tasks.append(DocTask(phong if phong in known else "", " ".join(str(t.get("nguoi", "")).split())[:120],
                             " ".join(str(t.get("noi_dung", "")).split())[:500], han if _RX_ISO.match(han) else "",
                             loai if loai in ("ngay", "dinh_ky", "khong_co") else "", page if page in valid_pages else None,
                             NGUON_AI))
    chu_tri, phoi_hop = units("phong_chu_tri"), units("phong_phoi_hop")
    if len(chu_tri) > 1:     # chỉ một phòng chủ trì: giữ phòng đầu, các phòng sau chuyển sang phối hợp
        extra = [(c, why) for c, why in chu_tri[1:] if c not in {x for x, _ in phoi_hop}]
        chu_tri, phoi_hop = chu_tri[:1], extra + phoi_hop
    chung = data.get("chung") is True or str(data.get("chung", "")).strip().lower() == "true"
    return AIResult(chu_tri, phoi_hop, tasks, units("phong_can_xem"), chung)


class AIBudgetExceeded(RuntimeError):
    pass


class AILedger:
    """Sổ các lượt gọi AI: mỗi văn bản một lần cho mỗi phiên bản; tổng chi là căn cứ hạn mức ENRICH_AI_BUDGET_USD."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS enrich_ai (doc_id TEXT, version TEXT, in_tokens INTEGER, "
                      "out_tokens INTEGER, cost_usd REAL, created_at TEXT, PRIMARY KEY (doc_id, version))")
            if "raw" not in {r[1] for r in c.execute("PRAGMA table_info(enrich_ai)")}:
                c.execute("ALTER TABLE enrich_ai ADD COLUMN raw TEXT")       # phản hồi gốc, để tra soát

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def spent_usd(self) -> float:
        with self._conn() as c:
            return float(c.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM enrich_ai").fetchone()[0])

    def done(self, version: str) -> set:
        with self._conn() as c:
            return {r[0] for r in c.execute("SELECT doc_id FROM enrich_ai WHERE version = ?", (version,))}

    def record(self, doc_id: str, version: str, tin: int, tout: int, cost: float, raw: str = "") -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO enrich_ai (doc_id, version, in_tokens, out_tokens, cost_usd, created_at, raw) "
                      "VALUES (?,?,?,?,?,?,?)",
                      (doc_id, version, tin, tout, cost, datetime.now().isoformat(timespec="seconds"), raw))


@dataclass
class AIConfig:
    budget_usd: float
    price_in_per_m: float
    price_out_per_m: float
    max_out_tokens: int = 3000           # gồm cả token suy luận


def save_ai_result(store, doc_id: str, res: AIResult, page1: int, version: str) -> None:
    store.set_doc_fields(doc_id, [
        DocField("phong_chu_tri", ";".join(c for c, _ in res.chu_tri), NGUON_AI, page1, "", version,
                 " | ".join(f"{c}: {why}" for c, why in res.chu_tri)),
        DocField("phong_phoi_hop", ";".join(c for c, _ in res.phoi_hop), NGUON_AI, page1, "", version,
                 " | ".join(f"{c}: {why}" for c, why in res.phoi_hop)),
        DocField("phong_can_xem", ";".join(res.view_codes()), NGUON_AI, page1, "", version,
                 " | ".join(f"{c}: {why}" for c, why in res.can_xem)),
    ])
    store.set_doc_tasks(doc_id, res.tasks, NGUON_AI)


def reparse_ai(store, ledger: AILedger, departments: Dict[str, object],
               versions: Sequence[str] = (AI_VERSION, *AI_PREVIOUS)) -> Dict[str, AIResult]:
    """
    Áp lại bộ đọc kết quả (parse_ai_output) lên phản hồi gốc đã lưu: không gọi API. Dùng khi đổi cách chuẩn hóa
    kết quả. Mỗi văn bản lấy phản hồi của phiên bản mới nhất trong `versions`.
    """
    order = {v: i for i, v in enumerate(versions)}
    latest: Dict[str, Tuple[int, str, str]] = {}
    with ledger._conn() as c:
        rows = c.execute(f"SELECT doc_id, version, raw FROM enrich_ai WHERE version IN ({','.join('?' * len(versions))})",
                         list(versions)).fetchall()
    for doc_id, version, raw in rows:
        if doc_id not in latest or order[version] < latest[doc_id][0]:
            latest[doc_id] = (order[version], version, raw or "")
    docs = store.get_documents(list(latest))
    results: Dict[str, AIResult] = {}
    for doc_id, (_, version, raw) in latest.items():
        doc = docs.get(doc_id)
        if doc is None:
            continue
        pages = list(document_pages(store, doc_id)) or [1]
        res = parse_ai_output(raw, list(departments), pages)
        save_ai_result(store, doc_id, res, min(pages), version)
        results[doc_id] = res
    return results


def run_ai(store, docs: Sequence[DocRecord], llm, outbound_guard: Callable[[Sequence[str]], None],
           departments: Dict[str, object], ledger: AILedger, cfg: AIConfig,
           version: str = AI_VERSION) -> Tuple[EnrichStats, Dict[str, AIResult]]:
    """
    Một lượt gọi LLM mỗi văn bản. Trước MỖI lần gửi: kiểm tra danh sách chặn tài liệu mật (thiếu danh sách chặn → dừng)
    và hạn mức chi phí (ước tính trước khi gửi, ghi lại sau khi nhận).
    """
    from src.indexing.indexer import BlockedDocumentError, BlocklistNotConfigured

    stats, results = EnrichStats(), {}
    done = ledger.done(version)
    if version == AI_VERSION:
        for old in AI_COMPATIBLE:
            done |= ledger.done(old)
    for doc in docs:
        if doc.doc_id in done:
            stats.counts["ai_da_co"] += 1
            continue
        pages = document_pages(store, doc.doc_id)
        if not pages:
            stats.counts["ai_khong_noi_dung"] += 1
            continue
        hv = store.doc_fields([doc.doc_id], ("huong_vb",)).get(doc.doc_id, {}).get("huong_vb") \
            if hasattr(store, "doc_fields") else None
        user = build_ai_input(doc, pages, departments, huong=hv.gia_tri if hv else "")
        # tiếng Việt ~2,4 ký tự/token với bộ tách từ của API (đo thực tế); lấy mức cao để ước tính không bị thấp
        tin = max(approx_tokens(AI_SYSTEM + user), len(AI_SYSTEM + user) // 2)
        estimate = (tin * cfg.price_in_per_m + cfg.max_out_tokens * cfg.price_out_per_m) / 1e6
        spent = ledger.spent_usd()
        if spent + estimate > cfg.budget_usd:
            raise AIBudgetExceeded(f"Dừng bước AI: đã chi {spent:.4f} USD, gửi tiếp có thể vượt ENRICH_AI_BUDGET_USD="
                                   f"{cfg.budget_usd} USD. Đã xử lý {stats.counts['ai_xong']} văn bản trong lần chạy này.")
        try:
            outbound_guard([doc.doc_id])          # kiểm tra lại ngay trước khi gửi
        except BlocklistNotConfigured:
            raise
        except BlockedDocumentError:
            stats.counts["ai_bi_chan"] += 1
            continue
        reply = llm.complete(AI_SYSTEM, user, max_tokens=cfg.max_out_tokens)
        usage = getattr(llm, "last_usage", None)
        tin, tout = usage if usage else (tin, approx_tokens(reply))    # có số liệu thật từ API thì dùng
        ledger.record(doc.doc_id, version, tin, tout, (tin * cfg.price_in_per_m + tout * cfg.price_out_per_m) / 1e6, reply)
        if not reply.strip():
            stats.counts["ai_tra_rong"] += 1
        res = parse_ai_output(reply, list(departments), list(pages))
        results[doc.doc_id] = res
        save_ai_result(store, doc.doc_id, res, min(pages), version)
        stats.counts["ai_xong"] += 1
        stats.counts["nhiem_vu"] += len(res.tasks)
    return stats, results


# ====================================================================== phiếu kiểm tra tay
REVIEW_COLUMNS = ["doc_id", "so_hieu", "ngay", "trich_yeu", "source_file",
                  "huong_vb", "huong_vb_ly_do", "dung_huong_vb",
                  "trang_thai_hieu_luc", "thay_the_boi", "quan_he_thay_the_bai_bo", "dung_hieu_luc",
                  "ai_phong_chu_tri", "ai_phong_phoi_hop", "ai_ly_do", "dung_phong_chu_tri", "phong_dung",
                  "ai_nhiem_vu", "dung_nhiem_vu", "ghi_chu"]


def review_started(path: Path) -> bool:
    """Phiếu đã có người điền cột dung_* / phong_dung / ghi_chu chưa."""
    path = Path(path)
    if not path.is_file():
        return False
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if any((row.get(k) or "").strip() for k in ("dung_huong_vb", "dung_hieu_luc", "dung_phong_chu_tri",
                                                        "phong_dung", "dung_nhiem_vu", "ghi_chu")):
                return True
    return False


def export_review(store, path: Path) -> int:
    """
    Phiếu kiểm tra độ chính xác: mỗi văn bản đã qua bước AI một dòng, kèm kết quả luật để kiểm cùng lúc.
    Người kiểm điền các cột dung_* (Đ / S / ?), phong_dung (mã đúng nếu AI sai), ghi_chu.
    """
    docs, _ = store.list_documents(Filters(), limit=1_000_000)
    ids = [d.doc_id for d in docs]
    fields = store.doc_fields(ids)
    chosen = [d for d in docs if fields[d.doc_id].get("phong_chu_tri") and fields[d.doc_id]["phong_chu_tri"].nguon == NGUON_AI]
    rel = {}
    for r in store.doc_relations([d.doc_id for d in chosen]):
        if r.loai in ("thay_the", "bai_bo"):
            rel.setdefault(r.doc_id, []).append(f"{r.loai} {r.so_hieu_dich} (tr.{r.trang})")
    tasks = {}
    for t in store.doc_tasks([d.doc_id for d in chosen]):
        tasks.setdefault(t.doc_id, []).append(
            f"[{t.phong or '?'}] {t.nguoi + ': ' if t.nguoi else ''}{t.noi_dung}"
            + (f" – hạn {t.han}" if t.han else "") + (f" (tr.{t.trang})" if t.trang else ""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if review_started(path):
        raise FileExistsError(f"{path} đã có kết quả kiểm tra tay; không ghi đè. Ghi ra file khác bằng --out.")
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for d in chosen:
            f = fields[d.doc_id]
            get = lambda k, attr="gia_tri": getattr(f[k], attr) if k in f else ""
            w.writerow({
                "doc_id": d.doc_id, "so_hieu": d.so_hieu, "ngay": d.ngay_ban_hanh,
                "trich_yeu": (d.trich_yeu or d.title)[:200], "source_file": d.source_file,
                "huong_vb": get("huong_vb"), "huong_vb_ly_do": get("huong_vb", "ghi_chu"),
                "trang_thai_hieu_luc": get("trang_thai_hieu_luc"), "thay_the_boi": get("thay_the_boi"),
                "quan_he_thay_the_bai_bo": " | ".join(rel.get(d.doc_id, [])),
                "ai_phong_chu_tri": get("phong_chu_tri"), "ai_phong_phoi_hop": get("phong_phoi_hop"),
                "ai_ly_do": " | ".join(x for x in (get("phong_chu_tri", "ghi_chu"), get("phong_phoi_hop", "ghi_chu")) if x),
                "ai_nhiem_vu": "\n".join(tasks.get(d.doc_id, [])),
            })
    return len(chosen)


# ====================================================================== thống kê, CLI
def summarize(store) -> Dict[str, Counter]:
    docs, _ = store.list_documents(Filters(), limit=1_000_000)
    ids = [d.doc_id for d in docs]
    fields = store.doc_fields(ids)
    out: Dict[str, Counter] = {"tong": Counter({"van_ban": len(ids)})}
    for name in ("huong_vb", "trang_thai_hieu_luc"):
        out[name] = Counter(f[name].gia_tri for f in fields.values() if name in f)
        out[name]["(chưa chạy)"] = sum(name not in f for f in fields.values())
    out["quan_he"] = Counter(r.loai for r in store.doc_relations(ids))
    out["quan_he"]["co_doc_id_dich"] = sum(bool(r.doc_id_dich) for r in store.doc_relations(ids))
    low = Counter()
    for f in fields.values():
        if "ty_le_doc_duoc" in f:
            low["doc_kem (<0.8)" if float(f["ty_le_doc_duoc"].gia_tri or 1) < 0.8 else "doc_tot"] += 1
        if "phong_chu_tri" in f and f["phong_chu_tri"].gia_tri:
            low["ai_goi_y_phong"] += 1
    out["chat_luong_va_ai"] = low
    out["nhiem_vu_ai"] = Counter({"so_nhiem_vu": len(store.doc_tasks(ids))})
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Bổ sung metadata văn bản (luật; AI khi có --ai)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("chay")
    s.add_argument("--ai", action="store_true", help="Bật bước AI (gọi LLM, tốn phí; mặc định tắt)")
    s.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N văn bản đầu (chạy thử)")
    s.add_argument("--doc", default="", help="Chỉ một doc_id")
    s.add_argument("--doc-file", type=Path, default=None, help="Chỉ các doc_id trong file (JSON list hoặc mỗi dòng một id)")
    s.add_argument("--ngau-nhien", action="store_true", help="Chọn --limit văn bản ngẫu nhiên (cố định hạt giống) thay vì N văn bản đầu")
    s.add_argument("--hat-giong", type=int, default=2026)
    s.add_argument("--chi-ai", action="store_true", help="Bỏ qua bước luật (đã chạy), chỉ chạy bước AI")
    s.add_argument("--chi-trang-thai", action="store_true", help="Chỉ tính lại trạng thái hiệu lực từ quan hệ đã có (nhanh)")
    s.add_argument("--results", type=Path, default=None, help="Thư mục pages_routed.csv (mặc định CORPUS_RESULTS_DIR)")
    s.add_argument("--labels-file", type=Path, default=None, help="File duyệt nhãn để ghi gợi ý AI (cột phong_goi_y)")
    sub.add_parser("thong-ke")
    s = sub.add_parser("xuat-kiem-tra")
    s.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("tinh-lai-ai", help="Áp lại bộ đọc kết quả lên phản hồi AI đã lưu (không gọi API)")
    s.add_argument("--labels-file", type=Path, default=None)
    args = ap.parse_args(argv)

    from src.config import settings
    from src.runtime import build_runtime
    rt = build_runtime()
    if args.cmd == "xuat-kiem-tra":
        out = args.out or settings.project_root / "data/access/kiem_tra_enrich.csv"
        try:
            print(f"Đã ghi {export_review(rt.store, out)} văn bản vào {out}. Điền các cột dung_* (Đ / S / ?), phong_dung, ghi_chu.")
        except FileExistsError as e:
            print(e)
            return 1
        return 0
    if args.cmd == "tinh-lai-ai":
        from src.access.labels import merge_ai_suggestions
        from src.access.policy import load_policy
        cfg = settings.agent
        cfg_dir = Path(cfg.access_config_dir)
        policy = load_policy(cfg_dir if cfg_dir.is_absolute() else settings.project_root / cfg_dir, enabled=False)
        ledger_path = Path(cfg.enrich_ai_ledger)
        ai_ledger = AILedger(ledger_path if ledger_path.is_absolute() else settings.project_root / ledger_path)
        results = reparse_ai(rt.store, ai_ledger, policy.departments)
        print(f"Đã tính lại {len(results)} văn bản từ phản hồi gốc (không gọi API).")
        labels_file = args.labels_file or settings.project_root / "data/access/nhan_van_ban.csv"
        merged = merge_ai_suggestions(labels_file, results)
        print(f"Cập nhật cột phong_goi_y trong {labels_file}: {merged} dòng." if merged is not None else f"Chưa có {labels_file}.")
        review = settings.project_root / "data/access/kiem_tra_enrich.csv"
        try:
            print(f"Phiếu kiểm tra tay: {export_review(rt.store, review)} văn bản → {review}")
        except FileExistsError as e:
            print(e)
        return 0
    if args.cmd == "thong-ke":
        for group, counter in summarize(rt.store).items():
            print(f"{group}: " + ", ".join(f"{k}={v}" for k, v in counter.most_common()))
        return 0

    docs, _ = rt.store.list_documents(Filters(), limit=1_000_000)
    all_docs = docs
    if args.doc:
        docs = [d for d in docs if d.doc_id == args.doc.lower()]
    if args.doc_file:
        import json as _json
        raw = args.doc_file.read_text(encoding="utf-8-sig").strip()
        wanted = set(_json.loads(raw)) if raw.startswith("[") else {x.strip().lower() for x in raw.splitlines() if x.strip()}
        docs = [d for d in docs if d.doc_id in wanted]
    if args.limit and args.ngau_nhien:
        import random
        with_text = [d for d in docs if d.total_pages > 0]
        docs = random.Random(args.hat_giong).sample(with_text, min(args.limit, len(with_text)))
    elif args.limit:
        docs = docs[:args.limit]
    cfg = settings.agent
    results_dir = args.results or Path(cfg.enrich_results_dir)
    routing = RoutingTable.load(results_dir)
    ledger = LedgerQuality.load(settings.ingestion.state_db_path.with_name("vlm_ledger.db"), settings.vlm.max_tokens)
    vector = rt.vector
    if args.chi_trang_thai:
        print("Trạng thái hiệu lực:", dict(update_status(rt.store, all_docs, vector)))
        return 0
    if not args.chi_ai:
        print(f"Luật ({RULES_VERSION}) cho {len(docs)} văn bản…")
        print("Kết quả:", run_rules(rt.store, docs, routing, ledger, vector, all_docs=all_docs))
    if not args.ai:
        return 0

    from src.access.labels import merge_ai_suggestions
    from src.access.policy import load_policy
    import os
    from src.agent.llm import OpenAIChatModel, create_chat_model
    cfg_dir = Path(cfg.access_config_dir)
    policy = load_policy(cfg_dir if cfg_dir.is_absolute() else settings.project_root / cfg_dir, enabled=False)
    ledger_path = Path(cfg.enrich_ai_ledger)
    ai_ledger = AILedger(ledger_path if ledger_path.is_absolute() else settings.project_root / ledger_path)
    ai_cfg = AIConfig(cfg.enrich_ai_budget_usd, cfg.enrich_price_in_per_m, cfg.enrich_price_out_per_m)
    print(f"Bước AI ({AI_VERSION}) cho tối đa {len(docs)} văn bản; đã chi {ai_ledger.spent_usd():.4f} / "
          f"{ai_cfg.budget_usd} USD (ước tính theo số token).")
    try:
        llm, version = create_chat_model(), AI_VERSION
        if isinstance(llm, OpenAIChatModel):     # bước phân loại cần suy luận; gateway hỏi đáp vẫn dùng LLM_REASONING_EFFORT
            effort = os.getenv("ENRICH_REASONING_EFFORT", "low")
            llm = OpenAIChatModel(settings.llm.temperature, settings.llm.max_tokens, effort)
            version = AI_VERSION if effort == "low" else f"{AI_VERSION}-{effort}"   # mỗi mức suy luận một phiên bản
        print(f"Phiên bản: {version}")
        stats, results = run_ai(rt.store, docs, llm, rt.blocklist.assert_allowed,
                                policy.departments, ai_ledger, ai_cfg, version)
    except AIBudgetExceeded as e:
        print(e)
        return 3
    print("Kết quả AI:", stats)
    labels_file = args.labels_file or settings.project_root / "data/access/nhan_van_ban.csv"
    merged = merge_ai_suggestions(labels_file, {d: r for d, r in results.items()})
    review = settings.project_root / "data/access/kiem_tra_enrich.csv"
    try:
        print(f"Phiếu kiểm tra tay: {export_review(rt.store, review)} văn bản → {review}")
    except FileExistsError as e:
        print(e, "Dùng: python -m src.indexing.enrich xuat-kiem-tra --out <file mới>")
    print(f"Gợi ý phòng của AI đã ghi vào {labels_file} (cột phong_goi_y, chưa áp dụng): {merged} dòng. "
          "Người duyệt điền cột phong_duyet rồi chạy python -m src.access.labels ap-dung." if merged is not None else
          f"Chưa có {labels_file}: chạy python -m src.access.labels goi-y (gợi ý AI đã lưu, sẽ có trong file).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
