"""
Hanoi Metro AI Agent - Chunking Profile Classifier (Mục 4.3 đề xuất).
Chấm điểm theo luật để gán văn bản vào 1 trong 4 hồ sơ chunking, kèm lý do để rà soát:
  H1 – Văn bản hành chính, H2 – Văn bản quy phạm, H3 – Tài liệu kỹ thuật dài, H4 – Bảng biểu.
Không cần dữ liệu gán nhãn, mỗi quyết định có lý do rõ ràng, chạy vài mili-giây mỗi văn bản.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.indexing.structure import AdminInfo, Block
from src.text.normalize import fold

PROFILES = ("H1", "H2", "H3", "H4")

_RX_ADMIN_FILENAME = re.compile(r"(?<![A-Za-z])(CV|QĐ|QD|TB|TTr|TT|GM|BC|KH|BB|HĐ|HD)(?![a-z])")
_ADMIN_FOLDERS = ("cv di", "cv den", "cong van", "giay moi", "van ban scan", "van thu")
_TECH_COVER = ("manual", "so tay", "quy trinh", "sop", "revision", "ma so ho so", "huong dan van hanh", "operation", "maintenance")
_TECH_FOLDERS = ("cmms", "quy trinh", "van hanh", "bao tri")
_NORMATIVE_TITLES = ("quy che", "quy dinh", "dieu le", "noi quy")
_FORM_MARKERS = ("maintenance task analysis", "estimated time", "phieu cong viec")


@dataclass
class ProfileDecision:
    profile: str
    confidence: str                  # "high" | "medium" | "low"
    scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class Segment:
    """Một phần của văn bản lai (Quyết định kèm Quy chế, Tờ trình kèm Phụ lục bảng)."""
    profile: str
    blocks: List[Block]


def classify_profile(path: str, total_pages: int, admin: AdminInfo, blocks: List[Block], toc: List[str]) -> ProfileDecision:
    s = {p: 0.0 for p in PROFILES}
    why: List[str] = []
    folded_path = fold(path.replace("\\", "/"))
    filename = path.replace("\\", "/").rsplit("/", 1)[-1]
    first_text = fold(" ".join(b.text for b in blocks[:8]))[:1500]

    marks = len(set(admin.markers))
    if marks >= 3:
        s["H1"] += 3
        why.append(f"thể thức hành chính {marks}/5 (+3 H1)")
    elif marks == 2:
        s["H1"] += 1.5
        why.append("thể thức hành chính 2/5 (+1,5 H1)")
    if _RX_ADMIN_FILENAME.search(filename) or re.search(r"ngay\s*\d{6,8}", fold(filename)):
        s["H1"] += 1
        why.append("tên file có ký hiệu loại văn bản (+1 H1)")
    if any(k in folded_path for k in _ADMIN_FOLDERS):
        s["H1"] += 1
        why.append("thư mục văn thư (+1 H1)")
    if 0 < total_pages <= 10:
        s["H1"] += 0.5

    n_dieu = sum(1 for b in blocks if b.kind == "heading" and b.legal == "dieu")
    if n_dieu >= 3:
        s["H2"] += 3
        why.append(f"{n_dieu} Điều (+3 H2)")
    if any(b.legal == "chuong" for b in blocks):
        s["H2"] += 1
        why.append("có Chương (+1 H2)")
    if any(fold(b.text).startswith(_NORMATIVE_TITLES) for b in blocks[:15] if len(b.text) < 120):
        s["H2"] += 1
        why.append("tiêu đề quy phạm (+1 H2)")

    if total_pages > 10:
        s["H3"] += 2
        why.append(f"{total_pages} trang (+2 H3)")
    if toc:
        s["H3"] += 1
        why.append("có mục lục (+1 H3)")
    if any(b.kind == "heading" and re.match(r"^\d+\.\d+\.\d+", b.text) for b in blocks):
        s["H3"] += 1
        why.append("đề mục 3 cấp (+1 H3)")
    tech_hits = sum(1 for k in _TECH_COVER if k in first_text) + (1 if re.search(r"[A-Z]{2,}-[A-Z]{2,}-", filename) else 0)
    tech_hits += sum(1 for k in _TECH_FOLDERS if k in folded_path)
    if tech_hits:
        s["H3"] += tech_hits
        why.append(f"dấu hiệu tài liệu kỹ thuật x{tech_hits} (+{tech_hits} H3)")

    total_chars = sum(len(b.text) for b in blocks) or 1
    table_ratio = sum(len(b.text) for b in blocks if b.kind == "table") / total_chars
    if table_ratio >= 0.5:
        s["H4"] += 3
        why.append(f"bảng chiếm {table_ratio:.0%} (+3 H4)")
    full_text = fold(" ".join(b.text for b in blocks[:40]))
    if any(k in full_text for k in _FORM_MARKERS):
        s["H4"] += 3
        why.append("mẫu phiếu công việc (+3 H4)")
    if "cidm" in fold(filename) or "dinh muc" in folded_path:
        s["H4"] += 1
        why.append("tên file/thư mục định mức (+1 H4)")

    ranked = sorted(s.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < 2:
        best = "H1" if total_pages <= 5 else "H3"
        why.append(f"không đủ tín hiệu (điểm cao nhất {top}); {total_pages} trang -> {best}")
        return ProfileDecision(best, "low", s, why)
    return ProfileDecision(best, "high" if top - second >= 2 else "medium", s, why)


def segment_document(profile: str, blocks: List[Block]) -> List[Segment]:
    """
    Văn bản lai: Quyết định >= 5 Điều kèm "ban hành kèm theo"/tiêu đề quy chế -> phần quyết định H1,
    phần quy chế H2; phần "PHỤ LỤC" có bảng >= 30% -> H4. Quyết định thường chỉ 3–4 Điều nên không bị tách.
    """
    segments = [Segment(profile, blocks)]
    n_dieu = sum(1 for b in blocks if b.legal == "dieu")
    text = fold(" ".join(b.text for b in blocks))
    if profile in ("H1", "H2") and n_dieu >= 5 and ("ban hanh kem theo" in text):
        for k, b in enumerate(blocks):
            if k > 0 and len(b.text) < 120 and fold(b.text.strip("*# ")).startswith(_NORMATIVE_TITLES) and any(x.legal == "dieu" for x in blocks[:k]):
                segments = [Segment("H1", blocks[:k]), Segment("H2", blocks[k:])]
                break

    out: List[Segment] = []
    for seg in segments:
        cut = next((k for k, b in enumerate(seg.blocks) if b.kind == "heading" and b.legal == "phu_luc"), None)
        if cut is None or cut == 0:
            out.append(seg)
            continue
        tail = seg.blocks[cut:]
        ratio = sum(len(b.text) for b in tail if b.kind == "table") / (sum(len(b.text) for b in tail) or 1)
        if ratio >= 0.3:
            out.extend([Segment(seg.profile, seg.blocks[:cut]), Segment("H4", tail)])
        else:
            out.append(seg)
    return out
