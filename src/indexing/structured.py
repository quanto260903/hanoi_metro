"""
Hanoi Metro AI Agent - Structured Profile Chunker (Mục 4 đề xuất).
Quy trình 5 bước: làm sạch → nhận diện hồ sơ → dựng cây cấu trúc → cắt theo hồ sơ → gắn ngữ cảnh.
- H1: cả phần nội dung là 1 chunk nếu <= 1.000 token; dài hơn cắt theo I./1./a).
- H2: 1 Điều = 1 chunk; Điều dài tách theo Khoản, vẫn giữ tên Điều; không overlap.
- H3: chunk con 250–500 token, mục cha tới ~2.000 token; overlap chỉ khi tách một mục dài.
- Bảng (mọi hồ sơ): bảng <= 600 token = 1 chunk; bảng lớn chia theo hàng, lặp hàng tiêu đề.
Mỗi chunk mang dòng ngữ cảnh [số hiệu | ngày | đơn vị | trích yếu] › đề mục, trong cả vector lẫn chỉ mục từ khóa.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.indexing.document import SourceDocument, load_markdown
from src.indexing.models import Chunk, DocRecord, ParentSection
from src.indexing.profiles import classify_profile, segment_document
from src.indexing.structure import (
    _RX_DATE_VN, Block, admin_from_filename, build_blocks, clean_pages, extract_admin,
    flatten_table, iso_date, split_table,
)
from src.text.codes import extract_codes
from src.text.normalize import approx_tokens

logger = logging.getLogger("hanoi_metro.structured_chunker")

_VI_MARKS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")
_RX_DOC_CODE = re.compile(r"[A-Z]{2,}(?:-[A-Z0-9]{2,}){2,}")
_RX_FULL_DOC_CODE = re.compile(r"([A-Z]{2,5}(?:-[A-Z0-9]{1,6}){5,})(?:\.\d{1,3})?$")


@dataclass
class ChunkingParams:
    h1_single_max: int = 1000
    child_min: int = 250
    child_max: int = 500
    parent_max: int = 2000
    overlap: int = 60
    h2_article_max: int = 800
    table_single_max: int = 600
    table_rows: int = 20


@dataclass
class _Section:
    path: List[str]
    heading: str
    blocks: List[Block] = field(default_factory=list)
    heading_page: int = 1

    @property
    def text(self) -> str:
        body = "\n\n".join(b.text for b in self.blocks)
        return f"{self.heading}\n{body}".strip() if self.heading else body

    @property
    def pages(self) -> Tuple[int, int]:
        if not self.blocks:
            return (0, 0)
        return (min(b.page_from for b in self.blocks), max(b.page_to for b in self.blocks))


def _join_path(path: List[str]) -> str:
    return " › ".join(path)


def _common_prefix(paths: List[List[str]]) -> List[str]:
    out: List[str] = []
    for parts in zip(*paths):
        if len(set(parts)) != 1:
            break
        out.append(parts[0])
    return out


def _detect_lang(text: str) -> str:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return "vi"
    return "vi" if sum(c in _VI_MARKS for c in letters) / len(letters) > 0.015 else "en"


def split_long_text(text: str, max_tokens: int, overlap_tokens: int, unit_pattern: str = r"\n\s*\n") -> List[str]:
    """Tách đoạn dài theo đoạn văn (rồi theo câu), đóng gói tới max_tokens, chồng lấn overlap_tokens."""
    units: List[str] = []
    for part in re.split(unit_pattern, text):
        part = part.strip()
        if not part:
            continue
        if approx_tokens(part) <= max_tokens:
            units.append(part)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.;!?])\s+", part) if s.strip())

    chunks: List[str] = []
    cur: List[str] = []
    for unit in units:
        if cur and approx_tokens("\n".join(cur + [unit])) > max_tokens:
            chunks.append("\n".join(cur))
            carry: List[str] = []
            for prev in reversed(cur):
                if approx_tokens("\n".join([prev] + carry)) > overlap_tokens:
                    break
                carry.insert(0, prev)
            cur = carry if overlap_tokens else []
        cur.append(unit)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


class StructuredChunker:
    def __init__(self, params: Optional[ChunkingParams] = None):
        self.p = params or ChunkingParams()

    def chunk_file(self, path: Path) -> Tuple[DocRecord, List[ParentSection], List[Chunk]]:
        return self.chunk_document(load_markdown(path))

    def chunk_document(self, src: SourceDocument) -> Tuple[DocRecord, List[ParentSection], List[Chunk]]:
        lines, toc = clean_pages([(p.number, p.text) for p in src.pages])
        admin, body = extract_admin(lines)
        blocks = build_blocks(body)
        doc = self._build_record(src, admin, blocks)
        decision = classify_profile(src.source_path or src.source_file, src.total_pages, admin, blocks, toc)
        doc.ho_so, doc.ho_so_confidence, doc.ho_so_reasons = decision.profile, decision.confidence, decision.reasons

        parents: List[ParentSection] = []
        chunks: List[Chunk] = []
        for seg in segment_document(decision.profile, blocks):
            self._chunk_segment(doc, seg.profile, seg.blocks, parents, chunks)
        logger.info(f"'{src.source_file}': hồ sơ {doc.ho_so} ({doc.ho_so_confidence}), {len(parents)} mục cha, {len(chunks)} chunk.")
        return doc, parents, chunks

    # --------------------------------------------------------------------------
    def _build_record(self, src: SourceDocument, admin, blocks: List[Block]) -> DocRecord:
        body_text = "\n".join(b.text for b in blocks)
        fn_so_hieu, fn_ngay = admin_from_filename(src.source_file)
        ngay = admin.ngay_ban_hanh
        if not ngay:
            m = _RX_DATE_VN.search(body_text[:600])
            ngay = iso_date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else fn_ngay
        nam = int(ngay[:4]) if ngay else None
        if nam is None:
            years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", src.source_path.replace("\\", "/").rsplit("/", 1)[0])
            nam = int(years[-1]) if years else None
        # ô số hiệu thường để trống rồi đóng số bằng tay sau khi ký ("Số:   /QĐ-SXD"): khi thiếu phần số, lấy từ tên file
        so_hieu = admin.so_hieu if admin.so_hieu[:1].isdigit() or not fn_so_hieu else fn_so_hieu
        if so_hieu.startswith("/"):
            # tên file ghi loại trước số: "TTr 1085 Đường sắt…", "CV số 52 …" → ghép số vào "/TTR-NV" → "1085/TTR-NV"
            kind = so_hieu[1:].split("-")[0]
            m = re.search(rf"(?<![\w]){re.escape(kind)}\.?\s*(?:số\s*)?(\d{{1,5}})(?![\d.])", src.source_file, re.I)                 or re.search(rf"(?<![\d.])(\d{{1,5}})[\s._-]*{re.escape(kind)}(?![\w])", src.source_file, re.I)   # "So 7240.SGTVT"
            if m:
                so_hieu = f"{m.group(1)}{so_hieu}"
        # tài liệu kỹ thuật dự án đặt tên theo mã tài liệu ("PIC-OMP-PRC-ZMO-L00-00171-E-2A", "UJV-OPM-...-S200-3.05"):
        # mã đầy đủ mới là định danh; mảnh "00171/E-2A" mà bộ trích số hiệu bắt được không có nghĩa
        full = _RX_FULL_DOC_CODE.match(re.sub(r"\.(pdf|md|docx?)$", "", src.source_file, flags=re.I))
        if full:
            so_hieu = full.group(1)
        code = _RX_DOC_CODE.search(src.source_file)
        lang = _detect_lang(body_text[:5000])
        pair_id = re.sub(r"[-_ ]?(E|V|EN|VN|VI)$", "", code.group(0)) if code else ""
        cites = [c for c in extract_codes(body_text) if "/" in c and c != so_hieu]
        return DocRecord(
            doc_id=src.doc_id,
            source_file=src.source_file,
            source_path=src.source_path,
            title=src.title,
            total_pages=src.total_pages,
            so_hieu=so_hieu,
            ngay_ban_hanh=ngay,
            nam=nam,
            loai_vb=admin.loai_vb,
            co_quan=admin.co_quan,
            don_vi=admin.don_vi,
            trich_yeu=admin.trich_yeu,
            kinh_gui=admin.kinh_gui,
            nguoi_ky=admin.nguoi_ky,
            noi_nhan=admin.noi_nhan,
            lang=lang,
            pair_id=pair_id,
            cites=cites,
        )

    @staticmethod
    def context_line(doc: DocRecord) -> str:
        if doc.so_hieu or doc.ho_so in ("H1", "H2") or doc.loai_vb:
            ngay = "/".join(reversed(doc.ngay_ban_hanh.split("-"))) if doc.ngay_ban_hanh else ""
            head = f"{doc.loai_vb} {doc.so_hieu}".strip()
            subject = f"V/v {doc.trich_yeu}" if doc.trich_yeu and doc.so_hieu else doc.trich_yeu
            parts = [head, ngay, doc.don_vi or doc.co_quan, subject]
            if not doc.so_hieu:
                parts.append(doc.title)
        else:
            code = _RX_DOC_CODE.search(doc.source_file)
            parts = [f"{doc.title} ({code.group(0)})" if code else doc.title, doc.lang.upper()]
        return "[" + " | ".join(p for p in parts if p) + "]"

    # --------------------------------------------------------------------------
    def _sections(self, profile: str, blocks: List[Block]) -> List[_Section]:
        sections: List[_Section] = [_Section([], "")]
        stack: List[Tuple[int, str]] = []
        in_article = False
        for b in blocks:
            if b.kind == "heading":
                # Trong Điều (H2), dòng "1. ..." là Khoản: nội dung, không phải đề mục
                if in_article and not b.legal and b.level >= 3:
                    sections[-1].blocks.append(Block("text", b.text, b.page_from, b.page_to))
                    continue
                in_article = b.legal == "dieu" or (in_article and not b.legal and b.level > 3)
                while stack and stack[-1][0] >= b.level:
                    stack.pop()
                stack.append((b.level, b.text))
                sections.append(_Section([t for _, t in stack], b.text, heading_page=b.page_from))
            else:
                sections[-1].blocks.append(b)
        # Mục chỉ có đề mục ("6. Thời gian thực hiện: Năm 2026." nhận là đề mục vì ngắn, có số thứ tự): đưa chính dòng đó
        # vào nội dung, nếu không nó không thuộc chunk nào và thông tin bị mất khỏi chỉ mục
        for sec in sections:
            if sec.heading and not sec.blocks:
                sec.blocks.append(Block("text", sec.heading, sec.heading_page, sec.heading_page))
        return [s for s in sections if s.blocks]

    def _parent_groups(self, profile: str, sections: List[_Section]) -> List[List[_Section]]:
        """Nhóm mục thành mục cha: theo Điều (H2) hoặc theo đề mục cấp cao nhất; quá ~2.000 token thì gom tuần tự."""
        if profile == "H2":
            key = lambda s: next((s.path[: k + 1] for k, t in enumerate(s.path) if re.match(r"(?i)^(điều)\s+\d+", t)), s.path[:1])
        else:
            tops = {tuple(s.path[:1]) for s in sections if s.path}
            depth = 1 if len(tops) > 1 else 2
            key = lambda s: s.path[:depth]

        groups: List[List[_Section]] = []
        for s in sections:
            if groups and key(groups[-1][-1]) == key(s):
                groups[-1].append(s)
            else:
                groups.append([s])

        out: List[List[_Section]] = []
        for g in groups:
            if approx_tokens("\n\n".join(s.text for s in g)) <= self.p.parent_max:
                out.append(g)
                continue
            cur: List[_Section] = []
            for s in g:
                if cur and approx_tokens("\n\n".join(x.text for x in cur + [s])) > self.p.parent_max:
                    out.append(cur)
                    cur = []
                cur.append(s)
            if cur:
                out.append(cur)
        return out

    def _chunk_segment(self, doc: DocRecord, profile: str, blocks: List[Block],
                       parents: List[ParentSection], chunks: List[Chunk]) -> None:
        if not blocks:
            return
        line = self.context_line(doc)
        sections = self._sections(profile, blocks)

        total = approx_tokens("\n\n".join(s.text for s in sections))
        if profile == "H1" and total <= self.p.h1_single_max:
            parent = self._add_parent(doc, parents, sections)
            self._emit(doc, chunks, parent, line, [], "\n\n".join(s.text for s in sections), sections[0].pages[0] or 1,
                       max(s.pages[1] for s in sections) or 1, blocks)
            return

        max_tokens = self.p.h2_article_max if profile == "H2" else (self.p.h1_single_max if profile == "H1" else self.p.child_max)
        overlap = 0 if profile in ("H1", "H2") else self.p.overlap
        unit_pattern = r"\n(?=\d{1,2}\.\s)|\n\s*\n" if profile == "H2" else r"\n\s*\n"

        for group in self._parent_groups(profile, sections):
            parent = self._add_parent(doc, parents, group)
            buf: List[Tuple[_Section, Block]] = []

            def flush() -> None:
                if not buf:
                    return
                paths = [s.path for s, _ in buf]
                crumb = _common_prefix(paths)
                parts: List[str] = []
                last_heading = None
                for s, b in buf:
                    if s.heading and s.heading != last_heading and s.path != crumb and b.text != s.heading:
                        parts.append(s.heading)
                    last_heading = s.heading
                    parts.append(b.text)
                text = "\n\n".join(parts)
                pf, pt = min(b.page_from for _, b in buf), max(b.page_to for _, b in buf)
                for piece in split_long_text(text, max_tokens, overlap, unit_pattern):
                    self._emit(doc, chunks, parent, line, crumb, piece, pf, pt, [])
                buf.clear()

            for s in group:
                for b in s.blocks:
                    if b.kind == "table":
                        flush()
                        self._emit_table(doc, chunks, parent, line, s.path, b)
                    else:
                        buf.append((s, b))
                if buf and approx_tokens("\n".join(b.text for _, b in buf)) >= (self.p.child_min if profile != "H2" else 0):
                    flush()
            flush()

    def _add_parent(self, doc: DocRecord, parents: List[ParentSection], group: List[_Section]) -> ParentSection:
        pages = [s.pages for s in group if s.blocks]
        parent = ParentSection(
            parent_id=f"{doc.doc_id}:p{len(parents):03d}",
            doc_id=doc.doc_id,
            breadcrumb=_join_path(_common_prefix([s.path for s in group])),
            text="\n\n".join(s.text for s in group),
            page_from=min((p[0] for p in pages), default=1),
            page_to=max((p[1] for p in pages), default=1),
        )
        parents.append(parent)
        return parent

    def _emit_table(self, doc: DocRecord, chunks: List[Chunk], parent: ParentSection, line: str, path: List[str], b: Block) -> None:
        pieces = [b.text] if approx_tokens(b.text) <= self.p.table_single_max else split_table(b.text, self.p.table_rows)
        for piece in pieces:
            self._emit(doc, chunks, parent, line, path, piece, b.page_from, b.page_to, [Block("table", piece, b.page_from, b.page_to)], is_table=True)

    def _emit(self, doc: DocRecord, chunks: List[Chunk], parent: ParentSection, line: str, path: List[str],
              text: str, page_from: int, page_to: int, blocks: List[Block], is_table: bool = False) -> None:
        crumb = _join_path(path)
        context = f"{line} › {crumb}" if crumb else line
        searchable = text
        for b in blocks:
            if b.kind == "table" and b.text in searchable:
                searchable = searchable.replace(b.text, flatten_table(b.text, caption=crumb))
        search_text = f"{context}\n{searchable}"
        seq = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}:{seq:04d}",
            doc_id=doc.doc_id,
            parent_id=parent.parent_id,
            seq=seq,
            page_from=page_from,
            page_to=page_to,
            breadcrumb=crumb,
            context=context,
            text=text,
            search_text=search_text,
            codes=extract_codes(search_text),
            is_table=is_table or any(b.kind == "table" for b in blocks),
            tokens=approx_tokens(text),
        ))
