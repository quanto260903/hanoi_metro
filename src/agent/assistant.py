"""
Hanoi Metro AI Agent - Document Assistant.
Các chức năng trên kho tài liệu nội bộ:
- ask: hỏi đáp có trích dẫn [n] → file gốc, trang; câu hỏi liệt kê trả danh sách đầy đủ từ kho văn bản.
- summarize: tóm tắt một văn bản (theo số hiệu / doc_id / tên), văn bản dài tóm tắt map-reduce theo từng phần.
- draft_report: soạn dự thảo báo cáo / tờ trình / công văn / bản tổng hợp từ tài liệu liên quan, khối thể thức
  theo Nghị định 30, chỗ thiếu thông tin đánh dấu [cần bổ sung].
Danh sách nguồn do mã chương trình ghép từ số [n] mà LLM trích, nên luôn trỏ đúng file và trang.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from src.agent import prompts
from src.agent.llm import ChatModel
from src.config import AgentConfig
from src.indexing.enrich import document_warnings
from src.indexing.models import DocRecord
from src.retrieval.engine import Evidence, SearchEngine, SearchResult
from src.store.base import DocumentStore, Filters, access_allows
from src.text.codes import extract_codes
from src.text.normalize import approx_tokens, content_terms

logger = logging.getLogger("hanoi_metro.assistant")


@dataclass
class Source:
    n: int
    citation: str
    doc_id: str
    chunk_id: str
    source_file: str
    pages: str
    body: str = field(default="", repr=False)
    anchor: str = ""            # trang của đoạn liên quan nhất: liên kết mở tài liệu gốc nhảy tới đây


@dataclass
class Answer:
    text: str                                   # Markdown hoàn chỉnh (nội dung + nguồn)
    sources: List[Source] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    document: Optional[DocRecord] = None


class DocumentNotFound(LookupError):
    pass


def _fmt_date(iso: str) -> str:
    return "/".join(reversed(iso.split("-"))) if iso else ""


def format_pages(pages: Sequence[int]) -> str:
    """{2, 3, 5} → "2–3, 5"."""
    out: List[str] = []
    nums = sorted(set(pages))
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}" if i == j else f"{nums[i]}–{nums[j]}")
        i = j + 1
    return ", ".join(out)


def build_context(evidence: Sequence[Evidence], budget_tokens: int) -> Tuple[str, List[Source]]:
    """
    Đánh số [n] theo VĂN BẢN: nhiều đoạn của cùng một văn bản (vd. mục 4, 5, 6 của một quyết định) dùng chung một số,
    nguồn gộp các trang, nên danh sách nguồn không lặp một văn bản nhiều lần. Ưu tiên mục cha, không lặp mục cha.
    """
    parts: List[str] = []
    sources: List[Source] = []
    by_doc: Dict[str, Tuple[Source, set]] = {}
    used_parents = set()
    used = 0
    for ev in evidence:
        if ev.parent and ev.parent.parent_id not in used_parents and approx_tokens(ev.parent.text) <= budget_tokens // 2:
            body = ev.parent.text
            used_parents.add(ev.parent.parent_id)
        elif ev.parent and ev.parent.parent_id in used_parents:
            continue
        else:
            body = ev.chunk.text
        cost = approx_tokens(body)
        if used + cost > budget_tokens:
            if parts:
                break
            body = " ".join(body.split()[: int(budget_tokens / 1.3)])
        span = range(ev.chunk.page_from, ev.chunk.page_to + 1)
        if ev.doc.doc_id in by_doc:
            src, pages = by_doc[ev.doc.doc_id]
            pages.update(span)
            src.pages = format_pages(pages)
            src.citation = ev.citation.rsplit(", trang ", 1)[0] + f", trang {src.pages}"
            src.body += "\n\n" + body
        else:
            src = Source(len(sources) + 1, ev.citation, ev.doc.doc_id, ev.chunk.chunk_id, ev.doc.source_file, ev.pages, body,
                         anchor=ev.pages)
            by_doc[ev.doc.doc_id] = (src, set(span))
            sources.append(src)
        parts.append(f"[{src.n}] {ev.citation}\n{ev.chunk.context}\n{body}")
        used += cost
    return "\n\n---\n\n".join(parts), sources


def cited_sources(body: str, sources: List[Source]) -> List[Source]:
    """
    Nguồn được LLM trích [n]. Mô hình nhỏ hay quên ghi [n]: khi đó giữ các nguồn có nhiều từ/mã trùng với
    câu trả lời (ít nhất một nửa mức trùng cao nhất), thay vì liệt kê cả những đoạn không dùng tới.
    """
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", body)}
    if cited:
        return [s for s in sources if s.n in cited]
    if len(sources) < 2:
        return list(sources)
    answer_terms = {t for t in content_terms(body) if len(t) > 2} | set(extract_codes(body))
    overlap = [(len(answer_terms & ({t for t in content_terms(s.body)} | set(extract_codes(s.body)))), s) for s in sources]
    best = max(n for n, _ in overlap)
    return [s for n, s in overlap if n and n >= best / 2] or list(sources)


_RX_FACT = re.compile(r"\d{1,5}(?:[/.,:-]\d+)+|(?<![\d\[])\d{2,}(?![\d\]])|\b[A-ZĐ]{1,6}\d[A-Z0-9-]{2,}\b")


def unsupported_facts(body: str, context: str) -> List[str]:
    """Số liệu, ngày tháng, mã trong câu trả lời mà không có trong tài liệu nguồn: dấu hiệu LLM bịa, cần người kiểm tra."""
    text = re.sub(r"\[\d+\]", " ", body)
    compact = context.replace(" ", "")
    facts = dict.fromkeys(m.group(0).rstrip(".,") for m in _RX_FACT.finditer(text))
    return [f for f in facts if f not in context and f.replace(" ", "") not in compact]


# (doc_id, source_file, pages) → URL mở file gốc tại trang được trích; None = trích dẫn dạng chữ
Linker = Callable[[str, str, str], Optional[str]]

_RX_MD_SPECIAL = re.compile(r"([\\\[\]*_`])")


def cite_text(s: Source, linker: Optional[Linker] = None) -> str:
    url = linker(s.doc_id, s.source_file, s.anchor or s.pages) if linker else None
    if not url:
        return s.citation
    label = _RX_MD_SPECIAL.sub(r"\\\1", s.citation)
    return f"[{label}]({url})"


def sources_block(sources: List[Source], title: str = "Nguồn", linker: Optional[Linker] = None) -> str:
    if not sources:
        return ""
    return f"**{title}:**\n" + "\n".join(f"- [{s.n}] {cite_text(s, linker)}" for s in sources)


def render_sources(body: str, sources: List[Source], title: str = "Nguồn",
                   linker: Optional[Linker] = None) -> Tuple[str, List[Source]]:
    chosen = cited_sources(body, sources)
    block = sources_block(chosen, title, linker)
    return (f"{body}\n\n{block}" if block else body), chosen


def format_listing(docs: List[DocRecord], total: int) -> str:
    if not docs:
        return "Không có văn bản nào khớp điều kiện."
    rows = ["| STT | Số hiệu | Ngày | Loại | Trích yếu | File |", "|---|---|---|---|---|---|"]
    for i, d in enumerate(docs, start=1):
        subject = (d.trich_yeu or d.title).replace("|", "/")
        rows.append(f"| {i} | {d.so_hieu or '—'} | {_fmt_date(d.ngay_ban_hanh) or '—'} | {d.loai_vb or '—'} | {subject} | {d.source_file.replace('|', '/')} |")
    more = f"\n\n*Hiển thị {len(docs)}/{total} văn bản.*" if total > len(docs) else f"\n\n*Tổng cộng {total} văn bản.*"
    return "\n".join(rows) + more



class DocumentAssistant:
    def __init__(self, engine: SearchEngine, store: DocumentStore, llm: ChatModel, config: AgentConfig,
                 outbound_guard: Optional[Callable[[Sequence[str]], None]] = None, linker: Optional[Linker] = None):
        self.engine = engine
        self.store = store
        self.llm = llm
        self.cfg = config
        # gọi trước mỗi lần gửi nội dung tài liệu tới LLM bên ngoài; ném lỗi nếu có tài liệu bị chặn
        self.outbound_guard = outbound_guard
        # tạo liên kết mở tài liệu gốc trong danh sách nguồn (gateway gắn vào; CLI/MCP để None = dạng chữ)
        self.linker = linker

    # ------------------------------------------------------------------ tra cứu
    def search(self, query: str, filters: Optional[Filters] = None, k: Optional[int] = None) -> SearchResult:
        return self.engine.search(query, filters, k)

    def resolve_document(self, ref: str, filters: Optional[Filters] = None) -> Tuple[DocRecord, List[str]]:
        """Tìm văn bản theo doc_id, số hiệu (khớp chính xác rồi gần đúng) hoặc mô tả tự nhiên."""
        ref = ref.strip()
        notes: List[str] = []
        f = filters or Filters()
        if re.fullmatch(r"[0-9a-fA-F]{32}", ref):
            doc = self.store.get_document(ref.lower())
            if doc and self._allowed(doc, f):
                return doc, notes
            raise DocumentNotFound(f"Không có văn bản doc_id={ref}")
        for code in dict.fromkeys(extract_codes(ref) + extract_codes(ref.upper())):
            docs = [d for d in self.store.find_documents_by_code(code) if self._allowed(d, f)]
            if docs:
                if docs[0].so_hieu.upper() != code:
                    notes.append(f"Không có số hiệu {code}; dùng văn bản gần nhất {docs[0].so_hieu}.")
                return docs[0], notes
        res = self.engine.search(ref, f, k=3)
        if not res.evidence:
            raise DocumentNotFound(f"Không tìm thấy văn bản khớp '{ref}'")
        doc = res.evidence[0].doc
        notes.append(f"Đã chọn văn bản khớp nhất với mô tả: {doc.so_hieu or doc.source_file}.")
        return doc, notes

    def _complete(self, system: str, user: str) -> str:
        return self.llm.complete(system, user, temperature=self.cfg.temperature)

    @staticmethod
    def _flag_unsupported(body: str, context: str, notes: List[str]) -> None:
        facts = unsupported_facts(body, context)
        if facts:
            notes.append("Cần kiểm tra – số liệu/mã không thấy trong tài liệu nguồn: " + ", ".join(facts[:10]))

    def _allowed(self, doc: DocRecord, f: Filters) -> bool:
        if doc.muc_mat > f.max_muc_mat or (f.don_vi and doc.don_vi not in f.don_vi):
            return False
        if f.access is None:
            return True
        return access_allows(f.access, doc.doc_id, self.store.access_labels([doc.doc_id]).get(doc.doc_id, []))

    def describe(self, doc: DocRecord) -> str:
        rows = [
            ("Số hiệu", doc.so_hieu), ("Loại", doc.loai_vb), ("Ngày", _fmt_date(doc.ngay_ban_hanh)),
            ("Cơ quan / đơn vị", " – ".join(dict.fromkeys(x for x in (doc.co_quan, doc.don_vi) if x))),
            ("Trích yếu", doc.trich_yeu), ("Kính gửi", "; ".join(doc.kinh_gui)), ("Người ký", doc.nguoi_ky),
            ("Số trang", str(doc.total_pages)), ("File", doc.source_file), ("Văn bản được dẫn chiếu", ", ".join(doc.cites)),
            ("doc_id", doc.doc_id),
        ]
        return "\n".join(f"- **{k}:** {v}" for k, v in rows if v)

    def read_document(self, ref: str, page_from: Optional[int] = None, page_to: Optional[int] = None,
                      filters: Optional[Filters] = None, max_tokens: int = 8000) -> Answer:
        doc, notes = self.resolve_document(ref, filters)
        parts, used = [], 0
        for ch in self.store.get_document_chunks(doc.doc_id):
            if (page_from and ch.page_to < page_from) or (page_to and ch.page_from > page_to):
                continue
            if used + ch.tokens > max_tokens:
                notes.append(f"Nội dung bị cắt ở trang {ch.page_from} (giới hạn {max_tokens} token); đọc tiếp bằng tham số trang.")
                break
            crumb = f"*{ch.breadcrumb}* (trang {ch.page_from})\n" if ch.breadcrumb else f"(trang {ch.page_from})\n"
            parts.append(crumb + ch.text)
            used += ch.tokens
        text = f"### {doc.so_hieu or doc.title}\n{self.describe(doc)}\n\n" + "\n\n".join(parts)
        return Answer(text, notes=notes, document=doc)

    def list_documents(self, query: str = "", filters: Optional[Filters] = None, limit: int = 50) -> Answer:
        q = self.engine.parse(query) if query else self.engine.parse("tất cả")
        docs, total = self.engine.list_documents(q, filters, limit=limit)
        return Answer(format_listing(docs, total))

    # ------------------------------------------------------------------ chạy kế hoạch sinh nội dung
    def run(self, plan: "Plan") -> Answer:
        """Sinh trọn câu trả lời (CLI, MCP)."""
        if not plan.system:
            return Answer(plan.direct, plan.sources, plan.notes, plan.document)
        self._check_outbound(plan.doc_ids)
        body = self._complete(plan.system, plan.user)
        return self._finish(plan, body)

    def run_stream(self, plan: "Plan") -> Iterator[str]:
        """Trả từng mảnh chữ (dịch vụ hỏi đáp 1 lượt cho LibreChat); cảnh báo kiểm tra số liệu ở cuối."""
        if not plan.system:
            yield plan.direct
            return
        self._check_outbound(plan.doc_ids)
        if plan.head:
            yield plan.head
        parts: List[str] = []
        for piece in self.llm.stream(plan.system, plan.user, temperature=self.cfg.temperature):
            parts.append(piece)
            yield piece
        body = "".join(parts)
        answer = self._finish(plan, body)
        tail = answer.text[len(plan.head) + len(body):]
        if tail:
            yield tail
        if answer.notes:
            yield "\n\n" + "\n".join(f"> {n}" for n in answer.notes)

    def _finish(self, plan: "Plan", body: str) -> Answer:
        notes = list(plan.notes)
        self._flag_unsupported(body, plan.context, notes)
        cited = cited_sources(body, plan.sources) if plan.sources else []
        used = [s.doc_id for s in cited + plan.fixed_sources] + ([plan.document.doc_id] if plan.document else [])
        notes += document_warnings(self.store, used, self.cfg.low_read_ratio)
        tail = plan.tail(body, cited) if plan.tail else ""
        return Answer(f"{plan.head}{body}{tail}", cited, notes, plan.document)

    def _check_outbound(self, doc_ids: Sequence[str]) -> None:
        """Kiểm tra lại danh sách chặn tài liệu mật ngay trước khi nội dung rời máy chủ nội bộ."""
        if self.outbound_guard and getattr(self.llm, "external", False):
            self.outbound_guard(doc_ids)

    # ------------------------------------------------------------------ hỏi đáp
    def plan_ask(self, question: str, filters: Optional[Filters] = None, history: str = "",
                 search_query: Optional[str] = None) -> "Plan":
        res = self.engine.search(search_query or question, filters)
        notes = [f"Nhánh tìm kiếm '{b}' đang lỗi; kết quả chỉ từ nhánh còn lại." for b in res.degraded]
        if res.query.listing and res.listing:
            docs, total = res.listing
            return Plan(direct=format_listing(docs, total), notes=notes)
        if not res.evidence:
            return Plan(direct="Không tìm thấy tài liệu nội bộ nào liên quan đến câu hỏi.", notes=notes)
        context, sources = build_context(res.evidence, self.cfg.context_budget_tokens)
        user = prompts.ANSWER_USER.format(context=context, question=question)
        if history:
            user = f"LỊCH SỬ HỘI THOẠI GẦN ĐÂY (chỉ để hiểu câu hỏi, không phải nguồn):\n{history}\n\n{user}"
        return Plan(
            system=prompts.ANSWER_SYSTEM.format(org=self.cfg.organization), user=user, context=context,
            sources=sources, notes=notes, tail=lambda body, cited: "\n\n" + sources_block(cited, linker=self.linker) if cited else "",
        )

    def ask(self, question: str, filters: Optional[Filters] = None) -> Answer:
        return self.run(self.plan_ask(question, filters))

    # ------------------------------------------------------------------ tóm tắt
    def plan_summarize(self, ref: str, style: str = "chi_tiet", focus: str = "", filters: Optional[Filters] = None) -> "Plan":
        doc, notes = self.resolve_document(ref, filters)
        chunks = self.store.get_document_chunks(doc.doc_id)
        if not chunks:
            raise DocumentNotFound(f"Văn bản {doc.doc_id} chưa có nội dung trong chỉ mục")
        style_text = prompts.SUMMARY_STYLES.get(style, prompts.SUMMARY_STYLES["chi_tiet"])
        focus_text = f"\nTrọng tâm: {focus}" if focus else ""
        title = " – ".join(x for x in (f"{doc.loai_vb} {doc.so_hieu}".strip(), doc.trich_yeu or doc.title) if x)
        system = prompts.SUMMARY_SYSTEM.format(org=self.cfg.organization)

        texts, last_crumb = [], None
        for ch in chunks:
            texts.append(f"{ch.breadcrumb}\n{ch.text}" if ch.breadcrumb and ch.breadcrumb != last_crumb else ch.text)
            last_crumb = ch.breadcrumb
        full = "\n\n".join(texts)

        if approx_tokens(full) <= self.cfg.context_budget_tokens:
            user = prompts.SUMMARY_SINGLE_USER.format(title=title, style=style_text, focus=focus_text, text=full)
        else:
            # văn bản dài: tóm tắt từng phần trước (không phát trực tiếp), lượt gộp cuối mới phát cho người dùng
            self._check_outbound([doc.doc_id])
            batches = self._batch(texts, self.cfg.summary_batch_tokens)
            partials = [
                self._complete(system, prompts.SUMMARY_MAP_USER.format(title=title, part=i, total=len(batches), text=b))
                for i, b in enumerate(batches, start=1)
            ]
            while approx_tokens("\n\n".join(partials)) > self.cfg.context_budget_tokens and len(partials) > 1:
                groups = self._batch(partials, self.cfg.context_budget_tokens)
                partials = [self._complete(system, prompts.SUMMARY_REDUCE_USER.format(title=title, style="Gộp thành gạch đầu dòng.", focus="", text=g)) for g in groups]
            user = prompts.SUMMARY_REDUCE_USER.format(title=title, style=style_text, focus=focus_text, text="\n\n".join(partials))
            notes.append(f"Văn bản dài: tóm tắt theo {len(batches)} phần rồi gộp lại.")

        pages = f"trang 1–{doc.total_pages}" if doc.total_pages > 1 else "trang 1"
        source = Source(1, f"{doc.so_hieu + ' – ' if doc.so_hieu else ''}{doc.source_file}, {pages}", doc.doc_id, chunks[0].chunk_id, doc.source_file, pages)
        return Plan(
            head=f"### Tóm tắt: {title}\n{self.describe(doc)}\n\n", system=system, user=user, context=full,
            notes=notes, document=doc, extra_doc_ids=[doc.doc_id], fixed_sources=[source],
            tail=lambda body, cited: f"\n\n**Nguồn:** {cite_text(source, self.linker)}",
        )

    def summarize(self, ref: str, style: str = "chi_tiet", focus: str = "", filters: Optional[Filters] = None) -> Answer:
        plan = self.plan_summarize(ref, style, focus, filters)
        answer = self.run(plan)
        answer.sources = plan.fixed_sources
        return answer

    @staticmethod
    def _batch(texts: Sequence[str], budget: int) -> List[str]:
        batches: List[str] = []
        cur: List[str] = []
        for t in texts:
            if cur and approx_tokens("\n\n".join(cur + [t])) > budget:
                batches.append("\n\n".join(cur))
                cur = []
            cur.append(t)
        if cur:
            batches.append("\n\n".join(cur))
        return batches

    # ------------------------------------------------------------------ soạn báo cáo
    def plan_report(self, topic: str, report_type: str = "bao_cao", doc_refs: Optional[Sequence[str]] = None,
                    recipient: str = "", unit: str = "", instructions: str = "", filters: Optional[Filters] = None) -> "Plan":
        spec = prompts.REPORT_TYPES.get(report_type)
        if spec is None:
            raise ValueError(f"Loại văn bản không hỗ trợ: {report_type}. Chọn: {', '.join(prompts.REPORT_TYPES)}")
        notes: List[str] = []
        evidence: List[Evidence] = []
        if doc_refs:
            for ref in doc_refs:
                doc, n = self.resolve_document(ref, filters)
                notes.extend(n)
                evidence.extend(Evidence(0, 0.0, ch, doc) for ch in self.store.get_document_chunks(doc.doc_id))
        else:
            res = self.engine.search(topic, filters, k=12)
            evidence = res.evidence
            notes.extend(f"Nhánh tìm kiếm '{b}' đang lỗi." for b in res.degraded)
        if not evidence:
            return Plan(direct=f"Không tìm thấy tài liệu nội bộ liên quan đến chủ đề '{topic}' để soạn {spec['ten'].lower() or 'công văn'}.", notes=notes)

        context, sources = build_context(evidence, self.cfg.context_budget_tokens)
        extra = "\n".join(x for x in (
            f"Kính gửi: {recipient}" if recipient else "",
            f"Đơn vị soạn thảo: {unit}" if unit else "",
            f"Yêu cầu thêm: {instructions}" if instructions else "",
        ) if x)
        head, foot = self._form_parts(report_type, spec, topic, recipient, unit)
        return Plan(
            head=head,
            system=prompts.REPORT_SYSTEM.format(org=self.cfg.organization, loai=(spec["ten"] or "công văn").lower()),
            user=prompts.REPORT_USER.format(topic=topic, extra=extra, outline="\n".join(spec["muc"]), context=context),
            context=context, sources=sources, notes=notes,
            tail=lambda body, cited: f"{foot}\n\n{sources_block(cited, 'Tài liệu tham khảo (nội bộ)', self.linker)}",
        )

    def draft_report(self, topic: str, report_type: str = "bao_cao", doc_refs: Optional[Sequence[str]] = None,
                     recipient: str = "", unit: str = "", instructions: str = "", filters: Optional[Filters] = None) -> Answer:
        return self.run(self.plan_report(topic, report_type, doc_refs, recipient, unit, instructions, filters))

    def _form_parts(self, report_type: str, spec: Dict, topic: str, recipient: str, unit: str) -> Tuple[str, str]:
        """Khối thể thức Nghị định 30 (phần trước và sau nội dung): cơ quan, quốc hiệu – tiêu ngữ, số ký hiệu, nơi nhận, chữ ký."""
        if report_type == "tong_hop":
            return f"# {spec['ten']}\n**{topic}**\n\n", ""
        unit_name = unit or "[TÊN ĐƠN VỊ SOẠN THẢO]"
        today = date.today()
        so = f"Số: …/{spec['ky_hieu']}-[viết tắt đơn vị]"
        vv = f"<br>V/v {topic}" if report_type == "cong_van" else ""
        head = [
            f"| {self.cfg.organization}<br>**{unit_name.upper()}** | **CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**<br>**Độc lập - Tự do - Hạnh phúc** |",
            "|:---:|:---:|",
            f"| {so}{vv} | *Hà Nội, ngày … tháng {today.month:02d} năm {today.year}* |",
            "",
        ]
        if spec["ten"]:
            head += [f"<p align=\"center\"><b>{spec['ten']}</b><br><b>{topic}</b></p>", ""]
        if report_type in ("to_trinh", "cong_van"):
            head += [f"Kính gửi: {recipient or '[cần bổ sung: nơi nhận]'}", ""]
        foot = [
            "",
            "",
            "| **Nơi nhận:**<br>- Như trên;<br>- Lưu: VT, " + (unit or "[đơn vị]") + ". | **[CHỨC VỤ NGƯỜI KÝ]**<br><br>*(Ký, ghi rõ họ tên)* |",
            "|:---|:---:|",
            "",
            "> *Dự thảo do trợ lý AI soạn từ tài liệu nội bộ; cán bộ soạn thảo kiểm tra số liệu, bổ sung các mục [cần bổ sung] trước khi trình ký.*",
        ]
        return "\n".join(head) + "\n", "\n".join(foot)


@dataclass
class Plan:
    """Một lượt sinh nội dung đã chuẩn bị: phần đầu cố định, prompt, nguồn và cách ghép phần cuối."""
    direct: str = ""                                   # trả thẳng, không gọi LLM (liệt kê, không tìm thấy)
    head: str = ""
    system: str = ""
    user: str = ""
    context: str = ""
    sources: List[Source] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    document: Optional[DocRecord] = None
    tail: Optional[Callable[[str, List[Source]], str]] = None
    extra_doc_ids: List[str] = field(default_factory=list)
    fixed_sources: List[Source] = field(default_factory=list)

    @property
    def doc_ids(self) -> List[str]:
        return list(dict.fromkeys([s.doc_id for s in self.sources] + self.extra_doc_ids))
