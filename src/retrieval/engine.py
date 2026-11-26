"""
Hanoi Metro AI Agent - Two-branch Hybrid Search Engine (Mục 5.6 đề xuất).
1. Phân tích câu hỏi bằng quy tắc; 2. chạy song song nhánh ngữ nghĩa (Qdrant) và nhánh từ khóa với cùng bộ lọc
quyền; 3. gộp RRF: điểm = Σ 1/(60 + thứ hạng), chunk khớp chính xác số hiệu lên đầu, gộp bản E/V cùng pair_id
(ưu tiên tiếng Việt), tối đa 3 chunk mỗi văn bản; 4. reranker chấm lại, lấy mục cha của các chunk tốt nhất.
Câu hỏi liệt kê ("tất cả quyết định năm 2024 về…") trả thẳng danh sách từ kho văn bản thay vì top-k.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from src.indexing.models import Chunk, DocRecord, ParentSection
from src.retrieval.query import listing_keywords, parse_query
from src.store.base import DocumentStore, Filters, ParsedQuery, Ranked, SearchBranch, access_allows
from src.text.abbrev import AbbreviationDictionary

logger = logging.getLogger("hanoi_metro.engine")


@dataclass
class SearchConfig:
    rrf_k: int = 60
    branch_k: int = 50
    max_per_doc: int = 3
    rerank_top: int = 30
    final_k: int = 8
    parent_expand: int = 6


@dataclass
class Evidence:
    rank: int
    score: float
    chunk: Chunk
    doc: DocRecord
    parent: Optional[ParentSection] = None
    branches: Dict[str, int] = field(default_factory=dict)   # thứ hạng trong từng nhánh

    @property
    def pages(self) -> str:
        c = self.chunk
        return f"{c.page_from}" if c.page_from == c.page_to else f"{c.page_from}–{c.page_to}"

    @property
    def citation(self) -> str:
        head = f"{self.doc.so_hieu} – " if self.doc.so_hieu else ""
        return f"{head}{self.doc.source_file}, trang {self.pages}"

    def to_dict(self, with_parent: bool = True) -> Dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 5),
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.doc.doc_id,
            "citation": self.citation,
            "source_file": self.doc.source_file,
            "so_hieu": self.doc.so_hieu,
            "ngay_ban_hanh": self.doc.ngay_ban_hanh,
            "loai_vb": self.doc.loai_vb,
            "pages": self.pages,
            "breadcrumb": self.chunk.breadcrumb,
            "context": self.chunk.context,
            "text": self.chunk.text,
            "parent_text": self.parent.text if (with_parent and self.parent) else None,
            "branches": self.branches,
        }


@dataclass
class SearchResult:
    query: ParsedQuery
    evidence: List[Evidence]
    listing: Optional[Tuple[List[DocRecord], int]] = None
    degraded: List[str] = field(default_factory=list)     # nhánh bị lỗi (kết quả chỉ từ nhánh còn lại)
    latency_ms: float = 0.0


def rrf_fuse(results: Dict[str, Ranked], k: int) -> List[Tuple[str, float, Dict[str, int]]]:
    scores: Dict[str, float] = {}
    ranks: Dict[str, Dict[str, int]] = {}
    for branch, ranked in results.items():
        for cid, rank in ranked:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[branch] = rank
    return [(cid, s, ranks[cid]) for cid, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


class SearchEngine:
    def __init__(
        self,
        store: DocumentStore,
        branches: Sequence[SearchBranch],
        abbrev: AbbreviationDictionary,
        reranker=None,
        config: Optional[SearchConfig] = None,
    ):
        self.store = store
        self.branches = list(branches)
        self.abbrev = abbrev
        self.reranker = reranker
        self.cfg = config or SearchConfig()
        self._pool = ThreadPoolExecutor(max_workers=max(2, len(self.branches)), thread_name_prefix="search")

    def parse(self, text: str) -> ParsedQuery:
        return parse_query(text, self.abbrev)

    def list_documents(self, q: ParsedQuery, filters: Optional[Filters] = None, limit: int = 200) -> Tuple[List[DocRecord], int]:
        f = replace(filters or Filters())
        if q.loai_vb and not f.loai_vb:
            f.loai_vb = q.loai_vb
        if q.years and not (f.year_from or f.year_to):
            f.year_from, f.year_to = min(q.years), max(q.years)
        if q.huong_vb and not f.huong_vb:
            f.huong_vb = q.huong_vb
        return self.store.list_documents(f, keyword=listing_keywords(q), limit=limit)

    def search(self, query: Union[str, ParsedQuery], filters: Optional[Filters] = None, k: Optional[int] = None,
               branches: Optional[Sequence[str]] = None) -> SearchResult:
        started = time.perf_counter()
        q = self.parse(query) if isinstance(query, str) else query
        f = filters or Filters()
        k = k or self.cfg.final_k

        active = [b for b in self.branches if not branches or b.name in branches]
        futures = {b.name: self._pool.submit(b.search, q, f, self.cfg.branch_k) for b in active}
        results: Dict[str, Ranked] = {}
        degraded: List[str] = []
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except Exception as e:
                logger.warning(f"Nhánh '{name}' lỗi, dùng kết quả nhánh còn lại: {e}")
                degraded.append(name)
        if active and len(degraded) == len(active):
            raise RuntimeError(f"Tất cả nhánh tìm kiếm đều lỗi: {', '.join(degraded)}")

        fused = rrf_fuse(results, self.cfg.rrf_k)
        pool_ids = [cid for cid, _, _ in fused[: self.cfg.rerank_top * 3]]
        chunks = self.store.get_chunks(pool_ids)
        docs = self.store.get_documents(list({c.doc_id for c in chunks.values()}))

        candidates = self._diversify(fused, chunks, docs, q)
        if q.codes:
            owners = self._owner_evidence(q, f, {ev.doc.doc_id for ev in candidates})
            if owners:
                candidates = sorted(owners + candidates, key=lambda ev: self._code_tier(ev, q.codes))
        candidates = self._rerank(q, candidates[: self.cfg.rerank_top])
        top = candidates[:k]

        parent_ids = [ev.chunk.parent_id for ev in top[: self.cfg.parent_expand]]
        parents = self.store.get_parents(list(dict.fromkeys(parent_ids)))
        for i, ev in enumerate(top, start=1):
            ev.rank = i
            if i <= self.cfg.parent_expand:
                ev.parent = parents.get(ev.chunk.parent_id)

        listing = self.list_documents(q, f) if q.listing else None
        return SearchResult(q, top, listing, degraded, (time.perf_counter() - started) * 1000)

    def _diversify(self, fused, chunks: Dict[str, Chunk], docs: Dict[str, DocRecord], q: ParsedQuery) -> List[Evidence]:
        evidence = [
            Evidence(0, score, chunks[cid], docs[chunks[cid].doc_id], branches=ranks)
            for cid, score, ranks in fused
            if cid in chunks and chunks[cid].doc_id in docs
        ]
        # hỏi đích danh số hiệu/mã: chính văn bản mang số hiệu đó đứng đầu, rồi đến đoạn có nhắc tới mã
        # (văn bản trích dẫn "thực hiện văn bản số 1910/UBND-ĐT" không được vượt lên trên chính văn bản 1910)
        if q.codes:
            evidence.sort(key=lambda ev: (self._code_tier(ev, q.codes), -ev.score))

        vi_pairs = {ev.doc.pair_id for ev in evidence if ev.doc.pair_id and ev.doc.lang == "vi"}
        # Văn bản được hỏi đích danh theo số hiệu/mã: người dùng cần đọc kỹ chính văn bản đó, không giới hạn số chunk
        named = {ev.doc.doc_id for ev in evidence if q.codes and (set(q.codes) & ({ev.doc.so_hieu.upper()} | set(ev.chunk.codes)))}
        per_doc: Dict[str, int] = {}
        out: List[Evidence] = []
        for ev in evidence:
            if ev.doc.pair_id in vi_pairs and ev.doc.lang != "vi" and ev.doc.doc_id not in named:
                continue
            n = per_doc.get(ev.doc.doc_id, 0)
            if n >= self.cfg.max_per_doc and ev.doc.doc_id not in named:
                continue
            per_doc[ev.doc.doc_id] = n + 1
            out.append(ev)
        return out

    @staticmethod
    def _code_tier(ev: Evidence, codes: Sequence[str]) -> int:
        """0: văn bản có chính số hiệu/mã được hỏi; 1: đoạn có nhắc tới mã; 2: còn lại."""
        wanted = set(codes)
        so_hieu = ev.doc.so_hieu.upper()
        if so_hieu and (so_hieu in wanted or any(so_hieu.startswith(c + "-") for c in wanted if len(c) >= 12)):
            return 0
        return 1 if wanted & set(ev.chunk.codes) else 2

    @staticmethod
    def _passes(doc: DocRecord, f: Filters) -> bool:
        """Cùng điều kiện các nhánh tìm kiếm áp dụng (kể cả quyền đơn vị): văn bản chèn thêm không được vượt quyền."""
        if doc.muc_mat > f.max_muc_mat:
            return False
        if f.doc_ids is not None and doc.doc_id not in f.doc_ids:
            return False
        if f.don_vi and doc.don_vi not in f.don_vi:
            return False
        if f.loai_vb and doc.loai_vb not in f.loai_vb:
            return False
        if f.lang and doc.lang != f.lang:
            return False
        if (f.year_from or f.year_to) and doc.nam is None:
            return False
        if f.year_from and doc.nam < f.year_from:
            return False
        if f.year_to and doc.nam > f.year_to:
            return False
        return True

    def _owner_evidence(self, q: ParsedQuery, f: Filters, present: Set[str]) -> List[Evidence]:
        """Văn bản mang đúng số hiệu được hỏi nhưng không lọt vào top của 2 nhánh tìm kiếm: lấy đoạn đầu của nó."""
        out: List[Evidence] = []
        for code in q.codes:
            for doc in self.store.find_documents_by_code(code, limit=3):
                if doc.doc_id in present or not self._passes(doc, f):
                    continue
                if f.huong_vb and self._huong_vb(doc.doc_id) not in f.huong_vb:
                    continue
                if f.access is not None and not access_allows(
                        f.access, doc.doc_id, self.store.access_labels([doc.doc_id]).get(doc.doc_id, [])):
                    continue
                chunks = self.store.get_document_chunks(doc.doc_id)[:2]
                out += [Evidence(0, 0.0, c, doc, branches={}) for c in chunks]
                present.add(doc.doc_id)
        return out

    def _huong_vb(self, doc_id: str) -> str:
        field = self.store.doc_fields([doc_id], ["huong_vb"]).get(doc_id, {}).get("huong_vb")
        return field.gia_tri if field else ""

    def _rerank(self, q: ParsedQuery, candidates: List[Evidence]) -> List[Evidence]:
        if not self.reranker or len(candidates) < 2:
            return candidates
        scores = self.reranker.compute_score(q.text, [ev.chunk.search_text for ev in candidates])
        pinned = sorted((ev for ev in candidates if q.codes and self._code_tier(ev, q.codes) < 2),
                        key=lambda ev: self._code_tier(ev, q.codes))
        rest = [(s, ev) for s, ev in zip(scores, candidates) if ev not in pinned]
        rest.sort(key=lambda x: x[0], reverse=True)
        for s, ev in rest:
            ev.score = float(s)
        return pinned + [ev for _, ev in rest]
