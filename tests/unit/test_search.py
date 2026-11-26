from dataclasses import replace

import pytest

from src.indexing.models import Chunk, DocRecord, ParentSection
from src.retrieval.engine import SearchConfig, SearchEngine, rrf_fuse
from src.retrieval.query import parse_query
from src.store.base import Filters
from src.text.abbrev import AbbreviationDictionary

from conftest import ROOT

ABBREV = AbbreviationDictionary.load(ROOT / "data" / "dictionaries" / "synonym_dict.json")
CV_ID = "622bde1f7dbcca665917be569946d1cd"
BB_ID = "e2311c730de842ef3c56314f98f4250e"


class StaticBranch:
    def __init__(self, name, ranked=None, error=None):
        self.name = name
        self.ranked = ranked or []
        self.error = error

    def search(self, query, filters, k):
        if self.error:
            raise self.error
        return self.ranked[:k]


def test_parse_query_detects_codes_years_listing_and_abbreviations():
    q = parse_query("Liệt kê tất cả quyết định năm 2024 về AFC liên quan 916/CV-VT&QLPT", ABBREV)
    assert q.codes == ["916/CV-VT&QLPT"]
    assert q.years == [2024] and q.listing and "QĐ" in q.loai_vb
    assert any(e.matched == "AFC" for e in q.expansions)


def test_keyword_branch_exact_code_then_unaccented_fulltext(indexed_store):
    top = indexed_store.search(parse_query("số hiệu 916/CV-VT&QLPT", ABBREV), Filters(), 5)
    assert top[0][0].startswith(CV_ID)
    top = indexed_store.search(parse_query("tinh trang san switch tai tu rack", ABBREV), Filters(), 5)
    assert top[0][0].startswith(BB_ID)
    top = indexed_store.search(parse_query("server DELL J89T3F4", ABBREV), Filters(), 5)
    assert top[0][0].startswith(BB_ID)


def test_keyword_branch_applies_filters_before_ranking(indexed_store):
    assert indexed_store.search(parse_query("916/CV-VT&QLPT", ABBREV), Filters(loai_vb=["BB"]), 5) == [] or \
        all(cid.startswith(BB_ID) for cid, _ in indexed_store.search(parse_query("916/CV-VT&QLPT", ABBREV), Filters(loai_vb=["BB"]), 5))
    assert indexed_store.search(parse_query("kiểm đếm thiết bị", ABBREV), Filters(year_from=2030), 5) == []


def test_fuzzy_document_number_lookup(indexed_store):
    docs = indexed_store.find_documents_by_code("916/CV-VTQLPT")
    assert docs and docs[0].doc_id == CV_ID


def test_list_documents_is_complete_and_filtered(indexed_store):
    docs, total = indexed_store.list_documents(Filters())
    assert total == 2
    docs, total = indexed_store.list_documents(Filters(loai_vb=["BB"], year_from=2026, year_to=2026))
    assert total == 1 and docs[0].doc_id == BB_ID
    docs, total = indexed_store.list_documents(Filters(), keyword="khảo sát AFC")
    assert total == 1 and docs[0].doc_id == CV_ID


def test_rrf_fuse_sums_reciprocal_ranks():
    fused = rrf_fuse({"vector": [("a", 1), ("b", 2)], "keyword": [("b", 1)]}, k=60)
    assert fused[0][0] == "b"
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0][2] == {"vector": 2, "keyword": 1}


def test_engine_degrades_when_vector_branch_fails(indexed_store):
    engine = SearchEngine(indexed_store, [StaticBranch("vector", error=ConnectionError("ollama off")), indexed_store], ABBREV)
    res = engine.search("biên bản kiểm đếm thiết bị Duy Tân")
    assert res.degraded == ["vector"]
    assert res.evidence[0].doc.doc_id == BB_ID
    assert res.evidence[0].parent is not None, "chunk tốt nhất được mở rộng sang mục cha"
    assert res.evidence[0].citation == "Biên bản thiết bị DC ở Duy Tân.pdf, trang 1–2"


def test_engine_raises_when_every_branch_fails(indexed_store):
    engine = SearchEngine(indexed_store, [StaticBranch("vector", error=ConnectionError("x"))], ABBREV)
    with pytest.raises(RuntimeError):
        engine.search("bất kỳ")


def test_exact_code_match_is_pinned_first(indexed_store):
    vector = StaticBranch("vector", [(f"{BB_ID}:0000", 1), (f"{CV_ID}:0000", 2)])
    engine = SearchEngine(indexed_store, [vector], ABBREV)
    res = engine.search("485/TT-KHDA")
    assert res.evidence[0].doc.doc_id == CV_ID


class _MemoryStore:
    """Kho tối thiểu để kiểm tra luật đa dạng hóa kết quả với dữ liệu dựng tay."""

    def __init__(self, docs, chunks):
        self.docs = {d.doc_id: d for d in docs}
        self.chunks = {c.chunk_id: c for c in chunks}

    def get_chunks(self, ids):
        return {i: self.chunks[i] for i in ids if i in self.chunks}

    def get_documents(self, ids):
        return {i: self.docs[i] for i in ids if i in self.docs}

    def get_parents(self, ids):
        return {i: ParentSection(i, "x", "", "cha", 1, 1) for i in ids}

    def find_documents_by_code(self, code, limit=5):
        return [d for d in self.docs.values() if d.so_hieu.upper() == code][:limit]

    def get_document_chunks(self, doc_id):
        return sorted((c for c in self.chunks.values() if c.doc_id == doc_id), key=lambda c: c.seq)


def _chunk(doc_id, seq):
    return Chunk(f"{doc_id}:{seq:04d}", doc_id, f"{doc_id}:p000", seq, 1, 1, "", "[ctx]", "nội dung", "[ctx]\nnội dung")


def test_max_three_chunks_per_document_and_vietnamese_pair_preferred():
    base = DocRecord(doc_id="", source_file="", source_path="", title="", total_pages=1)
    vi = replace(base, doc_id="v" * 32, source_file="UJV-OPM-001_V.pdf", lang="vi", pair_id="UJV-OPM-001")
    en = replace(base, doc_id="e" * 32, source_file="UJV-OPM-001_E.pdf", lang="en", pair_id="UJV-OPM-001")
    other = replace(base, doc_id="o" * 32, source_file="khac.pdf")
    chunks = [_chunk(vi.doc_id, i) for i in range(5)] + [_chunk(en.doc_id, 0), _chunk(other.doc_id, 0)]
    ranked = [(c.chunk_id, r) for r, c in enumerate(chunks, start=1)]
    engine = SearchEngine(_MemoryStore([vi, en, other], chunks), [StaticBranch("vector", ranked)], ABBREV, config=SearchConfig(final_k=10))
    docs = [ev.doc.doc_id for ev in engine.search("sổ tay").evidence]
    assert docs.count(vi.doc_id) == 3
    assert en.doc_id not in docs
    assert other.doc_id in docs


def test_document_named_by_number_is_not_capped():
    base = DocRecord(doc_id="", source_file="", source_path="", title="", total_pages=5)
    target = replace(base, doc_id="t" * 32, source_file="1977-QĐ-SXD.pdf", so_hieu="1977/QĐ-SXD")
    other = replace(base, doc_id="o" * 32, source_file="khac.pdf")
    chunks = [replace(_chunk(target.doc_id, i), codes=["1977/QĐ-SXD"]) for i in range(6)] + [_chunk(other.doc_id, i) for i in range(5)]
    ranked = [(c.chunk_id, r) for r, c in enumerate(chunks, start=1)]
    engine = SearchEngine(_MemoryStore([target, other], chunks), [StaticBranch("vector", ranked)], ABBREV, config=SearchConfig(final_k=12))
    docs = [ev.doc.doc_id for ev in engine.search("Quyết định 1977/QĐ-SXD chi phí bao nhiêu").evidence]
    assert docs.count(target.doc_id) == 6 and docs.count(other.doc_id) == 3


def test_document_with_the_number_outranks_documents_citing_it():
    base = DocRecord(doc_id="", source_file="", source_path="", title="", total_pages=2)
    target = replace(base, doc_id="t" * 32, source_file="368 - 1910 UBND-ĐT.pdf", so_hieu="1910/UBND-ĐT")
    citing = replace(base, doc_id="c" * 32, source_file="3995 SXD.pdf", so_hieu="3995/SXD-ĐSĐT")
    chunks = [replace(_chunk(citing.doc_id, 0), codes=["1910/UBND-ĐT"]), replace(_chunk(target.doc_id, 0), codes=["1910/UBND-ĐT"])]
    ranked = [(c.chunk_id, r) for r, c in enumerate(chunks, start=1)]       # văn bản trích dẫn đứng trên ở nhánh tìm kiếm
    engine = SearchEngine(_MemoryStore([target, citing], chunks), [StaticBranch("vector", ranked)], ABBREV)
    assert engine.search("1910/UBND-ĐT").evidence[0].doc.doc_id == target.doc_id


def test_document_with_the_number_is_added_when_search_branches_miss_it():
    base = DocRecord(doc_id="", source_file="", source_path="", title="", total_pages=2)
    target = replace(base, doc_id="t" * 32, source_file="181 BC-TTĐHGT.pdf", so_hieu="181/BC-TTĐHGT", loai_vb="BC")
    near = replace(base, doc_id="n" * 32, source_file="183 BC-TTĐHGT.pdf", so_hieu="183/BC-TTĐHGT", loai_vb="BC")
    chunks = [_chunk(near.doc_id, 0), _chunk(target.doc_id, 0)]
    engine = SearchEngine(_MemoryStore([target, near], chunks), [StaticBranch("vector", [(chunks[0].chunk_id, 1)])], ABBREV)
    assert engine.search("181/BC-TTĐHGT").evidence[0].doc.doc_id == target.doc_id
    from src.store.base import Filters
    found = engine.search("181/BC-TTĐHGT", Filters(doc_ids=[near.doc_id])).evidence
    assert target.doc_id not in [ev.doc.doc_id for ev in found], "văn bản chèn thêm vẫn phải qua bộ lọc (quyền)"
