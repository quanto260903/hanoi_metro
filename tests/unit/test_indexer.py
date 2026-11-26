import shutil

import pytest

from src.indexing.indexer import Blocklist, BlocklistNotConfigured, DocumentIndexer
from src.indexing.structured import StructuredChunker

from conftest import BB_DUY_TAN, CV_916

BB_ID = "e2311c730de842ef3c56314f98f4250e"


class RecordingVector:
    name = "vector"

    def __init__(self, fail=False):
        self.fail = fail
        self.indexed = {}

    def index(self, doc, chunks, phong=()):
        if self.fail:
            raise RuntimeError("Embedding API (bge-m3) lỗi")
        self.indexed[doc.doc_id] = [c.chunk_id for c in chunks]

    def delete(self, doc_id):
        self.indexed.pop(doc_id, None)


def _write_lists(tmp_path, restricted=(), manual=(), allow=()):
    d = tmp_path / "ket_qua"
    d.mkdir()
    (d / "restricted_md5.txt").write_text("# tự sinh\n" + "\n".join(restricted), encoding="utf-8")
    (d / "blocklist_md5.txt").write_text("\n".join(f"{m}  # ghi chú" for m in manual), encoding="utf-8")
    (d / "allowlist_md5.txt").write_text("\n".join(allow), encoding="utf-8")
    return d


def test_missing_blocklist_refuses_every_outbound_send(tmp_path):
    no_restricted = tmp_path / "chua_chay_restricted"
    no_restricted.mkdir()
    (no_restricted / "blocklist_md5.txt").write_text("c" * 32, encoding="utf-8")
    for bl in (Blocklist.from_dir(None), Blocklist.from_dir(no_restricted)):
        assert not bl.configured
        with pytest.raises(BlocklistNotConfigured):
            bl.assert_allowed(["a" * 32])
    Blocklist.from_dir(_write_lists(tmp_path)).assert_allowed(["a" * 32])   # đủ danh sách, không bị chặn: cho gửi


def test_blocklist_union_minus_allowlist(tmp_path):
    d = _write_lists(tmp_path, restricted=["a" * 32, "b" * 32], manual=["C" * 32], allow=["b" * 32])
    bl = Blocklist.from_dir(d)
    assert bl.is_blocked("a" * 32) and bl.is_blocked("c" * 32)
    assert not bl.is_blocked("b" * 32)


def test_blocked_document_never_reaches_either_branch_and_is_purged(tmp_path, store):
    vector = RecordingVector()
    open_indexer = DocumentIndexer(StructuredChunker(), store, vector, Blocklist(set()))
    open_indexer.index_paths([BB_DUY_TAN])
    assert store.get_document(BB_ID) and BB_ID in vector.indexed

    blocked = DocumentIndexer(StructuredChunker(), store, vector, Blocklist.from_dir(_write_lists(tmp_path, manual=[BB_ID])))
    report = blocked.index_paths([BB_DUY_TAN, CV_916])
    assert report.blocked == ["Biên bản thiết bị DC ở Duy Tân.pdf"]
    assert store.get_document(BB_ID) is None and BB_ID not in vector.indexed
    assert len(report.indexed) == 1


def test_unchanged_documents_are_skipped_and_edits_reindexed(tmp_path, store):
    md = tmp_path / "bb.md"
    shutil.copy(BB_DUY_TAN, md)
    indexer = DocumentIndexer(StructuredChunker(), store, RecordingVector(), Blocklist(set()))
    assert len(indexer.index_paths([md]).indexed) == 1
    assert len(indexer.index_paths([md]).unchanged) == 1
    md.write_text(md.read_text(encoding="utf-8") + "\n\nBổ sung: thiết bị Q.NAP đã được thay thế.", encoding="utf-8")
    assert len(indexer.index_paths([md]).indexed) == 1
    assert any("Q.NAP đã được thay thế" in c.text for c in store.get_document_chunks(BB_ID))


def test_vector_failure_leaves_both_branches_without_the_document(store):
    indexer = DocumentIndexer(StructuredChunker(), store, RecordingVector(fail=True), Blocklist(set()))
    report = indexer.index_paths([BB_DUY_TAN])
    assert "Biên bản thiết bị DC ở Duy Tân.pdf" in report.failed
    assert store.get_document(BB_ID) is None
