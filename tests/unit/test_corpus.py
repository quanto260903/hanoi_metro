import json
from types import SimpleNamespace

import fitz
import pytest

from src.indexing.document import load_markdown
from src.indexing.indexer import Blocklist, BlocklistNotConfigured
from src.indexing.structured import StructuredChunker
from src.ingestion.cloud_vlm import CloudVLMParser, VLMLedger
from src.ingestion.corpus import Corpus, CorpusStore, direct_text_markdown, load_plan

A, B, C = "a" * 32, "b" * 32, "c" * 32


def _pdf(path, pages):
    doc = fitz.open()
    for draw in pages:
        page = doc.new_page(width=595, height=842)
        draw(page)
    doc.save(path)


def _text(lines):
    def draw(page):
        for i, line in enumerate(lines):
            page.insert_text((72, 100 + 20 * i), line, fontname="helv", fontsize=11)
    return draw


def _table(page):
    page.insert_text((72, 80), "Danh muc thiet bi", fontname="helv", fontsize=11)
    xs, ys = [72, 200, 330], [100, 125, 150, 175]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for r, row in enumerate([("STT", "Ten"), ("1", "Switch"), ("2", "Server")]):
        for c, cell in enumerate(row):
            page.insert_text((xs[c] + 5, ys[r] + 17), cell, fontname="helv", fontsize=10)
    page.insert_text((72, 220), "Ket thuc bang", fontname="helv", fontsize=11)


def test_direct_text_keeps_order_and_does_not_duplicate_table(tmp_path):
    _pdf(tmp_path / "t.pdf", [_table])
    md = direct_text_markdown(fitz.open(tmp_path / "t.pdf")[0])
    assert md.index("Danh muc thiet bi") < md.index("|") < md.index("Ket thuc bang")
    assert md.count("Switch") == 1, "chữ trong bảng chỉ xuất hiện trong bảng Markdown"


class FakeBatchClient:
    """Batch API là dịch vụ bên ngoài: giả lập files/batches và trả kết quả như OpenAI."""

    def __init__(self, reply="# Trang scan\\nNội dung"):
        self.uploaded = []
        self.reply = reply
        self.files = SimpleNamespace(create=self._upload, content=self._content)
        self.batches = SimpleNamespace(create=self._create, retrieve=self._retrieve)
        self.state = "in_progress"

    def _upload(self, file, purpose):
        self.uploaded.append([json.loads(line) for line in file.read().decode().splitlines()])
        return SimpleNamespace(id="file-in")

    def _create(self, **kw):
        return SimpleNamespace(id="batch-1")

    def _retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, status=self.state, output_file_id="file-out" if self.state == "completed" else None, error_file_id=None)

    def _content(self, file_id):
        lines = []
        for req in self.uploaded[-1]:
            ok = not req["custom_id"].startswith(B)          # trang của file B lỗi phía API
            lines.append(json.dumps({"custom_id": req["custom_id"], "response": {
                "status_code": 200 if ok else 500,
                "body": {"choices": [{"message": {"content": self.reply}}], "usage": {"prompt_tokens": 3000, "completion_tokens": 600}} if ok else {},
            }}))
        return SimpleNamespace(text="\n".join(lines))


def _setup(tmp_path, blocked=()):
    root = tmp_path / "kho"
    (root / "CV").mkdir(parents=True)
    _pdf(root / "CV" / "916 CV-VT&QLPT.pdf", [_text(["CONG TY TNHH MTV DUONG SAT HA NOI", "So: 916/CV-VT&QLPT"]), _text([]), _text(["trang scan"])])
    _pdf(root / "CV" / "khac.pdf", [_text(["trang 1"])])
    _pdf(root / "CV" / "mat.pdf", [_text(["tai lieu mat"])])
    res = tmp_path / "ket_qua"
    res.mkdir()
    (res / "ingest_manifest.csv").write_text(
        "path,md5,pages\nCV\\916 CV-VT&QLPT.pdf,%s,3\nCV\\khac.pdf,%s,1\nCV\\mat.pdf,%s,1\n" % (A, B, C), encoding="utf-8")
    (res / "pages_routed.csv").write_text(
        "path,md5,page,route,reasons\n"
        f"x,{A},0,DIRECT_TEXT,\nx,{A},1,BLANK,\nx,{A},2,VLM,table\nx,{B},0,STANDARD_OCR,clean_scan\nx,{C},0,VLM,table\n", encoding="utf-8")
    parser = CloudVLMParser(client=SimpleNamespace(), blocklist=Blocklist(set(blocked)), ledger=VLMLedger(tmp_path / "ledger.db"))
    parser.budget, parser.price_in, parser.price_out, parser.strips, parser.dpi = 1.0, 0.10, 0.50, 1, 40
    return Corpus(root, CorpusStore(tmp_path / "corpus.db"), parser, tmp_path / "md"), load_plan(res)


def test_batch_flow_assembles_complete_files_only(tmp_path, monkeypatch):
    corpus, plans = _setup(tmp_path, blocked=[C])
    monkeypatch.setattr("src.ingestion.corpus.os.remove", lambda p: None)
    assert corpus.run_local(plans)["pages"] == 2                       # DIRECT_TEXT + BLANK; OCR đi API
    pending = corpus.pending_api_pages(plans)
    assert {(pl.md5, p) for pl, p in pending} == {(A, 2), (B, 0)}       # tài liệu mật C không có

    client = FakeBatchClient()
    assert corpus.submit_batches(plans, client, batch_pages=10) == ["batch-1"]
    req = client.uploaded[0][0]
    assert req["url"] == "/v1/chat/completions" and req["body"]["messages"][1]["content"][1]["type"] == "image_url"
    assert corpus.pending_api_pages(plans) == [], "trang đang chờ Batch không bị gửi lại"

    assert corpus.collect_batches(client)["waiting"] == 1
    client.state = "completed"
    assert corpus.collect_batches(client) == {"done": 1, "failed": 1, "waiting": 0}
    assert corpus.parser.ledger.stats()["cost_usd"] > 0

    stats = corpus.assemble(plans)
    assert stats["assembled"] == 1 and stats["incomplete"] == 1        # A đủ trang; B lỗi trang Batch
    assert [(pl.md5, p) for pl, p in corpus.pending_api_pages(plans)] == [(B, 0)], "trang lỗi được gửi lại lần sau"

    src = load_markdown(tmp_path / "md" / f"{A}.md")
    assert src.doc_id == A and src.source_file == "916 CV-VT&QLPT.pdf" and [p.number for p in src.pages] == [1, 3]
    doc, _, chunks = StructuredChunker().chunk_document(src)
    assert doc.so_hieu == "916/CV-VT&QLPT" and chunks
    assert corpus.store.file_status()[C] == "blocked"


def test_batch_submit_respects_budget(tmp_path):
    corpus, plans = _setup(tmp_path)
    corpus.parser.budget = 0.0001
    try:
        corpus.submit_batches(plans, FakeBatchClient())
        raise AssertionError("phải dừng vì hạn mức")
    except Exception as e:
        assert type(e).__name__ == "BudgetExceeded"


def test_wrapped_lines_joined_but_articles_kept_apart():
    from src.ingestion.corpus import _join_wrapped_lines
    text = ("5. Nguồn kinh phí: Chi thường xuyên từ ngân sách\nThành phố.\n6. Thời gian thực hiện: Năm 2026.\n"
            "Điều 2. Tổ chức thực hiện\n1. Công ty TNHH MTV Đường sắt Hà Nội chịu trách nhiệm\ntoàn diện về tính chính xác")
    assert _join_wrapped_lines(text).splitlines() == [
        "5. Nguồn kinh phí: Chi thường xuyên từ ngân sách",
        "Thành phố.",
        "6. Thời gian thực hiện: Năm 2026.",
        "Điều 2. Tổ chức thực hiện",
        "1. Công ty TNHH MTV Đường sắt Hà Nội chịu trách nhiệm toàn diện về tính chính xác",
    ]


def test_api_steps_refuse_without_blocklist(tmp_path):
    corpus, plans = _setup(tmp_path)
    corpus.parser.blocklist = Blocklist(set(), configured=False, problem="chưa đặt BLOCKLIST_DIR")
    with pytest.raises(BlocklistNotConfigured):
        corpus.run_vlm_sync(plans)
    with pytest.raises(BlocklistNotConfigured):
        corpus.submit_batches(plans, client=None, work_dir=tmp_path / "batches")


def test_partial_assembly_indexes_local_text_then_completes_later(tmp_path):
    corpus, plans = _setup(tmp_path, blocked=[C])
    corpus.run_local(plans)
    stats = corpus.assemble(plans, partial=True, min_partial_chars=10)
    assert stats["partial"] == 1 and stats["assembled"] == 0           # A: có text gốc, trang 3 chờ API; B: chỉ có trang API
    md = (tmp_path / "md" / f"{A}.md").read_text(encoding="utf-8")
    assert "pending_api_pages: 1" in md and "916/CV-VT&QLPT" in md
    assert corpus.store.file_status()[A] == "partial"
    assert C not in corpus.store.file_status() or corpus.store.file_status()[C] == "blocked"

    corpus.parser.ledger.record(A, 3, corpus.parser.model, "Bảng thiết bị", 10, 10, 0.0001)
    assert corpus.assemble(plans)["assembled"] == 1                     # API xong → ghép lại đủ trang, ghi đè bản tạm
    assert "pending_api_pages" not in (tmp_path / "md" / f"{A}.md").read_text(encoding="utf-8")


def test_page_that_crashed_is_retried_without_tables_then_sent_to_api(tmp_path, monkeypatch):
    corpus, plans = _setup(tmp_path, blocked=[C])
    calls = []
    real = __import__("src.ingestion.corpus", fromlist=["x"]).direct_text_markdown
    monkeypatch.setattr("src.ingestion.corpus.direct_text_markdown",
                        lambda page, tables=True: calls.append(tables) or real(page, tables))

    corpus.store.mark_inflight(A, 0, "full")             # lần trước sập khi đang trích trang 1 của A (có dò bảng)
    corpus.run_local(plans)
    assert calls == [False], "trang nghi gây sập được trích lại, bỏ bước dò bảng"
    assert corpus.store.inflight() == {} and "916/CV-VT&QLPT" in corpus.store.local_pages(A)[0]

    corpus.store.save_local(A, 0, "DIRECT_TEXT", "")       # giả lập: chưa xong, lại sập ở chế độ an toàn
    with corpus.store._conn() as c:
        c.execute("DELETE FROM local_pages WHERE md5=? AND page=0", (A,))
    corpus.store.mark_inflight(A, 0, "safe")
    corpus.run_local(plans)
    assert corpus.store.failed_local_pages() == {(A, 0)}
    assert (A, 0) in {(pl.md5, p) for pl, p in corpus.pending_api_pages(plans)}, "trang trích hỏng chuyển sang API"

    corpus.parser.ledger.record(A, 1, corpus.parser.model, "So: 916/CV-VT&QLPT (API)", 10, 10, 0.0001)
    corpus.parser.ledger.record(A, 3, corpus.parser.model, "Bảng thiết bị", 10, 10, 0.0001)
    assert corpus.assemble(plans)["assembled"] == 1
    assert "(API)" in (tmp_path / "md" / f"{A}.md").read_text(encoding="utf-8")
