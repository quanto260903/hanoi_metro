from types import SimpleNamespace

import pytest

from src.indexing.indexer import Blocklist, BlockedDocumentError, BlocklistNotConfigured
from src.ingestion.cloud_vlm import BudgetExceeded, CloudVLMParser, VLMLedger, VLMUnavailable, is_looping, render_strips

from conftest import ROOT

MD5 = "a" * 32


class FakePage:
    rect = SimpleNamespace(x0=0, y0=0, x1=595, y1=842, width=595, height=842)

    def __init__(self):
        self.clips = []

    def get_pixmap(self, dpi, alpha, clip=None):
        self.clips.append(clip)
        return SimpleNamespace(tobytes=lambda fmt: b"\x89PNG fake")


class FakeClient:
    """API VLM là dịch vụ bên ngoài: ghi lại yêu cầu và trả phản hồi kèm usage như Gemini."""

    def __init__(self, text="```markdown\n# Trang\n| a | b |\n```", tin=1800, tout=500, errors=()):
        self.requests = []
        self.text, self.tin, self.tout = text, tin, tout
        self.errors = list(errors)
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        return self._stream()

    def _stream(self):
        # stream=True kèm include_usage: các mảnh nội dung, rồi một sự kiện cuối chỉ có usage
        for i in range(0, len(self.text), 5):
            if self.closed:
                return
            delta = SimpleNamespace(content=self.text[i:i + 5])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None)
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=self.tin, completion_tokens=self.tout))


class RateLimited(Exception):
    status_code = 429


def _parser(tmp_path, client=None, blocked=(), budget=1.0):
    p = CloudVLMParser(client=client or FakeClient(), blocklist=Blocklist(set(blocked)), ledger=VLMLedger(tmp_path / "ledger.db"))
    p.budget, p.price_in, p.price_out, p.reasoning_effort, p.strips = budget, 0.30, 2.50, "none", 3
    p._sleep = lambda seconds: None
    return p


def test_records_real_usage_cost_and_strips_fences(tmp_path):
    client = FakeClient()
    parser = _parser(tmp_path, client)
    res = parser.parse_page(FakePage(), 3, MD5)
    assert res.markdown == "# Trang\n| a | b |"
    assert res.cost_usd == pytest.approx(1800 / 1e6 * 0.30 + 500 / 1e6 * 2.50)
    assert client.requests[0]["reasoning_effort"] == "none"
    images = [c for c in client.requests[0]["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(images) == 3 and images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parser.ledger.stats()["pages"] == 1


def test_each_page_is_sent_only_once(tmp_path):
    client = FakeClient()
    parser = _parser(tmp_path, client)
    parser.parse_page(FakePage(), 1, MD5)
    again = parser.parse_page(FakePage(), 1, MD5)
    assert again.cached and len(client.requests) == 1


def test_blocked_document_is_never_sent(tmp_path):
    client = FakeClient()
    with pytest.raises(BlockedDocumentError):
        _parser(tmp_path, client, blocked=[MD5]).parse_page(FakePage(), 1, MD5)
    assert client.requests == []


def test_unconfigured_blocklist_never_sends(tmp_path):
    client = FakeClient()
    parser = _parser(tmp_path, client)
    parser.blocklist = Blocklist(set(), configured=False, problem="chưa đặt BLOCKLIST_DIR")
    with pytest.raises(BlocklistNotConfigured):
        parser.parse_page(FakePage(), 1, MD5)
    assert client.requests == []


def test_stops_before_exceeding_budget(tmp_path):
    client = FakeClient(tin=100_000, tout=50_000)          # ~0.155 USD mỗi trang
    parser = _parser(tmp_path, client, budget=0.2)
    parser.parse_page(FakePage(), 1, MD5)
    with pytest.raises(BudgetExceeded):
        parser.parse_page(FakePage(), 2, MD5)
    assert len(client.requests) == 1


def test_pipeline_skips_blocked_pdf_entirely(tmp_path, monkeypatch):
    from src.ingestion.pipeline import IngestionPipeline, StateManager
    pdf = next((ROOT / "data" / "raw_pdfs").glob("Biên bản*.pdf"))
    pipe = IngestionPipeline(raw_dir=pdf.parent, processed_dir=tmp_path, db_path=tmp_path / "state.db")
    monkeypatch.setattr(pipe, "blocklist", Blocklist({StateManager.compute_md5(pdf)}))
    assert pipe.process_file(pdf) == "BLOCKED"
    assert list(tmp_path.glob("*.md")) == []


def test_strips_overlap_and_cover_whole_page():
    page = FakePage()
    render_strips(page, 3, dpi=200)
    clips = page.clips
    assert clips[0].y0 == 0 and clips[-1].y1 == 842
    assert all(a.y1 > b.y0 for a, b in zip(clips, clips[1:])), "các dải chồng lấn nhau"


def test_retries_rate_limit_then_succeeds(tmp_path):
    client = FakeClient(errors=[RateLimited(), RateLimited()])
    res = _parser(tmp_path, client).parse_page(FakePage(), 1, MD5)
    assert res.markdown.startswith("# Trang") and len(client.requests) == 3


def test_persistent_rate_limit_fails_without_recording(tmp_path):
    client = FakeClient(errors=[RateLimited()] * 10)
    parser = _parser(tmp_path, client)
    with pytest.raises(VLMUnavailable):
        parser.parse_page(FakePage(), 1, MD5)
    assert parser.ledger.stats()["pages"] == 0, "trang lỗi không được lưu như đã xong"


def test_loop_is_cut_and_collapsed(tmp_path):
    looping = "| 1 | a |\n" + "| | |\n" * 400
    client = FakeClient(text=looping)
    res = _parser(tmp_path, client).parse_page(FakePage(), 1, MD5)
    assert res.markdown.count("| | |") == 1
    assert res.out_tokens < 400, "ngắt sớm: không tính tiền phần lặp"


def test_is_looping_detects_repeated_tail_only():
    assert is_looping("x\n" + "| | |\n" * 12)
    assert not is_looping("\n".join(f"| {i} | thiết bị |" for i in range(30)))


def test_pilot_allocation_keeps_small_strata_and_total():
    from src.ingestion.pilot import allocate
    sizes = {"table": 36947, "complex_digital_table": 13162, "blurry": 367, "skewed": 123, "rare": 3}
    alloc = allocate(sizes, 200, 6)
    assert sum(alloc.values()) == 200
    assert alloc["blurry"] >= 6 and alloc["skewed"] >= 6 and alloc["rare"] == 3
    assert alloc["table"] > alloc["complex_digital_table"] > alloc["blurry"]


def test_inline_phrase_loop_detected_and_collapsed():
    from src.ingestion.cloud_vlm import collapse_inline_repeats
    loop = "| 1 | Dự án tại phường Tây Tựu, " + "quận Bắc Từ Liêm, " * 40
    assert is_looping(loop)
    assert is_looping("| a |" + " " * 300)
    assert collapse_inline_repeats(loop).count("quận Bắc Từ Liêm") == 1
    normal = "| 24.120 | Washer | 5.115.409-0 | 3 |\n| 24.121 | Countersunk screw M5X20 | 7.305.050-0 | 3 |"
    assert not is_looping(normal) and collapse_inline_repeats(normal) == normal


def test_cached_page_reports_recorded_tokens(tmp_path):
    parser = _parser(tmp_path, FakeClient())
    parser.parse_page(FakePage(), 1, MD5)
    again = parser.parse_page(FakePage(), 1, MD5)
    assert again.cached and (again.in_tokens, again.out_tokens) == (1800, 500)


def test_table_markup_is_never_treated_as_loop():
    from src.ingestion.cloud_vlm import collapse_inline_repeats
    wide = "| TT | A | B | C | D | E | F | G | H | I |\n|---|---|---|---|---|---|---|---|---|---|"
    assert collapse_inline_repeats(wide) == wide and not is_looping(wide)
    html = "<tr><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>"
    assert collapse_inline_repeats(html) == html and not is_looping(html)
    row = "| 4 | Server | BT | BT | BT | BT | BT | BT |"
    assert collapse_inline_repeats(row) == row and not is_looping(row)
    runaway = "| 1 | Dự án tại " + "quận Bắc Từ Liêm, " * 60 + " |"
    assert collapse_inline_repeats(runaway).count("Bắc Từ Liêm") == 1
    assert is_looping("| 1 | Dự án tại " + "quận Bắc Từ Liêm, " * 30)


def test_raw_output_kept_for_reprocessing(tmp_path):
    from src.ingestion.cloud_vlm import clean_vlm_output
    parser = _parser(tmp_path, FakeClient(text="| a | b |\n|---|---|"))
    parser.parse_page(FakePage(), 1, MD5)
    assert parser.ledger.reprocess(clean_vlm_output) == 0
    assert parser.ledger.reprocess(lambda raw: raw.upper()) == 1


def test_long_sentence_loop_inside_table_cell():
    from src.ingestion.cloud_vlm import collapse_inline_repeats
    sentence = "<br>- Dự án đã được UBND Thành phố phê duyệt chủ trương đầu tư tại Quyết định số 821/QĐ-UBND ngày 07/10/2014. "
    line = "|  | Cải tạo đê tả Bùi | 1 | " + sentence * 30
    assert is_looping(line)
    assert collapse_inline_repeats(line).count("821/QĐ-UBND") == 1
