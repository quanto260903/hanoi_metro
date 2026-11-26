import re
import json

import pytest
from starlette.testclient import TestClient

from src.agent.assistant import DocumentAssistant
from src.config import AgentConfig
from src.gateway.server import Turn, create_app, search_query_for
from src.indexing.indexer import Blocklist
from src.retrieval.engine import SearchEngine
from src.text.abbrev import AbbreviationDictionary

from conftest import FakeLLM, ROOT

ABBREV = AbbreviationDictionary.load(ROOT / "data" / "dictionaries" / "synonym_dict.json")
KEY = "test-key"
BB_ID = "e2311c730de842ef3c56314f98f4250e"


def _client(store, reply="Cisco Switch N3K93180 bị hỏng [1].", external=False, blocked=(), blocklist=None):
    llm = FakeLLM(reply, external=external)
    guard = (blocklist or Blocklist(set(blocked))).assert_allowed
    assistant = DocumentAssistant(SearchEngine(store, [store], ABBREV), store, llm, AgentConfig(), outbound_guard=guard)
    return TestClient(create_app(lambda: assistant, KEY)), llm


def _ask(client, text, model="hanoi-metro-hoi-dap", stream=False, history=()):
    messages = [{"role": r, "content": c} for r, c in history] + [{"role": "user", "content": text}]
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {KEY}"},
                       json={"model": model, "messages": messages, "stream": stream})


def _content(resp) -> str:
    return resp.json()["choices"][0]["message"]["content"]


def test_rejects_missing_or_wrong_key(indexed_store):
    client, _ = _client(indexed_store)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer sai"}).status_code == 401
    models = client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"}).json()["data"]
    assert {m["id"] for m in models} == {"hanoi-metro-hoi-dap", "hanoi-metro-tom-tat", "hanoi-metro-soan-thao"}


def test_question_uses_exactly_one_llm_call_and_appends_sources(indexed_store):
    client, llm = _client(indexed_store)
    text = _content(_ask(client, "Thiết bị nào tại DC Duy Tân bị hỏng?"))
    assert len(llm.calls) == 1
    assert text.startswith("Cisco Switch N3K93180 bị hỏng [1].")
    assert "- [1] Biên bản thiết bị DC ở Duy Tân.pdf, trang 1–2" in text


def test_streaming_follows_openai_sse_format(indexed_store):
    client, _ = _client(indexed_store)
    resp = _ask(client, "Thiết bị nào tại DC Duy Tân bị hỏng?", stream=True)
    events = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    joined = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "Cisco Switch" in joined and "**Nguồn:**" in joined


def test_commands_route_to_summary_and_listing(indexed_store):
    client, llm = _client(indexed_store, reply="Tóm tắt nội dung.")
    text = _content(_ask(client, "/tomtat 916/CV-VT&QLPT ngắn"))
    assert text.startswith("### Tóm tắt: CV 916/CV-VT&QLPT") and "3–5 câu" in llm.calls[0][1]
    listing = _content(_ask(client, "/lietke biên bản năm 2026"))
    assert "| 1 | — | 09/10/2026 | BB |" in listing and len(llm.calls) == 1


def test_report_model_detects_type_and_cited_document(indexed_store):
    client, llm = _client(indexed_store, reply="## 1. Căn cứ\nNội dung [1].")
    text = _content(_ask(client, "Tờ trình xin kinh phí khảo sát AFC theo 916/CV-VT&QLPT", model="hanoi-metro-soan-thao"))
    assert "<b>TỜ TRÌNH</b>" in text and "4. Kiến nghị" in llm.calls[0][1]
    assert "916.250926CV-VT&QLPT" in text


def test_external_llm_refuses_blocked_documents(indexed_store):
    client, llm = _client(indexed_store, external=True, blocked=[BB_ID])
    text = _content(_ask(client, "Thiết bị nào tại DC Duy Tân bị hỏng?"))
    assert llm.calls == []
    assert "dấu chỉ độ mật" in text


def test_external_llm_refused_when_blocklist_not_configured(indexed_store):
    unconfigured = Blocklist(set(), configured=False, problem="chưa đặt BLOCKLIST_DIR")
    client, llm = _client(indexed_store, external=True, blocklist=unconfigured)
    text = _content(_ask(client, "Thiết bị nào tại DC Duy Tân bị hỏng?"))
    assert llm.calls == []
    assert "BLOCKLIST_DIR" in text
    # LLM chạy nội bộ (không ra ngoài) vẫn trả lời bình thường
    client, llm = _client(indexed_store, external=False, blocklist=unconfigured)
    assert "[1]" in _content(_ask(client, "Thiết bị nào tại DC Duy Tân bị hỏng?")) and len(llm.calls) == 1


def test_short_follow_up_reuses_previous_question_for_search():
    turns = [Turn("user", "Thiết bị nào tại DC Duy Tân bị hỏng?"), Turn("assistant", "Cisco Switch [1]."), Turn("user", "còn thiết bị nào khác?")]
    assert search_query_for(turns).startswith("Thiết bị nào tại DC Duy Tân bị hỏng?")
    assert search_query_for([Turn("user", "Tờ trình 485/TT-KHDA")]) == "Tờ trình 485/TT-KHDA"


def test_help_message_without_llm(indexed_store):
    client, llm = _client(indexed_store)
    assert "/tomtat" in _content(_ask(client, "/help")) and llm.calls == []


@pytest.mark.parametrize("bad", [None, []])
def test_empty_messages_return_help(indexed_store, bad):
    client, _ = _client(indexed_store)
    resp = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {KEY}"}, json={"messages": bad})
    assert "Trợ lý tài liệu Hanoi Metro" in _content(resp)


def test_rate_limit_is_reported_clearly(indexed_store):
    from src.agent.llm import LLMError

    class QuotaLLM(FakeLLM):
        def stream(self, *args, **kwargs):
            raise LLMError("llm_rate_limited", "429 free tier, key=secret")
            yield ""

    llm = QuotaLLM()
    assistant = DocumentAssistant(SearchEngine(indexed_store, [indexed_store], ABBREV), indexed_store, llm, AgentConfig())
    client = TestClient(create_app(lambda: assistant, KEY))
    text = _content(_ask(client, "Thiết bị nào tại DC Duy Tân?"))
    assert "hạn mức gọi API" in text and "secret" not in text


def test_token_limit_parameter_matches_provider():
    from src.agent.llm import token_limit_kwargs
    assert token_limit_kwargs("https://api.openai.com/v1", 100) == {"max_completion_tokens": 100}
    assert token_limit_kwargs("https://generativelanguage.googleapis.com/v1beta/openai/", 100) == {"max_tokens": 100}
    assert token_limit_kwargs("http://127.0.0.1:11434/v1", 100) == {"max_tokens": 100}


# ---------------------------------------------------------------- mở tài liệu gốc từ trích dẫn
from dataclasses import replace as _replace
from urllib.parse import urlsplit

from src.gateway.documents import DocumentLinks, first_page


def _linked_client(store, tmp_path, blocked=()):
    pdf = tmp_path / "goc" / "Biên bản thiết bị DC ở Duy Tân.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4 test")
    doc = store.get_documents([BB_ID])[BB_ID]
    with store._conn() as c:  # chỉ đổi đường dẫn file gốc của văn bản mẫu sang file tạm
        import json as _json
        c.execute("UPDATE documents SET meta_json=? WHERE doc_id=?",
                  (_json.dumps(_replace(doc, source_path=str(pdf)).to_dict(), ensure_ascii=False), BB_ID))
    links = DocumentLinks.create("bi-mat", "http://127.0.0.1:8766", [tmp_path / "goc"], Blocklist(set(blocked)).is_blocked)
    assistant = DocumentAssistant(SearchEngine(store, [store], ABBREV), store, FakeLLM("Cisco Switch N3K93180 bị hỏng [1]."),
                                  AgentConfig(), outbound_guard=Blocklist(set()).assert_allowed, linker=links.url)
    return TestClient(create_app(lambda: assistant, KEY, links)), links


def test_sources_link_to_original_file_at_cited_page(indexed_store, tmp_path):
    client, links = _linked_client(indexed_store, tmp_path)
    text = _content(_ask(client, "Thiết bị nào tại DC Duy Tân bị hỏng?"))
    url = re.search(r"\]\((http://127\.0\.0\.1:8766/tai-lieu/[^)]+)\)", text).group(1)
    assert f"/tai-lieu/{BB_ID}/" in url and "#page=" in url and " " not in url
    parts = urlsplit(url)
    resp = client.get(f"{parts.path}?{parts.query}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/pdf")
    assert resp.headers["content-disposition"].startswith("inline")
    assert resp.content == b"%PDF-1.4 test"


def test_document_link_requires_valid_signature(indexed_store, tmp_path):
    client, links = _linked_client(indexed_store, tmp_path)
    assert client.get(f"/tai-lieu/{BB_ID}/x.pdf?k=sai").status_code == 403
    assert client.get(f"/tai-lieu/{BB_ID}/x.pdf").status_code == 403
    other = "0" * 32  # chữ ký đúng nhưng văn bản không có trong chỉ mục
    assert client.get(f"/tai-lieu/{other}/x.pdf?k={links.sign(other)}").status_code == 404
    assert client.get(f"/tai-lieu/..%2F..%2Fsecret/x.pdf?k=x").status_code in (403, 404)


def test_blocked_document_cannot_be_opened_even_with_old_link(indexed_store, tmp_path):
    client, links = _linked_client(indexed_store, tmp_path, blocked=(BB_ID,))
    assert client.get(f"/tai-lieu/{BB_ID}/x.pdf?k={links.sign(BB_ID)}").status_code == 403


def test_file_outside_allowed_roots_is_not_served(indexed_store, tmp_path):
    client, _ = _linked_client(indexed_store, tmp_path)
    links = DocumentLinks.create("bi-mat", "http://h", [tmp_path / "khac"], lambda _: False)
    status, path, _ = links.resolve(indexed_store, BB_ID, links.sign(BB_ID))
    assert status == 404 and path is None


def test_links_disabled_without_secret_keep_plain_citations(indexed_store):
    assert DocumentLinks.create("", "http://h", [], lambda _: False) is None
    client, _ = _client(indexed_store)
    text = _content(_ask(client, "Thiết bị nào tại DC Duy Tân bị hỏng?"))
    assert "/tai-lieu/" not in text and "Duy Tân" in text


def test_first_page_parsing():
    assert first_page("3–5") == 3 and first_page("12") == 12 and first_page("") == 1
