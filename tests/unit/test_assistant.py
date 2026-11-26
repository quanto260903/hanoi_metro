import pytest

from src.agent.assistant import DocumentAssistant, DocumentNotFound, Source, cited_sources, unsupported_facts
from src.config import AgentConfig
from src.retrieval.engine import SearchEngine
from src.text.abbrev import AbbreviationDictionary

from conftest import FakeLLM, ROOT

ABBREV = AbbreviationDictionary.load(ROOT / "data" / "dictionaries" / "synonym_dict.json")


def _assistant(store, reply="Trả lời [1].", **cfg):
    llm = FakeLLM(reply)
    engine = SearchEngine(store, [store], ABBREV)
    return DocumentAssistant(engine, store, llm, AgentConfig(**cfg)), llm


def test_ask_numbers_context_and_lists_only_cited_sources(indexed_store):
    assistant, llm = _assistant(indexed_store, reply="Cisco Switch N3K93180 bị hỏng [1].")
    answer = assistant.ask("Thiết bị nào ở DC Duy Tân bị hỏng?")
    system, user = llm.calls[0]
    assert "[1] Biên bản thiết bị DC ở Duy Tân.pdf, trang 1–2" in user
    assert "Q.NAP" in user and "Bình thường" in user
    assert [s.source_file for s in answer.sources] == ["Biên bản thiết bị DC ở Duy Tân.pdf"]
    assert answer.text.endswith("- [1] Biên bản thiết bị DC ở Duy Tân.pdf, trang 1–2")


def test_listing_question_answers_from_store_without_llm(indexed_store):
    assistant, llm = _assistant(indexed_store)
    answer = assistant.ask("liệt kê tất cả biên bản năm 2026")
    assert llm.calls == []
    assert "| 1 | — | 09/10/2026 | BB |" in answer.text and "Tổng cộng 1 văn bản" in answer.text


def test_uncited_answer_keeps_only_overlapping_sources():
    sources = [
        Source(1, "a.pdf", "a", "a:0", "a.pdf", "1", body="Cisco Switch N3K93180 bị hỏng tại tủ rack DC2"),
        Source(2, "b.pdf", "b", "b:0", "b.pdf", "1", body="Kế hoạch khảo sát AFC tại ga Lê Đức Thọ"),
    ]
    chosen = cited_sources("Thiết bị Cisco Switch N3K93180 bị hỏng tại tủ rack.", sources)
    assert [s.n for s in chosen] == [1]


def test_unsupported_facts_flags_numbers_not_in_sources():
    context = "Tờ trình số 485/TT-KHDA ngày 24/09/2026, lập 05 bản."
    body = "Theo tờ trình 485/TT-KHDA ngày 24/09/2026 [1], thời hạn đến 15/12/2026, kinh phí 350 triệu."
    assert unsupported_facts(body, context) == ["15/12/2026", "350"]


def test_summarize_by_document_number_adds_metadata_header(indexed_store):
    assistant, llm = _assistant(indexed_store, reply="Phòng VT&QLPT triển khai khảo sát AFC.")
    answer = assistant.summarize("916/CV-VT&QLPT", style="ngan")
    assert len(llm.calls) == 1 and "3–5 câu" in llm.calls[0][1]
    assert "- **Số hiệu:** 916/CV-VT&QLPT" in answer.text
    assert answer.text.endswith("**Nguồn:** 916/CV-VT&QLPT – 916.250926CV-VT&QLPT. Kế hoạch khảo sát hệ thống AFC.pdf, trang 1")


def test_summarize_long_document_uses_map_reduce(indexed_store):
    assistant, llm = _assistant(indexed_store, reply="- ý chính", context_budget_tokens=200, summary_batch_tokens=150)
    answer = assistant.summarize("biên bản kiểm đếm thiết bị Duy Tân")
    assert len(llm.calls) >= 2
    assert any("Văn bản dài" in n for n in answer.notes)


def test_resolve_unknown_document_raises(indexed_store):
    assistant, _ = _assistant(indexed_store)
    with pytest.raises(DocumentNotFound):
        assistant.resolve_document("f" * 32)


def test_draft_report_wraps_admin_form_and_appends_references(indexed_store):
    reply = "## I. TÌNH HÌNH\nCisco Switch bị hỏng [1].\n## III. ĐỀ XUẤT\n[cần bổ sung: đề xuất của đơn vị]"
    assistant, llm = _assistant(indexed_store, reply=reply)
    answer = assistant.draft_report("Kết quả kiểm đếm thiết bị DC Duy Tân", "to_trinh",
                                    doc_refs=["biên bản kiểm đếm Duy Tân"], recipient="Tổng Giám đốc", unit="Phòng VT&QLPT")
    text = answer.text
    assert "**CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**" in text and "Số: …/TTr-" in text
    assert "Kính gửi: Tổng Giám đốc" in text and "<b>TỜ TRÌNH</b>" in text
    assert text.index("**Nơi nhận:**") < text.index("**Tài liệu tham khảo (nội bộ):**")
    assert "dàn ý" in llm.calls[0][1].lower() and "4. Kiến nghị" in llm.calls[0][1]


def test_draft_report_rejects_unknown_type(indexed_store):
    assistant, _ = _assistant(indexed_store)
    with pytest.raises(ValueError):
        assistant.draft_report("x", "nghi_quyet")


def test_passages_of_one_document_share_one_source_number():
    from src.agent.assistant import build_context, format_pages
    from src.indexing.models import Chunk, DocRecord
    from src.retrieval.engine import Evidence
    doc = DocRecord("d" * 32, "QĐ 1977.pdf", "x", "QĐ", 3, so_hieu="1977/QĐ-SXD")
    other = DocRecord("e" * 32, "CV 916.pdf", "y", "CV", 2)

    def ev(d, seq, p_from, p_to):
        c = Chunk(f"{d.doc_id}-{seq}", d.doc_id, f"{d.doc_id}-p{seq}", seq, p_from, p_to, "", "", f"Mục {seq}", f"Mục {seq}")
        return Evidence(0, 1.0, c, d)

    context, sources = build_context([ev(doc, 4, 3, 3), ev(other, 1, 1, 1), ev(doc, 5, 3, 3), ev(doc, 2, 2, 2)], 6000)
    assert [s.n for s in sources] == [1, 2]
    assert sources[0].pages == "2–3" and sources[0].citation.endswith("QĐ 1977.pdf, trang 2–3")
    assert sources[0].anchor == "3"  # liên kết mở trang của đoạn xếp hạng cao nhất, không phải trang nhỏ nhất
    assert context.count("[1] ") == 3 and context.count("[2] ") == 1
    assert format_pages([5, 2, 3, 9]) == "2–3, 5, 9"
