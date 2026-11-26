"""Bổ sung metadata (src.indexing.enrich): luật, cảnh báo hiệu lực / đọc kém, bước AI. Dữ liệu giả hoàn toàn."""

import csv
import hashlib
import json
import sqlite3

import pytest

from src.access.labels import COLUMNS, merge_ai_suggestions
from src.access.policy import Department
from src.agent.assistant import DocumentAssistant
from src.config import AgentConfig
from src.indexing.enrich import (
    AI_VERSION, RULES_VERSION, AIBudgetExceeded, AIConfig, AILedger, LedgerQuality, RoutingTable, compute_status, detect_huong_vb,
    document_warnings, extract_effective_date, extract_relations, parse_ai_output, run_ai, run_rules,
)
from src.indexing.indexer import BlockedDocumentError, Blocklist, BlocklistNotConfigured
from src.indexing.models import Chunk, DocRecord, ParentSection
from src.retrieval.engine import SearchEngine
from src.retrieval.query import parse_query
from src.store.base import NGUON_AI, NGUON_DUYET, NGUON_LUAT, DocField, DocRelation, Filters

from conftest import FakeLLM
from test_gateway import ABBREV


def _doc(n: str, **kw) -> DocRecord:
    base = dict(doc_id=hashlib.md5(n.encode()).hexdigest(), source_file=f"{n}.pdf", source_path=f"Kho/{n}.pdf", title=f"Văn bản {n}", total_pages=2)
    base.update(kw)
    return DocRecord(**base)


def _put(store, doc: DocRecord, pages):
    """Ghi văn bản giả vào kho: mỗi trang một chunk."""
    parents, chunks = [], []
    for i, text in enumerate(pages, start=1):
        pid = f"{doc.doc_id}:p{i}"
        parents.append(ParentSection(pid, doc.doc_id, "", text, i, i))
        chunks.append(Chunk(f"{doc.doc_id}:{i}", doc.doc_id, pid, i, i, i, "", doc.so_hieu, text, f"{doc.so_hieu} {text}"))
    store.upsert_document(doc, parents, chunks)
    return doc


class FakeVector:
    def __init__(self):
        self.payloads = {}

    def ensure_enrich_indexes(self):
        pass

    def set_doc_payload(self, doc_id, payload):
        self.payloads[doc_id] = payload


EMPTY_ROUTING, EMPTY_LEDGER = RoutingTable({}), LedgerQuality({}, 16384)


# ---------------------------------------------------------------------- hướng văn bản
@pytest.mark.parametrize("doc, expected", [
    (_doc("a", source_path="2025/CV ĐI/12.pdf"), "di"),
    (_doc("a", source_path="2025\\Công văn đến\\12.pdf"), "den"),
    (_doc("a", so_hieu="12/CV-ĐSHN", co_quan="Công ty TNHH MTV Đường sắt Hà Nội", kinh_gui=["Sở Xây dựng Hà Nội"]), "di"),
    (_doc("a", so_hieu="13/TB-ĐSHN", co_quan="Công ty TNHH MTV Đường sắt Hà Nội", kinh_gui=["Phòng KHDA", "Xí nghiệp 2A"]), "noi_bo"),
    (_doc("a", so_hieu="14/CV-ĐSHN", co_quan="Công ty TNHH MTV Đường sắt Hà Nội"), "khong_ro"),
    (_doc("a", so_hieu="3278/QĐ-UBND", co_quan="UBND thành phố Hà Nội"), "den"),
    (_doc("a"), "khong_ro"),
])
def test_huong_vb_from_folder_issuer_and_recipients(doc, expected):
    f = detect_huong_vb(doc)
    assert (f.truong, f.gia_tri, f.nguon) == ("huong_vb", expected, NGUON_LUAT)


# ---------------------------------------------------------------------- quan hệ văn bản
PAGE1 = """CÔNG TY TNHH MTV ĐƯỜNG SẮT HÀ NỘI
Số: 50/QĐ-ĐSHN
Căn cứ Quyết định số 3278/QĐ-UBND ngày 20/6/2023 của UBND Thành phố;
Căn cứ Nghị định số 15/2025/NĐ-CP thay thế Nghị định số 06/2021/NĐ-CP;
Phúc đáp công văn số 916/CV-VT&QLPT ngày 03/4/2025 của Phòng VTQLPT;"""
PAGE2 = """Điều 2. Quyết định này có hiệu lực kể từ ngày 15/03/2026 và thay thế Quyết định số 45/QĐ-ĐSHN ngày 01/02/2024.
Bãi bỏ Quyết định số 30/QĐ-ĐSHN ngày 01/01/2023.
Bãi bỏ Điều 5 của Quyết định số 31/QĐ-ĐSHN ngày 02/01/2023.
Gửi kèm theo Biên bản số 7/BB-KHDA. Quyết định này dẫn chiếu số hiệu của chính nó 50/QĐ-ĐSHN."""


def test_relations_by_kind_with_page_and_partial_repeal_is_amendment():
    doc = _doc("b", so_hieu="50/QĐ-ĐSHN", cites=["3278/QĐ-UBND", "99/TB-ATCL"])
    rel = {(r.loai, r.so_hieu_dich): r.trang for r in extract_relations(doc, {1: PAGE1, 2: PAGE2})}
    assert rel[("can_cu", "3278/QĐ-UBND")] == 1 and rel[("can_cu", "15/2025/NĐ-CP")] == 1
    assert ("can_cu", "06/2021/NĐ-CP") in rel and ("thay_the", "06/2021/NĐ-CP") not in rel, \
        "'thay thế' nằm trong dòng Căn cứ là chuyện của văn bản được viện dẫn, không phải văn bản này"
    assert rel[("tra_loi", "916/CV-VT&QLPT")] == 1
    assert rel[("thay_the", "45/QĐ-ĐSHN")] == 2
    assert rel[("bai_bo", "30/QĐ-ĐSHN")] == 2
    assert ("sua_doi", "31/QĐ-ĐSHN") in rel and ("bai_bo", "31/QĐ-ĐSHN") not in rel, "bãi bỏ một điều = sửa đổi"
    assert ("kem_theo", "7/BB-KHDA") in rel
    assert ("lien_quan", "99/TB-ATCL") in rel and ("lien_quan", "3278/QĐ-UBND") not in rel
    assert all("50/QĐ-ĐSHN" != code for _, code in rel), "không tự trỏ về chính mình"


def test_effective_date_numeric_words_and_signing_day():
    doc = _doc("c", ngay_ban_hanh="2026-03-04")
    f = extract_effective_date(doc, {1: "x", 2: PAGE2})
    assert (f.gia_tri, f.trang) == ("2026-03-15", 2)
    assert extract_effective_date(doc, {1: "có hiệu lực từ ngày 5 tháng 3 năm 2026"}).gia_tri == "2026-03-05"
    signed = extract_effective_date(doc, {3: "Quyết định có hiệu lực thi hành kể từ ngày ký."})
    assert (signed.gia_tri, signed.trang, signed.ghi_chu) == ("2026-03-04", 3, "kể từ ngày ký")
    assert extract_effective_date(_doc("c"), {1: "có hiệu lực kể từ ngày ký"}) is None, "không biết ngày ký thì bỏ"
    assert extract_effective_date(doc, {1: "có hiệu lực từ ngày 31/02/2026"}) is None, "ngày không tồn tại"


def test_status_is_reverse_of_relations_and_ignores_older_sources():
    old = _doc("d", so_hieu="45/QĐ-ĐSHN", ngay_ban_hanh="2024-02-01")
    new = _doc("e", so_hieu="50/QĐ-ĐSHN", ngay_ban_hanh="2026-03-04")
    older = _doc("f", so_hieu="1/QĐ-ĐSHN", ngay_ban_hanh="2020-01-01")
    rel = DocRelation("thay_the", "45/QĐ-ĐSHN", old.doc_id, 2, NGUON_LUAT, new.doc_id)
    status, by = compute_status(old, [rel], {new.doc_id: new}, False)
    assert (status.gia_tri, by.gia_tri) == ("bi_thay_the", "50/QĐ-ĐSHN")
    bad = DocRelation("bai_bo", "45/QĐ-ĐSHN", old.doc_id, 1, NGUON_LUAT, older.doc_id)
    assert compute_status(old, [bad], {older.doc_id: older}, True)[0].gia_tri == "con_hieu_luc"
    assert compute_status(old, [], {}, False)[0].gia_tri == "khong_ro"
    repeal = DocRelation("bai_bo", "45/QĐ-ĐSHN", "", 2, NGUON_LUAT, new.doc_id)
    assert compute_status(old, [rel, repeal], {new.doc_id: new}, False)[0].gia_tri == "bai_bo"


# ---------------------------------------------------------------------- chất lượng bóc tách
def test_scan_ratio_and_readability_from_routing_and_ledger(tmp_path):
    md5 = _doc("q").doc_id
    (tmp_path / "pages_routed.csv").write_text(
        "md5,page,route\n" + "".join(f"{md5},{p},{r}\n" for p, r in
                                     enumerate(["DIRECT_TEXT", "VLM", "VLM", "VLM", "STANDARD_OCR", "BLANK", "VLM"])),
        encoding="utf-8")
    routing = RoutingTable.load(tmp_path)
    assert routing.scan_ratio(md5) == pytest.approx(5 / 6) and routing.scan_ratio("z" * 32) is None
    db = tmp_path / "vlm_ledger.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE vlm_pages (md5, page, model, prompt_version, markdown, in_tokens, out_tokens, cost_usd, created_at, raw)")
    rows = [(2, "nội dung tốt", 300), (3, "", 0), (4, "a [không rõ] b [không rõ] c [không rõ]", 400)]
    c.executemany("INSERT INTO vlm_pages VALUES (?,?,?,?,?,?,?,?,?,?)",
                  [(md5, p, "m", "v", md, 1, out, 0.0, "2026-01-01", None) for p, md, out in rows]
                  + [(md5, 2, "m", "old", "", 1, 0, 0.0, "2025-01-01", None)])     # bản cũ hơn bị bản mới thay
    c.commit()
    c.close()
    ledger = LedgerQuality.load(db, max_tokens=1000)
    assert ledger.bad_pages(md5) == [3, 4]
    ledger.pages[md5][2] = ("dài", 990)                  # chạm trần token
    assert ledger.bad_pages(md5) == [2, 3, 4]
    from src.indexing.enrich import quality_fields
    fields = {f.truong: f for f in quality_fields(_doc("q", total_pages=7), routing, ledger)}
    assert fields["ty_le_trang_scan"].gia_tri == "0.83"
    assert fields["trang_can_scan_lai"].gia_tri == "2,3,4"
    assert fields["ty_le_doc_duoc"].gia_tri == "0.43" and "1 trang chưa nhận dạng" in fields["ty_le_doc_duoc"].ghi_chu


# ---------------------------------------------------------------------- chạy luật trên kho, bộ lọc, nguồn ưu tiên
@pytest.fixture
def corpus(store):
    old = _put(store, _doc("d", so_hieu="45/QĐ-ĐSHN", ngay_ban_hanh="2024-02-01", loai_vb="QĐ", nam=2024,
                           source_path="QĐ/45.pdf", trich_yeu="Quy chế cũ"), ["Quy chế cũ về định mức vật tư.", "Hết."])
    new = _put(store, _doc("e", so_hieu="50/QĐ-ĐSHN", ngay_ban_hanh="2026-03-04", loai_vb="QĐ", nam=2026,
                           source_path="QĐ/50.pdf", trich_yeu="Quy chế mới"), [PAGE1, PAGE2])
    den = _put(store, _doc("g", so_hieu="77/CV-SXD", loai_vb="CV", nam=2025, co_quan="Sở Xây dựng",
                           source_path="2025/CV ĐẾN/77.pdf", trich_yeu="Về định mức"), ["Nội dung công văn đến.", "Hết."])
    return store, old, new, den


def test_run_rules_fills_tables_payload_and_filters(corpus):
    store, old, new, den = corpus
    vector = FakeVector()
    stats = run_rules(store, [old, new, den], EMPTY_ROUTING, EMPTY_LEDGER, vector)
    assert stats.counts["huong_den"] == 1
    rel = {(r.loai, r.so_hieu_dich): r.doc_id_dich for r in store.doc_relations([new.doc_id])}
    assert rel[("thay_the", "45/QĐ-ĐSHN")] == old.doc_id, "số hiệu đích có trong kho → điền doc_id_dich"
    assert rel[("can_cu", "3278/QĐ-UBND")] == ""
    f = store.doc_fields([old.doc_id, new.doc_id])
    assert f[old.doc_id]["trang_thai_hieu_luc"].gia_tri == "bi_thay_the"
    assert f[old.doc_id]["thay_the_boi"].gia_tri == "50/QĐ-ĐSHN"
    assert f[new.doc_id]["trang_thai_hieu_luc"].gia_tri == "con_hieu_luc"
    assert f[new.doc_id]["hieu_luc_tu"].phien_ban_bo_trich == RULES_VERSION
    assert vector.payloads[old.doc_id]["trang_thai_hieu_luc"] == "bi_thay_the"
    assert vector.payloads[den.doc_id]["huong_vb"] == "den"

    docs, total = store.list_documents(Filters(huong_vb=["den"]))
    assert [d.doc_id for d in docs] == [den.doc_id] and total == 1
    q = parse_query("liệt kê công văn đến năm 2025", ABBREV)
    assert q.huong_vb == ["den"]
    engine = SearchEngine(store, [store], ABBREV)
    assert [d.doc_id for d in engine.list_documents(q)[0]] == [den.doc_id]
    assert parse_query("văn bản đi về định mức", ABBREV).huong_vb == ["di"]
    assert parse_query("đi kiểm tra nhà ga", ABBREV).huong_vb == []


def test_reviewed_value_is_never_overwritten_by_rules_or_ai(store):
    d = _doc("h").doc_id
    store.set_doc_fields(d, [DocField("huong_vb", "di", NGUON_DUYET, nguoi_duyet="van_thu")])
    assert store.set_doc_fields(d, [DocField("huong_vb", "den", NGUON_LUAT)]) == []
    store.set_doc_fields(d, [DocField("phong_chu_tri", "KHDA", NGUON_LUAT)])
    store.set_doc_fields(d, [DocField("phong_chu_tri", "TCKT", NGUON_AI)])
    f = store.doc_fields([d])[d]
    assert (f["huong_vb"].gia_tri, f["huong_vb"].nguon) == ("di", NGUON_DUYET)
    assert f["phong_chu_tri"].gia_tri == "KHDA"


# ---------------------------------------------------------------------- cảnh báo trong câu trả lời
def test_assistant_warns_about_superseded_and_poorly_read_documents(corpus):
    store, old, new, den = corpus
    run_rules(store, [old, new, den], EMPTY_ROUTING, EMPTY_LEDGER)
    store.set_doc_fields(den.doc_id, [DocField("ty_le_doc_duoc", "0.40", NGUON_LUAT),
                                      DocField("trang_can_scan_lai", "2", NGUON_LUAT)])
    assistant = DocumentAssistant(SearchEngine(store, [store], ABBREV), store, FakeLLM("Tóm tắt."), AgentConfig())
    notes = assistant.summarize(old.doc_id).notes
    assert any("45/QĐ-ĐSHN đã bị thay thế bởi 50/QĐ-ĐSHN" in n for n in notes)
    notes = assistant.summarize(den.doc_id).notes
    assert any("đọc kém" in n and "nên xem bản gốc" in n and "trang cần scan lại: 2" in n for n in notes)
    assert not any("thay thế" in n or "đọc kém" in n for n in assistant.summarize(new.doc_id).notes)
    assert document_warnings(object(), [old.doc_id]) == [], "kho không hỗ trợ metadata bổ sung: không lỗi"


# ---------------------------------------------------------------------- bước AI
DEPTS = {c: Department(c, f"Phòng {c}", mo_ta=f"chức năng {c}") for c in ("KHDA", "TCKT", "VTQLPT")}
AI_REPLY = json.dumps({
    "phong_chu_tri": [{"ma": "KHDA", "ly_do": "trích yếu về kế hoạch | dự án"}, {"ma": "IT", "ly_do": "bịa"}],
    "phong_phoi_hop": [{"ma": "tckt", "ly_do": "kinh phí"}],
    "nhiem_vu": [
        {"phong": "KHDA", "nguoi": "Trưởng phòng", "noi_dung": "Lập kế hoạch bảo trì", "han": "2026-04-30", "loai_han": "ngay", "trang": 2},
        {"phong": "XYZ", "nguoi": "", "noi_dung": "Báo cáo hằng tháng", "han": "cuối tháng", "loai_han": "dinh_ky", "trang": 99},
        {"noi_dung": ""},
    ],
}, ensure_ascii=False)


def test_parse_ai_output_keeps_only_known_units_valid_pages_and_dates():
    res = parse_ai_output("Đây là kết quả:\n" + AI_REPLY, list(DEPTS), [1, 2])
    assert [c for c, _ in res.chu_tri] == ["KHDA"] and "|" not in res.chu_tri[0][1]
    assert [c for c, _ in res.phoi_hop] == ["TCKT"]
    assert [(t.phong, t.han, t.trang) for t in res.tasks] == [("KHDA", "2026-04-30", 2), ("", "", None)]
    assert parse_ai_output("không phải JSON", list(DEPTS), [1]).tasks == []


def _ai(store, docs, llm, blocklist, tmp_path, budget=1.0):
    return run_ai(store, docs, llm, blocklist.assert_allowed, DEPTS, AILedger(tmp_path / "ai.sqlite"),
                  AIConfig(budget, 0.10, 0.50))


def test_ai_step_one_call_per_document_skips_blocked_and_never_touches_access(corpus, tmp_path):
    store, old, new, den = corpus
    llm = FakeLLM(AI_REPLY)
    stats, results = _ai(store, [old, new, den], llm, Blocklist({old.doc_id}), tmp_path)
    assert stats.counts["ai_bi_chan"] == 1 and stats.counts["ai_xong"] == 2 and len(llm.calls) == 2
    assert all("Quy chế cũ" not in user for _, user in llm.calls), "văn bản bị chặn không được gửi"
    assert all("--- Trang 1 ---" in user and "KHDA: Phòng KHDA" in user for _, user in llm.calls)
    f = store.doc_fields([new.doc_id])[new.doc_id]
    assert (f["phong_chu_tri"].gia_tri, f["phong_chu_tri"].nguon, f["phong_chu_tri"].phien_ban_bo_trich) == ("KHDA", NGUON_AI, AI_VERSION)
    assert store.access_labels([new.doc_id])[new.doc_id] == [], "gợi ý AI không thành nhãn phân quyền"
    tasks = store.doc_tasks([new.doc_id])
    assert tasks[0].noi_dung == "Lập kế hoạch bảo trì" and tasks[0].nguon == NGUON_AI

    again, _ = _ai(store, [old, new, den], llm, Blocklist({old.doc_id}), tmp_path)
    assert again.counts["ai_da_co"] == 2 and len(llm.calls) == 2, "mỗi văn bản chỉ gọi một lần"

    # gợi ý ghi vào file duyệt: cột phong_goi_y; phong_duyet người duyệt đã điền giữ nguyên
    path = tmp_path / "nhan.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow({"doc_id": new.doc_id, "phong_goi_y": "ATCL", "ly_do": "mã đơn vị", "phong_duyet": "ATCL", "ghi_chu": "đã xem"})
        w.writerow({"doc_id": den.doc_id})
    assert merge_ai_suggestions(path, results) == 2
    rows = {r["doc_id"]: r for r in csv.DictReader(path.open(encoding="utf-8-sig"))}
    assert rows[new.doc_id]["phong_goi_y"] == "ATCL;KHDA;TCKT"
    assert rows[new.doc_id]["phong_duyet"] == "ATCL" and rows[new.doc_id]["ghi_chu"] == "đã xem"
    assert "AI (cần kiểm tra): KHDA chủ trì" in rows[new.doc_id]["ly_do"] and "mã đơn vị" in rows[new.doc_id]["ly_do"]
    assert merge_ai_suggestions(tmp_path / "khong_co.csv", results) is None


def test_ai_step_stops_without_blocklist_or_over_budget(corpus, tmp_path):
    store, old, new, den = corpus
    llm = FakeLLM(AI_REPLY)
    with pytest.raises(BlocklistNotConfigured):
        _ai(store, [new], llm, Blocklist(set(), configured=False), tmp_path)
    assert llm.calls == [], "thiếu danh sách chặn: không gửi gì"
    with pytest.raises(AIBudgetExceeded):
        _ai(store, [new, den], llm, Blocklist(set()), tmp_path / "b", budget=0.0000001)
    assert llm.calls == []
    assert issubclass(BlocklistNotConfigured, BlockedDocumentError)


def test_review_sheet_lists_ai_documents_with_rule_results(corpus, tmp_path):
    from src.indexing.enrich import export_review
    store, old, new, den = corpus
    run_rules(store, [old, new, den], EMPTY_ROUTING, EMPTY_LEDGER)
    _ai(store, [new], FakeLLM(AI_REPLY), Blocklist(set()), tmp_path)
    out = tmp_path / "kiem_tra.csv"
    assert export_review(store, out) == 1
    row = next(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert row["doc_id"] == new.doc_id and row["ai_phong_chu_tri"] == "KHDA" and row["huong_vb"]
    assert "thay_the 45/QĐ-ĐSHN (tr.2)" in row["quan_he_thay_the_bai_bo"]
    assert "Lập kế hoạch bảo trì – hạn 2026-04-30 (tr.2)" in row["ai_nhiem_vu"]
    assert row["dung_phong_chu_tri"] == "" and row["phong_dung"] == "", "cột để người kiểm điền"


def test_only_one_lead_department_rest_become_coordinators():
    reply = json.dumps({"phong_chu_tri": [{"ma": "KHDA", "ly_do": "a"}, {"ma": "TCKT", "ly_do": "b"}, {"ma": "VTQLPT", "ly_do": "c"}],
                        "phong_phoi_hop": [{"ma": "TCKT", "ly_do": "d"}]})
    res = parse_ai_output(reply, list(DEPTS), [1])
    assert [c for c, _ in res.chu_tri] == ["KHDA"]
    assert [c for c, _ in res.phoi_hop] == ["VTQLPT", "TCKT"], "không trùng lặp phòng phối hợp"


def test_compatible_version_not_called_again_and_reparse_without_api(corpus, tmp_path):
    from src.indexing.enrich import AI_PREVIOUS, reparse_ai
    store, old, new, den = corpus
    ledger = AILedger(tmp_path / "ai.sqlite")
    multi = json.dumps({"phong_chu_tri": [{"ma": "KHDA", "ly_do": "a"}, {"ma": "TCKT", "ly_do": "b"}]})
    from src.indexing.enrich import AI_VERSION
    ledger.record(old.doc_id, AI_VERSION, 10, 10, 0.0, AI_REPLY)            # đã có phiên bản hiện tại → không gọi lại
    llm = FakeLLM(AI_REPLY)
    stats, _ = run_ai(store, [old], llm, Blocklist(set()).assert_allowed, DEPTS, ledger, AIConfig(1.0, 0.1, 0.5))
    assert stats.counts["ai_da_co"] == 1 and llm.calls == []
    ledger.record(new.doc_id, AI_PREVIOUS[0], 10, 10, 0.0, multi)          # chỉ có phiên bản cũ → tinh-lai-ai vẫn đọc được
    store.set_doc_fields(new.doc_id, [DocField("phong_chu_tri", "KHDA;TCKT", NGUON_AI)])   # kết quả cũ chưa chuẩn hóa
    results = reparse_ai(store, ledger, DEPTS)
    f = store.doc_fields([new.doc_id])[new.doc_id]
    assert (f["phong_chu_tri"].gia_tri, f["phong_phoi_hop"].gia_tri) == ("KHDA", "TCKT")
    assert f["phong_chu_tri"].phien_ban_bo_trich == AI_PREVIOUS[0], "giữ phiên bản đã sinh ra phản hồi"
    assert [c for c, _ in results[new.doc_id].chu_tri] == ["KHDA"] and llm.calls == []


def test_review_sheet_with_reviewer_input_is_not_overwritten(corpus, tmp_path):
    from src.indexing.enrich import export_review
    store, old, new, den = corpus
    _ai(store, [new], FakeLLM(AI_REPLY), Blocklist(set()), tmp_path)
    out = tmp_path / "kiem_tra.csv"
    assert export_review(store, out) == 1
    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    rows[0]["dung_phong_chu_tri"] = "Đ"
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(FileExistsError):
        export_review(store, out)
    assert next(csv.DictReader(out.open(encoding="utf-8-sig")))["dung_phong_chu_tri"] == "Đ"


@pytest.mark.parametrize("doc", [
    _doc("t", co_quan="HANOÏ L3", trich_yeu="CIDM000338038_EN"),                       # tiêu đề trang, không số hiệu
    _doc("t", co_quan="LIST OF REVISIONS"),
    _doc("t", so_hieu="UJV-OPM-TLI-SIG-L00-40403-E-6A-P200-2", co_quan="UJV-MTS-00697-22-B-SIG"),
    _doc("t", so_hieu="12/XYZ", co_quan="||**CVS**||**PRODUCT**|"),                     # ô bảng bóc nhầm
])
def test_technical_manuals_with_bogus_issuer_are_not_incoming(doc):
    assert detect_huong_vb(doc).gia_tri == "khong_ro"


def test_replacement_mentioned_in_a_draft_is_not_a_relation():
    plain = _doc("u", so_hieu="5/UBND-HTĐTMT")
    text = ("Sau khi nghiên cứu bản dự thảo Quyết định thay thế Quyết định số 21/2023/QĐ-UBND ngày 19/9/2023;\n"
            "Quyết định này bãi bỏ Quyết định số 9/2020/QĐ-UBND.")
    rel = {(r.loai, r.so_hieu_dich) for r in extract_relations(plain, {1: text})}
    assert ("thay_the", "21/2023/QĐ-UBND") not in rel, "'dự thảo' đứng trước trong cùng mệnh đề"
    assert ("bai_bo", "9/2020/QĐ-UBND") in rel
    draft = _doc("v", so_hieu="6/UBND-HTĐTMT", trich_yeu="tham gia ý kiến vào dự thảo Quyết định thay thế Quyết định số 21/2023/QĐ-UBND")
    rel = {(r.loai, r.so_hieu_dich) for r in extract_relations(draft, {1: "Thực hiện văn bản về việc lấy ý kiến hồ sơ Quyết định "
                                                                        "thay thế Quyết định số 21/2023/QĐ-UBND ngày 19/9/2023."})}
    assert not any(loai in ("thay_the", "bai_bo", "sua_doi") for loai, _ in rel), "văn bản góp ý dự thảo"


def test_only_same_issuing_agency_can_replace():
    from src.indexing.enrich import issuing_agencies
    assert issuing_agencies("2068/QĐ-UBND") == {"UBND"} and issuing_agencies("45/2024/QĐ-ĐSHN") == {"DSHN"}
    assert issuing_agencies("916/CV-VT&QLPT") == {"VTQLPT"} and issuing_agencies("CIDM000338122") == set()
    ubnd = _doc("w", so_hieu="2068/QĐ-UBND", ngay_ban_hanh="2025-04-16")
    center = _doc("x", so_hieu="457/TTĐHGT-KTDV", ngay_ban_hanh="2026-04-29")
    same = _doc("y", so_hieu="3000/QĐ-UBND", ngay_ban_hanh="2026-01-01")
    rel = lambda src: DocRelation("thay_the", "2068/QĐ-UBND", ubnd.doc_id, 1, NGUON_LUAT, src.doc_id)
    assert compute_status(ubnd, [rel(center)], {center.doc_id: center}, False)[0].gia_tri == "khong_ro"
    assert compute_status(ubnd, [rel(same)], {same.doc_id: same}, False)[0].gia_tri == "bi_thay_the"


def test_parse_ai_view_codes_multi_department():
    from src.indexing.enrich import parse_ai_output
    raw = ('{"phong_chu_tri":[{"ma":"VTQLPT","ly_do":"a"}],"phong_phoi_hop":[{"ma":"XN2","ly_do":"b"}],'
           '"phong_can_xem":[{"ma":"XN3","ly_do":"c"},{"ma":"KHONGCO","ly_do":"x"}],"chung":true,"nhiem_vu":[]}')
    res = parse_ai_output(raw, ["VTQLPT", "XN2", "XN3"], [1])
    assert res.view_codes() == ["VTQLPT", "XN2", "XN3", "CHUNG"]
    res = parse_ai_output('{"phong_chu_tri":[],"chung":"false"}', ["VTQLPT"], [1])
    assert res.view_codes() == [] and res.chung is False


def test_ai_input_carries_direction_hint(corpus):
    from src.indexing.enrich import build_ai_input
    store, old, new, den = corpus
    text = build_ai_input(den, {1: "nội dung"}, DEPTS, huong="den")
    assert "Hướng: văn bản ĐẾN" in text
    assert "Hướng:" not in build_ai_input(den, {1: "nội dung"}, DEPTS, huong="khong_ro")
