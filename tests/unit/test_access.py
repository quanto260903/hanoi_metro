"""Phân quyền theo phòng ban: quyền là tra bảng; LLM chỉ thấy văn bản đã qua bộ lọc."""

import re
from dataclasses import replace
from urllib.parse import urlsplit

import pytest
from starlette.testclient import TestClient

from src.access.labels import cmd_apply, parse_labels, suggest
from src.access.policy import CHUNG, PolicySource, load_policy, user_key
from src.access.registration import Registrations
from src.agent.assistant import DocumentAssistant, DocumentNotFound
from src.config import AgentConfig
from src.gateway.documents import DocumentLinks
from src.gateway.server import NOT_REGISTERED, AccessLog, create_app
from src.indexing.indexer import Blocklist
from src.indexing.models import DocRecord
from src.retrieval.engine import SearchEngine
from src.store.base import AccessScope, Filters

from conftest import FakeLLM, ROOT

from test_gateway import ABBREV, BB_ID, KEY, _content

CONFIG = ROOT / "config" / "access"


def _policy(tmp_path, users="", grants="", unknown="tu_choi"):
    d = tmp_path / "access"
    d.mkdir(exist_ok=True)
    (d / "phong_ban.yaml").write_text((CONFIG / "phong_ban.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    roles = (CONFIG / "vai_tro.yaml").read_text(encoding="utf-8").replace("nguoi_dung_chua_dang_ky: tu_choi",
                                                                         f"nguoi_dung_chua_dang_ky: {unknown}")
    roles = roles.replace("  nhan_vien:\n    ten: Nhân viên\n    xem: [phong_minh, CHUNG]\n    chuc_nang: [hoi_dap, tom_tat, soan_thao, liet_ke]",
                          "  nhan_vien:\n    ten: Nhân viên\n    xem: [phong_minh, CHUNG]\n    chuc_nang: [hoi_dap, tom_tat, liet_ke]")
    (d / "vai_tro.yaml").write_text(roles, encoding="utf-8")
    (d / "nguoi_dung.csv").write_text("email,ho_ten,phong,vai_tro,xem_then\n" if False else
                                      "email,ho_ten,phong,vai_tro,xem_them\n" + users, encoding="utf-8")
    (d / "cap_rieng.csv").write_text("email,doc_id,ghi_chu\n" + grants, encoding="utf-8")
    return load_policy(d, enabled=True)


USERS = ("nv.khda@vidu.vn,Nhân viên KHDA,KHDA,nhan_vien,\n"
         "nv.vt@vidu.vn,Nhân viên VT,VTQLPT,nhan_vien,\n"
         "tp.khda@vidu.vn,Trưởng phòng KHDA,KHDA,truong_phong,TCKT\n"
         "ld@vidu.vn,Lãnh đạo,BDH,lanh_dao,\n")


def test_policy_is_a_table_lookup(tmp_path):
    p = _policy(tmp_path, USERS, grants=f"nv.khda@vidu.vn,{'a' * 32},\n")
    assert p.problems == []
    nv = p.resolve("NV.KHDA@vidu.vn ")
    assert nv.scope == AccessScope(frozenset({"KHDA", CHUNG}), frozenset({"a" * 32}))
    assert "soan_thao" not in nv.features and "hoi_dap" in nv.features
    assert p.resolve("tp.khda@vidu.vn").scope.phong == {"KHDA", CHUNG, "TCKT"}
    assert p.resolve("ld@vidu.vn").scope is None, "lãnh đạo xem toàn bộ"
    stranger = p.resolve("la@vidu.vn")
    assert stranger.scope.empty and not stranger.features and not stranger.registered
    assert p.resolve("").scope.empty
    assert p.by_key(user_key("nv.vt@vidu.vn")).phong == "VTQLPT" and p.by_key("0" * 16) is None


def test_unknown_user_mode_and_bad_rows(tmp_path):
    p = _policy(tmp_path, "x@vidu.vn,X,KHONGCO,nhan_vien,\ny@vidu.vn,Y,KHDA,giam_doc,\n", unknown="chi_tai_lieu_chung")
    assert len(p.problems) == 2 and "x@vidu.vn" not in p.users and "y@vidu.vn" not in p.users
    assert p.resolve("x@vidu.vn").scope.phong == {CHUNG}


def test_disabled_policy_does_not_restrict(tmp_path):
    p = load_policy(CONFIG, enabled=False)
    assert p.resolve("ai@vidu.vn").scope is None


@pytest.fixture
def labelled(indexed_store):
    cv_id = next(d.doc_id for d in indexed_store.list_documents(Filters())[0] if d.doc_id != BB_ID)
    indexed_store.set_access_labels(cv_id, ["VTQLPT"], "duyet")
    indexed_store.set_access_labels(BB_ID, ["KHDA"], "duyet")
    return indexed_store, cv_id


def test_store_filters_by_department_labels(labelled):
    store, cv_id = labelled
    ids = lambda scope: {d.doc_id for d in store.list_documents(Filters(access=scope))[0]}
    assert ids(AccessScope(frozenset({"VTQLPT"}))) == {cv_id}
    assert ids(AccessScope(frozenset({"KHDA", CHUNG}))) == {BB_ID}
    assert ids(AccessScope(frozenset({"TCKT"}), frozenset({cv_id}))) == {cv_id}, "cấp riêng từng văn bản"
    assert ids(AccessScope()) == set(), "phạm vi rỗng: không thấy gì"
    store.set_access_labels(BB_ID, [], "duyet")
    assert ids(AccessScope(frozenset({"KHDA"}))) == set(), "gỡ nhãn → văn bản chưa gán nhãn bị chặn"


def test_direct_lookup_by_number_or_id_respects_scope(labelled):
    store, cv_id = labelled
    assistant = DocumentAssistant(SearchEngine(store, [store], ABBREV), store, FakeLLM("x"), AgentConfig(),
                                  outbound_guard=Blocklist(set()).assert_allowed)
    khda = Filters(access=AccessScope(frozenset({"KHDA", CHUNG})))
    with pytest.raises(DocumentNotFound):
        assistant.resolve_document(cv_id, khda)
    doc, notes = assistant.resolve_document("916/CV-VT&QLPT", khda)     # không thấy 916 → chọn văn bản KHDA gần nhất
    assert doc.doc_id != cv_id and not any("916" in n for n in notes)
    assert assistant.resolve_document(cv_id, Filters(access=AccessScope(frozenset({"VTQLPT"}))))[0].doc_id == cv_id
    hits = assistant.search("916/CV-VT&QLPT", khda).evidence
    assert cv_id not in {ev.doc.doc_id for ev in hits}, "tìm đích danh số hiệu cũng không lọt văn bản ngoài quyền"


def _gateway(store, tmp_path, policy, reply="Trả lời [1]."):
    pdf = tmp_path / "goc" / "a.pdf"
    pdf.parent.mkdir(exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4 test")
    links = DocumentLinks.create("bi-mat", "http://127.0.0.1:8766", [tmp_path / "goc"], lambda _: False)
    llm = FakeLLM(reply)
    assistant = DocumentAssistant(SearchEngine(store, [store], ABBREV), store, llm, AgentConfig(),
                                  outbound_guard=Blocklist(set()).assert_allowed, linker=links.url)
    log = AccessLog(tmp_path / "log.jsonl")
    return TestClient(create_app(lambda: assistant, KEY, links, policy, log)), llm, log


def _ask_as(client, email, text, model="hanoi-metro-hoi-dap"):
    headers = {"Authorization": f"Bearer {KEY}"}
    if email is not None:
        headers["x-user-email"] = email
    return client.post("/v1/chat/completions", headers=headers,
                       json={"model": model, "messages": [{"role": "user", "content": text}]})


def test_gateway_applies_user_scope_features_and_logs(labelled, tmp_path):
    store, cv_id = labelled
    client, llm, log = _gateway(store, tmp_path, _policy(tmp_path, USERS))

    assert _content(_ask_as(client, "la@vidu.vn", "Thiết bị nào tại DC Duy Tân bị hỏng?")) == NOT_REGISTERED, (
        "không bật khai báo (registrations=None): giữ thông báo cũ")
    assert _content(_ask_as(client, None, "Thiết bị nào tại DC Duy Tân bị hỏng?")) == NOT_REGISTERED
    assert _content(_ask_as(client, "{{LIBRECHAT_USER_EMAIL}}", "câu hỏi")) == NOT_REGISTERED
    assert llm.calls == [], "người chưa đăng ký không làm phát sinh lời gọi LLM"

    text = _content(_ask_as(client, "nv.vt@vidu.vn", "/tomtat Biên bản thiết bị DC Duy Tân"))
    assert BB_ID not in text and "Biên bản thiết bị DC" not in text
    assert llm.calls == [] or all("Duy Tân" not in c[1] for c in llm.calls), "văn bản KHDA không tới LLM"

    listing = _content(_ask_as(client, "nv.khda@vidu.vn", "/lietke"))
    assert "916/CV-VT&QLPT" not in listing

    denied = _content(_ask_as(client, "nv.khda@vidu.vn", "Tờ trình về DC Duy Tân", model="hanoi-metro-soan-thao"))
    assert "chưa được cấp quyền dùng chức năng **Soạn thảo văn bản**" in denied

    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"outcome": "tu_choi"' in ln for ln in lines) and any("nv.vt@vidu.vn" in ln for ln in lines)
    assert all("Duy Tân" not in ln for ln in lines), "nhật ký không ghi nội dung câu hỏi"


def test_document_link_bound_to_user_and_rechecked_on_open(labelled, tmp_path):
    store, cv_id = labelled
    doc = store.get_documents([BB_ID])[BB_ID]
    import json as _json
    with store._conn() as c:
        c.execute("UPDATE documents SET meta_json=? WHERE doc_id=?",
                  (_json.dumps(replace(doc, source_path=str(tmp_path / "goc" / "a.pdf")).to_dict(), ensure_ascii=False), BB_ID))
    policy = _policy(tmp_path, USERS)
    client, _, _ = _gateway(store, tmp_path, policy, reply="Cisco Switch hỏng [1].")
    text = _content(_ask_as(client, "nv.khda@vidu.vn", "Thiết bị nào tại DC Duy Tân bị hỏng?"))
    url = re.search(r"\]\((http://127\.0\.0\.1:8766/tai-lieu/[^)]+)\)", text).group(1)
    assert "u=" + user_key("nv.khda@vidu.vn") in url and "@" not in url, "liên kết không chứa email"
    parts = urlsplit(url)
    assert client.get(f"{parts.path}?{parts.query}").status_code == 200

    forged = parts.query.replace(user_key("nv.khda@vidu.vn"), user_key("nv.vt@vidu.vn"))
    assert client.get(f"{parts.path}?{forged}").status_code == 403, "đổi người trong liên kết → sai chữ ký"
    links = DocumentLinks.create("bi-mat", "http://127.0.0.1:8766", [tmp_path / "goc"], lambda _: False)
    other = urlsplit(links.url(BB_ID, "a.pdf", "1", user_key("nv.vt@vidu.vn"), 3600))
    assert client.get(f"{other.path}?{other.query}").status_code == 403, "người phòng khác không mở được dù chữ ký đúng"
    old = f"/tai-lieu/{BB_ID}/a.pdf?k={links.sign(BB_ID)}"
    assert client.get(old).status_code == 403, "khi bật phân quyền, liên kết không gắn người dùng bị từ chối"

    store.set_access_labels(BB_ID, ["TCKT"], "duyet")          # thu hồi: văn bản chuyển sang phòng khác
    assert client.get(f"{parts.path}?{parts.query}").status_code == 403, "liên kết cũ hết hiệu lực ngay khi đổi quyền"

    expired = urlsplit(links.url(BB_ID, "a.pdf", "1", user_key("ld@vidu.vn"), 3600))
    q = re.sub(r"e=\d+", "e=1", expired.query)
    assert client.get(f"{expired.path}?{q}").status_code == 403


def test_label_suggestions_use_fixed_rules_only(tmp_path):
    p = _policy(tmp_path)
    base = DocRecord(doc_id="d" * 32, source_file="x.pdf", source_path="D:/kho/x.pdf", title="", total_pages=1)
    own = replace(base, so_hieu="31/TB-ATCL", co_quan="CÔNG TY TNHH MTV ĐƯỜNG SẮT HÀ NỘI")
    assert suggest(own, p)[0] == ["ATCL"]
    assert suggest(replace(base, so_hieu="31/TB-ATCL", co_quan="SỞ XÂY DỰNG"), p)[0] == [], "văn bản bên ngoài không gợi ý"
    assert suggest(replace(base, don_vi="PHÒNG KẾ HOẠCH - DỰ ÁN", co_quan="CÔNG TY TNHH MTV ĐƯỜNG SẮT HÀ NỘI"), p)[0] == ["KHDA"]
    assert suggest(replace(base, so_hieu="1094/QĐ-ĐSHN", co_quan="CÔNG TY TNHH MTV ĐƯỜNG SẮT HÀ NỘI"), p)[0] == []


def test_parse_and_apply_reviewed_labels_all_or_nothing(labelled, tmp_path):
    store, cv_id = labelled
    p = _policy(tmp_path)
    assert parse_labels(" khda; chung ", p) == ["KHDA", CHUNG] and parse_labels("", p) is None and parse_labels("-", p) == []
    with pytest.raises(ValueError):
        parse_labels("KHDA;IT", p)

    class RT:
        def __init__(self, s):
            self.store, self.vector, self.payloads = s, self, []

        def ensure_access_index(self):
            pass

        def set_access_payload(self, doc_id, labels):
            self.payloads.append((doc_id, labels))

    rt = RT(store)
    bad = tmp_path / "bad.csv"
    bad.write_text(f"doc_id,phong_duyet\n{cv_id},CHUNG\n{BB_ID},PHONG_LA\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cmd_apply(rt, p, bad)
    assert store.access_labels([cv_id])[cv_id] == ["VTQLPT"], "có lỗi thì không áp dòng nào"

    good = tmp_path / "good.csv"
    good.write_text(f"doc_id,phong_duyet\n{cv_id},CHUNG;VTQLPT\n{BB_ID},\n", encoding="utf-8")
    assert cmd_apply(rt, p, good) == {"dong_hop_le": 1, "da_cap_nhat": 1}
    assert sorted(store.access_labels([cv_id])[cv_id]) == [CHUNG, "VTQLPT"] and rt.payloads == [(cv_id, [CHUNG, "VTQLPT"])]
    assert store.access_labels([BB_ID])[BB_ID] == ["KHDA"], "ô trống = giữ nguyên"


def _reg_gateway(store, tmp_path, users=USERS, unknown="tu_choi"):
    _policy(tmp_path, users, unknown=unknown)
    d = tmp_path / "access"
    source = PolicySource(d, enabled=True)
    regs = Registrations(tmp_path / "dang_ky.csv", d / "nguoi_dung.csv")
    llm = FakeLLM("Trả lời [1].")
    assistant = DocumentAssistant(SearchEngine(store, [store], ABBREV), store, llm, AgentConfig(),
                                  outbound_guard=Blocklist(set()).assert_allowed)
    app = create_app(lambda: assistant, KEY, None, source, AccessLog(tmp_path / "log.jsonl"), registrations=regs)
    return TestClient(app), llm, regs, d


def test_new_user_declares_department_and_department_head_approves(labelled, tmp_path):
    store, cv_id = labelled
    client, llm, regs, d = _reg_gateway(store, tmp_path)
    ask = lambda who, text: _content(_ask_as(client, who, text))

    first = ask("moi@vidu.vn", "Thiết bị nào tại DC Duy Tân bị hỏng?")
    assert "/phongban" in first and "XNBT" in first and "KHDA" in first, "người mới được hướng dẫn khai báo"
    assert "không có trong danh mục" in ask("moi@vidu.vn", "/phongban ABC Nguyễn Văn Mới")
    assert "họ và tên" in ask("moi@vidu.vn", "/phongban KHDA")
    assert "chờ duyệt" in ask("moi@vidu.vn", "/phongban khda =Nguyễn, Văn Mới")
    assert regs.get("moi@vidu.vn")["ho_ten"] == "Nguyễn Văn Mới", "bỏ dấu phẩy, ký tự công thức Excel"
    assert "chờ duyệt" in ask("moi@vidu.vn", "Thiết bị nào tại DC Duy Tân bị hỏng?"), "khai báo chưa cấp quyền"
    assert llm.calls == []

    # người không có quyền duyệt / trưởng phòng khác đơn vị / cấp vai trò cao hơn
    assert "không có quyền duyệt" in ask("nv.vt@vidu.vn", "/duyet moi@vidu.vn")
    regs.declare("khac@vidu.vn", "TCKT", "Người TCKT")
    assert "khac@vidu.vn" not in ask("tp.khda@vidu.vn", "/choduyet") and "moi@vidu.vn" in ask("tp.khda@vidu.vn", "/choduyet")
    assert "chỉ duyệt được khai báo của đơn vị mình" in ask("tp.khda@vidu.vn", "/duyet khac@vidu.vn")
    assert "chỉ cấp được vai trò nhan_vien" in ask("tp.khda@vidu.vn", "/duyet moi@vidu.vn lanh_dao")

    assert "Đã duyệt" in ask("tp.khda@vidu.vn", "/duyet moi@vidu.vn")
    users = (d / "nguoi_dung.csv").read_text(encoding="utf-8-sig")
    assert "moi@vidu.vn,Nguyễn Văn Mới,KHDA,nhan_vien," in users and "tp.khda@vidu.vn" in users
    assert "Kế hoạch" in ask("moi@vidu.vn", "/phongban"), "có hiệu lực ngay, không khởi động lại"
    listing = ask("moi@vidu.vn", "/lietke")
    assert "916/CV-VT&QLPT" not in listing, "chỉ thấy văn bản của đơn vị mình"

    assert "Không có khai báo đang chờ duyệt" in ask("tp.khda@vidu.vn", "/duyet moi@vidu.vn")
    log = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert '"outcome": "duyet_khai_bao"' in log and '"outcome": "chua_khai_bao"' in log


def test_admin_rejects_and_unknown_identity_cannot_declare(labelled, tmp_path):
    store, _ = labelled
    users = USERS + "qt@vidu.vn,Quản trị,VP,quan_tri,\n"
    client, llm, regs, d = _reg_gateway(store, tmp_path, users)
    ask = lambda who, text: _content(_ask_as(client, who, text))
    assert "Không xác định được tài khoản" in ask("{{LIBRECHAT_USER_EMAIL}}", "/phongban KHDA A")
    assert _content(_ask_as(client, None, "câu hỏi")) == NOT_REGISTERED
    ask("x@vidu.vn", "/phongban XN3 Trần X")
    assert "Đã từ chối" in ask("qt@vidu.vn", "/tuchoi x@vidu.vn không thuộc đơn vị")
    assert "không được duyệt" in ask("x@vidu.vn", "câu hỏi") and "x@vidu.vn" not in (d / "nguoi_dung.csv").read_text(encoding="utf-8-sig")
    ask("x@vidu.vn", "/phongban XN3 Trần X")
    assert "Đã duyệt" in ask("qt@vidu.vn", "/duyet x@vidu.vn pho_phong"), "quản trị cấp được vai trò khác"
    assert load_policy(d, True).users["x@vidu.vn"] == {"ho_ten": "Trần X", "phong": "XN3", "vai_tro": "pho_phong", "xem_them": []}
    assert llm.calls == []


def test_auto_approve_and_email_domain_limit(labelled, tmp_path):
    store, _ = labelled
    client, llm, regs, d = _reg_gateway(store, tmp_path)
    roles = (d / "vai_tro.yaml").read_text(encoding="utf-8")
    (d / "vai_tro.yaml").write_text(roles.replace("tu_dong_duyet: false", "tu_dong_duyet: true")
                                    .replace("ten_mien_email: []", "ten_mien_email: [vidu.vn]"), encoding="utf-8")
    ask = lambda who, text: _content(_ask_as(client, who, text))
    assert "tên miền" in ask("ngoai@gmail.com", "/phongban KHDA Người Ngoài")
    assert "Đã ghi nhận bạn thuộc" in ask("tudong@vidu.vn", "/phongban VTQLPT Người Tự Động")
    assert load_policy(d, True).users["tudong@vidu.vn"]["vai_tro"] == "nhan_vien"
    # người đã có quyền đổi đơn vị: vẫn phải chờ duyệt, quyền cũ giữ nguyên
    assert "chờ duyệt" in ask("nv.vt@vidu.vn", "/phongban KHDA Nhân viên VT")
    assert load_policy(d, True).users["nv.vt@vidu.vn"]["phong"] == "VTQLPT"


def test_candidate_list_scores_but_never_labels(tmp_path):
    from src.access.labels import candidate_score
    from src.store.base import DocField
    p = _policy(tmp_path)
    assert "AFC" in p.departments["VTQLPT"].keywords
    tech = DocRecord("c" * 32, "CIDM000338122_EN.pdf", "Kho/CIDM000338122_EN.pdf", "CIDM000338122_EN", 3)
    score, why = candidate_score(tech, "VTQLPT", p, "MAINTENANCE TASK ANALYSIS Replacement of the door, Depot", {})
    assert score >= 3 and any("trích yếu/tên file" in w for w in why)
    cv = DocRecord("d" * 32, "916.pdf", "Kho/916.pdf", "x", 1, so_hieu="916/CV-VT&QLPT",
                   co_quan="Công ty TNHH MTV Đường sắt Hà Nội", trich_yeu="Về việc kiểm đếm thiết bị")
    ai = {"phong_chu_tri": DocField("phong_chu_tri", "VTQLPT", "ai")}
    score, why = candidate_score(cv, "VTQLPT", p, "", ai)
    assert score == 6 and "AI: chủ trì" in why
    other = DocRecord("e" * 32, "luong.pdf", "Kho/luong.pdf", "Bảng lương tháng 5", 1, trich_yeu="Bảng lương tháng 5")
    assert candidate_score(other, "VTQLPT", p, "Tiền lương, phụ cấp", {})[0] == 0


def test_draft_split_merges_rules_and_ai_view_codes(tmp_path):
    from types import SimpleNamespace
    from src.access.labels import draft_split_codes
    from src.store.base import DocField
    policy = _policy(tmp_path)
    code = next(iter(policy.departments))
    doc = SimpleNamespace(so_hieu="", co_quan="", don_vi="", source_path="x/y.pdf", source_file="y.pdf", trich_yeu="", title="")
    f = DocField("phong_can_xem", f"{code};KHONGCO;CHUNG", "ai", 1, "", "ai-v7", "")
    codes, src = draft_split_codes(doc, policy, {"phong_can_xem": f})
    assert codes == [code, "CHUNG"] and "AI ai-v7" in src
    assert draft_split_codes(doc, policy, {})[0] == []


def test_draft_split_doc_codes_and_personal_ids(tmp_path):
    from types import SimpleNamespace
    from src.access.labels import draft_split_codes, doc_code_hits, has_personal_ids
    from src.access.policy import Department
    policy = _policy(tmp_path)
    policy.departments["XN3"] = Department("XN3", "Xí nghiệp tuyến 3", doc_codes=("PIC-", "HANOÏ L3"))
    policy.departments["XN2"] = Department("XN2", "Xí nghiệp tuyến 2A", doc_codes=("TXWJ-",))
    policy.departments["TCNS"] = Department("TCNS", "Tổ chức nhân sự")
    doc = lambda f, sh="": SimpleNamespace(so_hieu=sh, co_quan="", don_vi="", source_path=f"x/{f}", source_file=f,
                                            trich_yeu="", title="")
    assert doc_code_hits(doc("PIC-OMP-PRC-ZMO-L00-00134-E-2A.pdf"), policy) == ["XN3"], "-2A là phiên bản, không phải XN2"
    assert doc_code_hits(doc("054 phương án.pdf"), policy, "Mã số hồ sơ: TXWJ-054 Phiên bản A") == ["XN2"]
    assert doc_code_hits(doc("bao cao.pdf"), policy, "nội dung thường") == []
    assert has_personal_ids("Căn cước công dân 037201004350") and not has_personal_ids("số 037201004350")
    codes, src = draft_split_codes(doc("1245 QĐ.pdf"), policy, {}, "Chứng minh nhân dân/Căn cước 015089000144")
    assert codes == ["TCNS"] and "CCCD" in src
