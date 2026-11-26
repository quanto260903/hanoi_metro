from src.text.abbrev import AbbreviationDictionary, AbbreviationEntry
from src.text.codes import extract_codes, normalize_code
from src.text.normalize import collapse_repetitions, content_terms, fold, normalize_text


def test_extract_codes_covers_document_numbers_and_equipment_codes():
    text = ("Số: 916/CV-VT&QLPT; căn cứ Tờ trình số: 485/TT-KHDA ngày 24/09/2026, 06-QC/TU, 983/ĐA-ĐSĐT-HCTC, "
            "thiết bị DC2-HN01, gói CP-06, tài liệu HNC-IMP-PPG-ZTO-D00-00015, serial N3K93180, Tuyến 2A")
    codes = extract_codes(text)
    for expected in ["916/CV-VT&QLPT", "485/TT-KHDA", "06-QC/TU", "983/ĐA-ĐSĐT-HCTC", "DC2-HN01", "CP-06",
                     "HNC-IMP-PPG-ZTO-D00-00015", "N3K93180"]:
        assert expected in codes
    assert not any(c.startswith("24/09") for c in codes), "ngày tháng không phải số hiệu"


def test_normalize_code_unifies_spacing_and_case():
    assert normalize_code(" 916 / cv-vt & qlpt ") == "916/CV-VT&QLPT"


def test_normalize_text_fixes_font_glyphs_and_glued_words():
    assert normalize_text("ĐƢỜNG sắt") == "ĐƯỜNG sắt"
    assert normalize_text("SởXây dựng") == "Sở Xây dựng"
    assert normalize_text("iPhone HPE McAfee") == "iPhone HPE McAfee"


def test_fold_removes_vietnamese_diacritics_including_d():
    assert fold("Báo cáo Đường sắt") == "bao cao duong sat"


def test_collapse_repetitions_removes_vlm_signature_loop():
    block = ["Đại diện", "Phòng", "VT&QLPT", "Xí nghiệp 2A"]
    lines = ["Biên bản này lập 05 bản."] + block * 35 + ["hết"]
    assert collapse_repetitions(lines) == ["Biên bản này lập 05 bản."] + block + ["hết"]


def test_collapse_repetitions_keeps_short_legit_repeats():
    lines = ["| 1 | a |", "| 2 | a |", "", "", "x"]
    assert collapse_repetitions(lines) == lines


def test_content_terms_keeps_bao_cao_but_drops_bao_nhieu():
    assert content_terms("bao nhiêu báo cáo") == ["báo", "cáo"]


def test_abbreviation_expands_both_directions_and_ignores_unapproved():
    d = AbbreviationDictionary([
        AbbreviationEntry("BCVT", ["báo cáo văn thư"], approved=True),
        AbbreviationEntry("VT", ["vận tải"], approved=False),
    ])
    forward = d.expand("Tìm BCVT tháng 9")
    assert forward[0].matched == "BCVT" and forward[0].alternatives == ["báo cáo văn thư"]
    backward = d.expand("bao cao van thu quý 3")
    assert backward[0].alternatives == ["BCVT"]
    assert d.expand("phòng VT") == []


def test_mine_candidates_uses_initials_for_vietnamese_names():
    found = AbbreviationDictionary.mine_candidates("Phòng Kế hoạch Dự án (KHDA) và Trung tâm điều hành (OCC)")
    assert ("KHDA", "Kế hoạch Dự án") in found
    assert ("OCC", "Trung tâm điều hành") in found  # viết tắt tiếng Anh: giữ cả cụm cho người duyệt cắt


def test_document_number_with_year_in_the_middle():
    from src.text.codes import extract_codes
    assert extract_codes("Căn cứ Nghị định số 30/2020/NĐ-CP và công văn 449/2026/CV-VHM-VSP") == ["30/2020/NĐ-CP", "449/2026/CV-VHM-VSP"]
    assert extract_codes("Số: 916/CV-VT&QLPT ngày 25/9/2026") [0] == "916/CV-VT&QLPT"
