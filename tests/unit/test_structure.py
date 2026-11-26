from src.indexing.document import load_markdown
from src.indexing.structure import (
    build_blocks, classify_heading, clean_pages, extract_admin, flatten_table, split_table, table_rows,
)

from conftest import BB_DUY_TAN, CV_916


def _lines(path):
    doc = load_markdown(path)
    return clean_pages([(p.number, p.text) for p in doc.pages])


def test_load_markdown_reads_md5_and_pages():
    doc = load_markdown(BB_DUY_TAN)
    assert doc.doc_id == "e2311c730de842ef3c56314f98f4250e"
    assert [p.number for p in doc.pages] == [1, 2]
    assert doc.pages[0].route == "VLM"


def test_extract_admin_moves_form_fields_out_of_body():
    lines, _ = _lines(CV_916)
    info, body = extract_admin(lines)
    assert info.so_hieu == "916/CV-VT&QLPT"
    assert info.loai_vb == "CV"
    assert info.trich_yeu.startswith("Triển khai kế hoạch tổ chức khảo sát hệ thống thu soát vé tự động AFC")
    assert info.ngay_ban_hanh == "2026-09-25"
    assert info.kinh_gui == ["Xí nghiệp vận hành Tuyến ĐSĐT số 2A", "Xí nghiệp vận hành Tuyến ĐSĐT số 3"]
    assert info.noi_nhan == ["Như trên", "KHDA( Tổ CNTT)", "Lưu: VT&QLPT"]
    assert "quoc_hieu" in info.markers
    assert info.nguoi_ky == "Nguyễn Xuân Hùng"
    assert info.don_vi == "PHÒNG VẬN TẢI & QUẢN LÝ PHƯƠNG TIỆN"
    body_text = "\n".join(ln.text for ln in body)
    assert "Kính gửi" not in body_text and "Nơi nhận" not in body_text and "Số: 916" not in body_text
    assert "Căn cứ Tờ trình số: 485/TT-KHDA" in body_text


def test_vlm_signature_loop_is_collapsed():
    block = "Đại diện\nPhòng\nVT&QLPT\nXí nghiệp 2A\nCông ty Thăng\nLong Soft"
    lines, _ = clean_pages([(1, "Biên bản lập 05 bản./.\n" + "\n".join([block] * 35))])
    assert sum(1 for ln in lines if ln.text.strip() == "Long Soft") == 1


def test_handwriting_markers_unwrapped():
    lines, _ = clean_pages([(1, "Hà Nội, ngày [Viết tay: 9] tháng 10 năm 2026\n| 1 | [Viết tay: Q.NAP] | [Viết tay: BT] |")])
    assert [ln.text for ln in lines] == ["Hà Nội, ngày 9 tháng 10 năm 2026", "| 1 | Q.NAP | BT |"]


def test_table_merged_across_pages():
    lines, _ = _lines(BB_DUY_TAN)
    info, body = extract_admin(lines)
    assert info.loai_vb == "BB"
    tables = [b for b in build_blocks(body) if b.kind == "table"]
    assert len(tables) == 1
    assert (tables[0].page_from, tables[0].page_to) == (1, 2)
    header, rows = table_rows(tables[0].text)
    assert header[0] == "STT" and len(rows) == 10


def test_repeated_header_footer_removed_on_long_documents():
    pages = [(i, f"DỰ ÁN TUYẾN 3.1 – MÃ HNC-IMP-001\nNội dung riêng trang {i} về bảo trì.\nTrang {i}/5") for i in range(1, 6)]
    lines, _ = clean_pages(pages)
    texts = [ln.text for ln in lines]
    assert not any("DỰ ÁN TUYẾN" in t for t in texts)
    assert not any(t.startswith("Trang ") for t in texts)
    assert sum("Nội dung riêng" in t for t in texts) == 5


def test_toc_lines_collected_but_form_dots_are_not_toc():
    lines, toc = clean_pages([(1, "MỤC LỤC\n1. Phạm vi áp dụng ........ 3\n2. Trách nhiệm ......... 5\nHọ tên: ....................")])
    assert toc == ["1. Phạm vi áp dụng", "2. Trách nhiệm"]
    assert any("Họ tên" in ln.text for ln in lines)


def test_classify_heading_recognizes_legal_and_numbered_structure():
    assert classify_heading("Điều 5. Phạm vi điều chỉnh")[2] == "dieu"
    assert classify_heading("Chương II") == (1, "Chương II", "chuong")
    assert classify_heading("1.1.1 Nhiệt độ và độ ẩm")[0] == 5
    assert classify_heading("II. NỘI DUNG")[0] == 2
    assert classify_heading("1. Giao phòng Kế hoạch chủ trì, phối hợp với các đơn vị;") is None


def test_html_table_with_rowspan_is_flattened_with_column_names():
    html = ("<table><tr><th>Ngày</th><th>Ga</th></tr><tr><td rowspan=\"2\">Thứ 7</td><td>Lê Đức Thọ</td></tr>"
            "<tr><td>Phùng Khoang</td></tr></table>")
    assert flatten_table(html) == "Ngày: Thứ 7; Ga: Lê Đức Thọ\nNgày: Thứ 7; Ga: Phùng Khoang"


def test_markdown_merged_cells_filled_down_when_flattened():
    md = "| Thời gian | Nhân sự |\n|---|---|\n| Thứ 7 | Lộc |\n|  | Tuấn |"
    assert flatten_table(md).splitlines()[1] == "Thời gian: Thứ 7; Nhân sự: Tuấn"


def test_split_table_repeats_header_on_every_part():
    md = "| STT | Tên |\n|---|---|\n" + "\n".join(f"| {i} | thiết bị {i} |" for i in range(45))
    parts = split_table(md, 20)
    assert len(parts) == 3
    assert all(p.startswith("| STT | Tên |") for p in parts)
    assert sum(len(table_rows(p)[1]) for p in parts) == 45


def test_document_number_and_date_from_filename():
    from src.indexing.structure import admin_from_filename
    assert admin_from_filename("470 TB-UBND Ngày 1452018.pdf") == ("470/TB-UBND", "2018-05-14")
    assert admin_from_filename("Quyết định số 1963-QĐ-SXD 14.10.26 duyệt chủ trương.pdf") == ("1963/QĐ-SXD", "2026-10-14")
    assert admin_from_filename("Tờ trình số 363-TTr-SXD 08.8.2026 về dự toán.pdf") == ("363/TTR-SXD", "2026-08-08")
    assert admin_from_filename("2034-CV-ĐSHN ngày 30.7.2026 - bản ký.pdf") == ("2034/CV-ĐSHN", "2026-07-30")
    assert admin_from_filename("916.250926CV-VT&QLPT. Kế hoạch khảo sát.pdf")[0] != "50926/CV-VT&QLPT"
