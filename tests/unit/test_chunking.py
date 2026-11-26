from src.indexing.profiles import classify_profile, segment_document
from src.indexing.structure import AdminInfo, build_blocks, Line
from src.indexing.structured import ChunkingParams, StructuredChunker, split_long_text
from src.text.normalize import approx_tokens

from conftest import BB_DUY_TAN, CV_916, make_doc

QUY_CHE = "\n".join(
    ["QUY CHẾ", "Quản lý tài sản công nghệ thông tin", "Chương I", "QUY ĐỊNH CHUNG"]
    + [line for n in range(1, 7) for line in (
        f"Điều {n}. Nội dung điều {n}",
        f"1. Khoản thứ nhất của điều {n} quy định trách nhiệm của các phòng ban liên quan trong việc quản lý.",
        f"2. Khoản thứ hai của điều {n} quy định thời hạn báo cáo định kỳ hằng quý.",
    )]
)


def test_real_documents_get_h1_with_one_chunk_and_metadata():
    chunker = StructuredChunker()
    doc, parents, chunks = chunker.chunk_file(CV_916)
    assert doc.ho_so == "H1" and len(chunks) == 1 and len(parents) == 1
    assert chunks[0].parent_id == parents[0].parent_id
    assert chunks[0].context.startswith("[CV 916/CV-VT&QLPT | 25/09/2026 | PHÒNG VẬN TẢI & QUẢN LÝ PHƯƠNG TIỆN | V/v Triển khai")
    assert "916/CV-VT&QLPT" in chunks[0].codes and doc.cites == ["485/TT-KHDA"]
    assert "Thời gian: Thứ 7" in chunks[0].search_text, "bảng được làm phẳng trong nội dung đánh chỉ mục"
    assert "| Thời gian |" in chunks[0].text, "nội dung hiển thị giữ bảng Markdown"

    doc, _, chunks = chunker.chunk_file(BB_DUY_TAN)
    assert (doc.loai_vb, doc.ngay_ban_hanh, doc.nam) == ("BB", "2026-10-09", 2026)
    assert doc.ho_so_confidence == "low", "văn bản thiếu tín hiệu phải vào danh sách rà soát"
    assert (chunks[0].page_from, chunks[0].page_to) == (1, 2)


def test_regulation_is_h2_one_article_per_chunk():
    doc, parents, chunks = StructuredChunker().chunk_document(make_doc([QUY_CHE], name="Quy che CNTT.pdf"))
    assert doc.ho_so == "H2"
    article_chunks = [c for c in chunks if "Điều" in c.breadcrumb]
    assert len(article_chunks) == 6
    assert all("2. Khoản thứ hai" in c.text for c in article_chunks), "Khoản nằm trong chunk của Điều"
    assert article_chunks[2].breadcrumb.endswith("Điều 3. Nội dung điều 3")
    assert len({c.parent_id for c in article_chunks}) == 6


def test_decision_with_attached_regulation_is_split_h1_then_h2():
    decision = "\n".join(
        ["QUYẾT ĐỊNH", "Về việc ban hành Quy chế quản lý tài sản"]
        + [f"Điều {n}. Ban hành kèm theo Quyết định này nội dung {n}." for n in range(1, 4)]
    )
    blocks = build_blocks([Line(t, 1) for t in (decision + "\n" + QUY_CHE).splitlines()])
    segments = segment_document("H2", blocks)
    assert [s.profile for s in segments] == ["H1", "H2"]
    assert segments[1].blocks[0].text.startswith("QUY CHẾ")


def test_long_technical_manual_is_h3_with_bounded_children_and_parents():
    para = "Kiểm tra nhiệt độ phòng thiết bị, ghi nhận độ ẩm và tình trạng quạt làm mát của tủ UPS theo ca trực. " * 12
    pages = [f"{s}. Mục lớn {s}\n" + "\n".join(f"{s}.{k}.1 Mục con {s}.{k}\n{para}\n\n{para}" for k in range(1, 4)) for s in range(1, 13)]
    params = ChunkingParams()
    doc, parents, chunks = StructuredChunker(params).chunk_document(make_doc(pages, name="UJV-OPM-UPS-001_V.pdf"))
    assert doc.ho_so == "H3"
    assert doc.pair_id == "UJV-OPM-UPS-001"
    assert all(c.tokens <= params.child_max + 5 for c in chunks)
    assert all(approx_tokens(p.text) <= params.parent_max * 1.1 for p in parents)
    assert {c.parent_id for c in chunks} <= {p.parent_id for p in parents}
    assert chunks[0].context.startswith("[UJV-OPM-UPS-001_V (UJV-OPM-UPS-001) | VI]")


def test_big_table_document_is_h4_and_split_with_header():
    rows = "\n".join(f"| {i} | Thay dầu máy nén khí ga {i} | 2 giờ | Kỹ thuật viên bậc 3 |" for i in range(80))
    table = "| STT | Công việc | Thời gian | Kỹ năng |\n|---|---|---|---|\n" + rows
    doc, _, chunks = StructuredChunker().chunk_document(make_doc(["Danh mục định mức", table], name="CIDM dinh muc.pdf"))
    assert doc.ho_so == "H4"
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 4
    assert all(c.text.startswith("| STT | Công việc | Thời gian | Kỹ năng |") for c in table_chunks)
    assert "Công việc: Thay dầu máy nén khí ga 79" in table_chunks[-1].search_text


def test_profile_falls_back_by_length_when_no_signal():
    assert classify_profile("giay_phep.pdf", 1, AdminInfo(), [], []).profile == "H1"
    decision = classify_profile("ban_ve.pdf", 30, AdminInfo(), [], [])
    assert decision.profile == "H3"


def test_split_long_text_overlaps_only_when_splitting():
    text = "\n\n".join(f"Đoạn {i} " + "nội dung kỹ thuật " * 40 for i in range(6))
    parts = split_long_text(text, max_tokens=250, overlap_tokens=120)
    assert len(parts) > 1
    assert parts[1].startswith("Đoạn 1") or "Đoạn 1" in parts[1].split("\n")[0]
    assert split_long_text("ngắn", 250, 60) == ["ngắn"]


def test_short_numbered_lines_are_not_lost_as_empty_headings():
    text = ("QUYẾT ĐỊNH\nĐiều 1. Phê duyệt chủ trương sửa chữa với nội dung sau:\n"
            "4. Dự kiến chi phí: 1.263.160.000 đồng\n5. Nguồn kinh phí: Chi thường xuyên từ ngân sách Thành phố.\n"
            "6. Thời gian thực hiện: Năm 2026.\nĐiều 2. Tổ chức thực hiện\n1. Công ty chịu trách nhiệm toàn diện về số liệu.")
    long_tail = "\n\n" + "Nội dung giải trình chi tiết về khối lượng công việc. " * 120
    _, _, chunks = StructuredChunker().chunk_document(make_doc([text + long_tail], name="Quyet dinh 1977-QĐ-SXD.pdf"))
    joined = "\n".join(c.text for c in chunks)
    for line in ("Dự kiến chi phí: 1.263.160.000", "Nguồn kinh phí", "Thời gian thực hiện: Năm 2026"):
        assert joined.count(line) == 1, line


def test_blank_number_on_print_is_taken_from_file_name_with_type_first():
    text = "CÔNG TY TNHH MTV ĐƯỜNG SẮT HÀ NỘI\nSố:      /TTr-NV\nTỜ TRÌNH\nV/v cử cán bộ đi công tác nước ngoài\n" + "Nội dung tờ trình. " * 80
    doc, _, _ = StructuredChunker().chunk_document(make_doc([text], name="TTr 1085 Đường sắt HN đi TQ.pdf"))
    assert doc.so_hieu.upper() == "1085/TTR-NV"


def test_project_document_code_in_filename_is_the_number():
    src = make_doc(["# Procedure\n\nMaintenance of escalators 00171-E-2A."], name="PIC-OMP-PRC-ZMO-L00-00171-E-2A.pdf")
    doc, _, _ = StructuredChunker().chunk_document(src)
    assert doc.so_hieu == "PIC-OMP-PRC-ZMO-L00-00171-E-2A"
    versioned = make_doc(["# Test"], name="UJV-OPM-TMA-TCS-L00-36087-E-3A-S200-3.05.pdf")
    assert StructuredChunker().chunk_document(versioned)[0].so_hieu == "UJV-OPM-TMA-TCS-L00-36087-E-3A-S200-3"
