"""
Hanoi Metro AI Agent - End-to-End Ingestion Pipeline.
Chịu trách nhiệm:
1. Quản lý trạng thái xử lý bằng SQLite (state.db) chống chạy lặp tài liệu đã xử lý.
2. Quét thư mục data/raw_pdfs/, phân loại từng trang bằng classifier và bóc tách qua parser tương ứng.
3. Tổng hợp tất cả các trang thành 1 file .md duy nhất lưu tại data/processed_md/ kèm Header Metadata chuẩn.
4. Cung cấp giao diện dòng lệnh (CLI) với đầy đủ tham số điều khiển.
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Suppress PyMuPDF FutureWarning về fitz alias (dùng import fitz vẫn hoạt động bình thường)
warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from src.config import settings
from src.ingestion.classifier import PageClassifier, PageType, PageClassificationResult
from src.ingestion.parsers import create_parser, BaseParser
from src.ingestion.cloud_vlm import BudgetExceeded, CloudVLMParser, VLMUnavailable
from src.indexing.indexer import Blocklist

logger = logging.getLogger("hanoi_metro.pipeline")


# ==============================================================================
# QUẢN LÝ TRẠNG THÁI BẰNG SQLITE (STATE.DB)
# ==============================================================================
class StateManager:
    """Quản lý trạng thái lưu vết xử lý tài liệu trong SQLite để tránh chạy lặp."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối tới SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Khởi tạo cấu trúc bảng SQLite nếu chưa tồn tại."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Bảng lưu trạng thái file PDF tổng thể
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS file_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    md5_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    total_pages INTEGER DEFAULT 0,
                    status TEXT NOT NULL,  -- PENDING, PROCESSING, COMPLETED, FAILED, SKIPPED
                    output_md_path TEXT,
                    page_stats TEXT,        -- JSON summary of page types
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Bảng chi tiết trạng thái phân loại từng trang
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS page_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    page_type TEXT NOT NULL,
                    parser_used TEXT NOT NULL,
                    reasons TEXT,           -- JSON array
                    confidence REAL,
                    status TEXT NOT NULL,   -- SUCCESS, FAILED
                    error_message TEXT,
                    processed_at TEXT NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES file_state (id) ON DELETE CASCADE
                )
                """
            )
            # Chỉ mục tăng tốc truy vấn MD5 và File Path
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_md5 ON file_state (md5_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON file_state (file_path)")
            conn.commit()

    @staticmethod
    def compute_md5(file_path: Path) -> str:
        """Tính mã hash MD5 của file nhị phân."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def should_process(self, file_path: Path, current_md5: str, force: bool = False) -> Tuple[bool, Optional[int]]:
        """
        Kiểm tra xem file có cần xử lý hay không.
        Trả về: (cần_xử_lý: bool, file_id_hiện_tại: Optional[int])
        """
        if force:
            return True, None

        normalized_path = str(file_path.resolve())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, md5_hash, status FROM file_state WHERE file_path = ? OR md5_hash = ?",
                (normalized_path, current_md5),
            )
            row = cursor.fetchone()
            if not row:
                return True, None

            file_id = row["id"]
            db_md5 = row["md5_hash"]
            db_status = row["status"]

            # Nếu file đã hoàn thành và nội dung không đổi (cùng MD5) -> Bỏ qua
            if db_status == "COMPLETED" and db_md5 == current_md5:
                return False, file_id

            # Nếu trước đó bị lỗi hoặc file đã bị sửa đổi nội dung -> Cần chạy lại
            return True, file_id

    def record_start(self, file_path: Path, md5_hash: str, file_size: int, total_pages: int) -> int:
        """Ghi nhận bắt đầu xử lý file vào SQLite."""
        normalized_path = str(file_path.resolve())
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO file_state (file_path, file_name, md5_hash, file_size_bytes, total_pages, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'PROCESSING', ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    md5_hash = excluded.md5_hash,
                    file_size_bytes = excluded.file_size_bytes,
                    total_pages = excluded.total_pages,
                    status = 'PROCESSING',
                    error_message = NULL,
                    updated_at = excluded.updated_at
                """,
                (normalized_path, file_path.name, md5_hash, file_size, total_pages, now, now),
            )
            conn.commit()

            cursor.execute("SELECT id FROM file_state WHERE file_path = ?", (normalized_path,))
            file_id = cursor.fetchone()["id"]

            # Xóa các bản ghi trang cũ nếu có
            cursor.execute("DELETE FROM page_state WHERE file_id = ?", (file_id,))
            conn.commit()
            return file_id

    def record_page(
        self,
        file_id: int,
        page_num: int,
        page_type: PageType,
        parser_name: str,
        reasons: List[str],
        confidence: float,
        success: bool,
        error_msg: Optional[str] = None,
    ) -> None:
        """Ghi nhận thông tin xử lý từng trang."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO page_state (file_id, page_number, page_type, parser_used, reasons, confidence, status, error_message, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    page_num,
                    page_type.value,
                    parser_name,
                    json.dumps(reasons, ensure_ascii=False),
                    confidence,
                    "SUCCESS" if success else "FAILED",
                    error_msg,
                    now,
                ),
            )
            conn.commit()

    def record_complete(
        self, file_id: int, output_md_path: Path, page_stats: Dict[str, int]
    ) -> None:
        """Cập nhật trạng thái hoàn thành cho file."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE file_state
                SET status = 'COMPLETED',
                    output_md_path = ?,
                    page_stats = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (str(output_md_path.resolve()), json.dumps(page_stats), now, file_id),
            )
            conn.commit()

    def record_failure(self, file_id: int, error_message: str) -> None:
        """Cập nhật trạng thái thất bại cho file."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE file_state
                SET status = 'FAILED',
                    error_message = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error_message, now, file_id),
            )
            conn.commit()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Lấy bảng thống kê tổng quát trạng thái ingestion."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(*) as count FROM file_state GROUP BY status")
            file_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

            cursor.execute("SELECT page_type, COUNT(*) as count FROM page_state GROUP BY page_type")
            page_counts = {row["page_type"]: row["count"] for row in cursor.fetchall()}

            return {
                "file_status_counts": file_counts,
                "page_type_counts": page_counts,
            }


# ==============================================================================
# PIPELINE CHÍNH (INGESTION PIPELINE)
# ==============================================================================
class IngestionPipeline:
    """Pipeline điều phối ingestion toàn diện cho tài liệu Hà Nội Metro."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.raw_dir = raw_dir or settings.ingestion.raw_pdfs_dir
        self.processed_dir = processed_dir or settings.ingestion.processed_md_dir
        self.db_path = db_path or settings.ingestion.state_db_path

        self.classifier = PageClassifier()
        self.state_manager = StateManager(self.db_path)
        # Tài liệu có dấu chỉ độ mật: không OCR, không gửi API, không sinh Markdown
        self.blocklist = Blocklist.from_dir(settings.retrieval.blocklist_dir)

        # Cache các parser instance để tái sử dụng
        self._parsers: Dict[PageType, BaseParser] = {}

    def _get_parser(self, page_type: PageType) -> BaseParser:
        """Lấy parser tương ứng từ cache hoặc khởi tạo mới."""
        if page_type not in self._parsers:
            self._parsers[page_type] = create_parser(page_type)
        return self._parsers[page_type]

    def process_all(self, force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """
        Quét và xử lý toàn bộ các file PDF trong thư mục raw_pdfs.
        """
        logger.info(f"=== BẮT ĐẦU INGESTION PIPELINE ===")
        logger.info(f"Thư mục PDF đầu vào: {self.raw_dir}")
        logger.info(f"Thư mục Markdown đầu ra: {self.processed_dir}")
        logger.info(f"Cơ sở dữ liệu State: {self.db_path}")

        pdf_files = sorted(list(self.raw_dir.glob("*.pdf")))
        if not pdf_files:
            logger.warning(f"Không tìm thấy file .pdf nào trong thư mục: {self.raw_dir}")
            return {"total_files": 0, "processed": 0, "skipped": 0, "failed": 0}

        logger.info(f"Tìm thấy {len(pdf_files)} file PDF cần quét.")
        stats = {"total_files": len(pdf_files), "processed": 0, "skipped": 0, "failed": 0}

        for pdf_path in pdf_files:
            try:
                res = self.process_file(pdf_path, force=force, dry_run=dry_run)
            except BudgetExceeded as e:
                logger.error(f"DỪNG PIPELINE – chạm hạn mức chi phí VLM: {e}")
                stats["stopped_budget"] = True
                break
            except Exception as e:
                logger.error(f"Lỗi không xử lý được file {pdf_path.name}: {str(e)}", exc_info=True)
                stats["failed"] += 1
                continue
            if res == "SKIPPED":
                stats["skipped"] += 1
            elif res == "PROCESSED":
                stats["processed"] += 1
            elif res == "BLOCKED":
                stats["blocked"] = stats.get("blocked", 0) + 1
            else:
                stats["failed"] += 1

        logger.info(f"=== KẾT THÚC PIPELINE: Đã xử lý {stats['processed']} | Bỏ qua {stats['skipped']} | Lỗi {stats['failed']} ===")
        return stats

    def process_file(self, pdf_path: Path, force: bool = False, dry_run: bool = False) -> str:
        """
        Xử lý từng file PDF đơn lẻ.
        Trả về trạng thái: "PROCESSED", "SKIPPED", hoặc "FAILED".
        """
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) chưa được cài đặt.")

        file_name = pdf_path.name
        file_size = pdf_path.stat().st_size
        md5_hash = StateManager.compute_md5(pdf_path)
        if self.blocklist.is_blocked(md5_hash):
            logger.warning(f"[BLOCKED] '{file_name}' có dấu chỉ độ mật: không xử lý, không gửi API, không sinh Markdown.")
            return "BLOCKED"

        # 1. Kiểm tra trạng thái trong SQLite
        should_run, existing_id = self.state_manager.should_process(pdf_path, md5_hash, force=force)
        if not should_run:
            logger.info(f"[SKIP] File '{file_name}' (MD5: {md5_hash[:8]}...) đã xử lý xong từ trước. Bỏ qua.")
            return "SKIPPED"

        logger.info(f"[PROCESSING] Bắt đầu xử lý file: '{file_name}' ({file_size / 1024:.1f} KB, MD5: {md5_hash[:8]}...)")

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
        except Exception as e:
            logger.error(f"Không thể mở file PDF '{file_name}': {e}")
            if existing_id:
                self.state_manager.record_failure(existing_id, f"Lỗi đọc file PDF: {str(e)}")
            return "FAILED"

        # Ghi nhận trạng thái bắt đầu vào DB
        file_id = self.state_manager.record_start(pdf_path, md5_hash, file_size, total_pages)

        page_markdowns: List[str] = []
        page_classification_details: List[Dict[str, Any]] = []
        page_type_counts: Dict[str, int] = {
            PageType.DIRECT_TEXT.value: 0,
            PageType.STANDARD_OCR.value: 0,
            PageType.VLM.value: 0,
        }

        try:
            # 2. Xử lý tuần tự từng trang
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc[page_idx]

                # Phân loại trang
                cls_result = self.classifier.classify_page(page, page_num)
                page_type = cls_result.page_type
                page_type_counts[page_type.value] += 1

                logger.info(
                    f"-> Trang {page_num}/{total_pages}: [{page_type.value}] - Lý do: {', '.join(cls_result.reasons[:2])}"
                )

                page_detail = {
                    "page": page_num,
                    "type": page_type.value,
                    "confidence": cls_result.confidence,
                    "reasons": cls_result.reasons,
                }
                page_classification_details.append(page_detail)

                # Nếu chạy ở chế độ dry-run, chỉ phân loại không gọi parser
                if dry_run:
                    page_content = f"*[Nội dung Trang {page_num} đã phân loại thành {page_type.value} (Dry-run mode)]*"
                    page_markdowns.append(page_content)
                    self.state_manager.record_page(
                        file_id=file_id,
                        page_num=page_num,
                        page_type=page_type,
                        parser_name="DryRunParser",
                        reasons=cls_result.reasons,
                        confidence=cls_result.confidence,
                        success=True,
                    )
                    continue

                # Lựa chọn parser và bóc tách
                parser = self._get_parser(page_type)
                parser_name = parser.__class__.__name__

                try:
                    if isinstance(parser, CloudVLMParser):
                        page_content = parser.parse(page, page_num=page_num, pdf_path=str(pdf_path), doc_md5=md5_hash)
                    else:
                        page_content = parser.parse(page, page_num=page_num, pdf_path=str(pdf_path))
                    self.state_manager.record_page(
                        file_id=file_id,
                        page_num=page_num,
                        page_type=page_type,
                        parser_name=parser_name,
                        reasons=cls_result.reasons,
                        confidence=cls_result.confidence,
                        success=True,
                    )
                except (BudgetExceeded, VLMUnavailable):
                    raise  # dừng / đánh dấu file thất bại, không thay bằng chữ thô
                except Exception as p_err:
                    logger.error(f"Lỗi khi bóc tách trang {page_num} bằng {parser_name}: {p_err}", exc_info=True)
                    # Ghi nhận lỗi và fallback trang
                    self.state_manager.record_page(
                        file_id=file_id,
                        page_num=page_num,
                        page_type=page_type,
                        parser_name=parser_name,
                        reasons=cls_result.reasons,
                        confidence=cls_result.confidence,
                        success=False,
                        error_msg=str(p_err),
                    )
                    # Fallback đơn giản PyMuPDF text để không làm mất dữ liệu toàn file
                    fallback_text = page.get_text("text") or ""
                    page_content = (
                        f"> [!WARNING]\n"
                        f"> Không thể hoàn tất bóc tách bằng {parser_name} ở trang {page_num}: {p_err}\n"
                        f"> Nội dung fallback từ lớp text thô:\n\n{fallback_text}"
                    )

                page_markdowns.append(page_content)

            # 3. Nối các trang và xử lý bảng bị ngắt quãng giữa các trang
            merged_body = self._stitch_pages_and_tables(page_markdowns, page_classification_details)

            # 4. Tạo Header Metadata YAML Frontmatter
            header_metadata = self._generate_header_metadata(
                file_name=file_name,
                pdf_path=pdf_path,
                md5_hash=md5_hash,
                total_pages=total_pages,
                page_stats=page_type_counts,
                page_details=page_classification_details,
            )

            # 5. Lưu file kết quả .md duy nhất vào data/processed_md/
            output_stem = pdf_path.stem
            output_file = self.processed_dir / f"{output_stem}.md"

            final_markdown_document = f"{header_metadata}\n\n{merged_body}\n"
            with open(output_file, "w", encoding="utf-8") as f_out:
                f_out.write(final_markdown_document)

            # 6. Cập nhật trạng thái hoàn thành vào SQLite
            self.state_manager.record_complete(file_id, output_file, page_type_counts)
            logger.info(f"[SUCCESS] Hoàn tất bóc tách file '{file_name}' -> Lưu tại: '{output_file.name}'")
            return "PROCESSED"

        except BudgetExceeded as budget_err:
            self.state_manager.record_failure(file_id, f"Dừng vì chạm hạn mức VLM: {budget_err}")
            raise
        except Exception as file_err:
            logger.error(f"Thất bại trong quá trình bóc tách file '{file_name}': {file_err}", exc_info=True)
            self.state_manager.record_failure(file_id, str(file_err))
            return "FAILED"
        finally:
            doc.close()

    def _generate_header_metadata(
        self,
        file_name: str,
        pdf_path: Path,
        md5_hash: str,
        total_pages: int,
        page_stats: Dict[str, int],
        page_details: List[Dict[str, Any]],
    ) -> str:
        """Tạo khối Header Metadata (YAML Frontmatter) ở đầu file Markdown."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Rút trích tiêu đề gợi ý từ tên file
        clean_title = file_name.replace(".pdf", "").replace("_", " ").replace("-", " ")

        yaml_lines = [
            "---",
            f'title: "{clean_title}"',
            f'source_file: "{file_name}"',
            f'source_path: "{pdf_path.as_posix()}"',
            f'md5_hash: "{md5_hash}"',
            f"total_pages: {total_pages}",
            f'processed_at: "{now_str}"',
            'organization: "Công ty TNHH MTV Đường sắt Hà Nội (Hanoi Metro)"',
            'system: "Hệ thống AI Agent Xử lý Tài liệu Nội bộ"',
            "classification_summary:",
            f"  DIRECT_TEXT: {page_stats.get(PageType.DIRECT_TEXT.value, 0)}",
            f"  STANDARD_OCR: {page_stats.get(PageType.STANDARD_OCR.value, 0)}",
            f"  VLM: {page_stats.get(PageType.VLM.value, 0)}",
            "page_routes:",
        ]

        for pd in page_details:
            reasons_str = "; ".join(pd["reasons"][:2]).replace('"', "'")
            yaml_lines.append(
                f'  - page: {pd["page"]}, type: {pd["type"]}, confidence: {pd["confidence"]}, note: "{reasons_str}"'
            )

        yaml_lines.append("---")
        return "\n".join(yaml_lines)

    def _stitch_pages_and_tables(
        self,
        pages_content: List[str],
        page_details: List[Dict[str, Any]],
    ) -> str:
        """
        Ghép các trang thành tài liệu liền mạch.
        Nhận diện và nối liền các bảng Markdown nếu bị ngắt ngang qua ranh giới giữa 2 trang liên tiếp.
        """
        if not pages_content:
            return ""

        stitched_sections: List[str] = []

        for idx, content in enumerate(pages_content):
            page_num = idx + 1
            page_type = page_details[idx]["type"]

            # Phân tách trang bằng marker rõ ràng
            page_marker = f"<!-- ========================================== -->\n<!-- TRANG {page_num} [{page_type}] -->\n<!-- ========================================== -->"

            # Kiểm tra xem có xảy ra hiện tượng nối bảng từ trang trước sang trang này không
            if idx > 0 and len(stitched_sections) > 0:
                prev_text = stitched_sections[-1].strip()
                curr_text = content.strip()

                # Nếu trang trước kết thúc bằng hàng bảng (`| ... |`) và trang này bắt đầu bằng hàng bảng
                prev_lines = prev_text.split("\n")
                curr_lines = curr_text.split("\n")

                if prev_lines and curr_lines:
                    prev_last_line = prev_lines[-1].strip()
                    curr_first_line = curr_lines[0].strip()

                    is_prev_table = prev_last_line.startswith("|") and prev_last_line.endswith("|")
                    is_curr_table = curr_first_line.startswith("|") and curr_first_line.endswith("|")

                    # Nếu cả hai đều là bảng và trang hiện tại lặp lại tiêu đề bảng giống hệt trang trước -> bỏ qua tiêu đề lặp
                    if is_prev_table and is_curr_table and len(curr_lines) >= 2:
                        second_line = curr_lines[1].strip()
                        # Dòng thứ 2 là đường phân cách `| --- | --- |`
                        if "---" in second_line:
                            logger.info(f"Phát hiện bảng nối tiếp qua trang {page_num}: Loại bỏ tiêu đề bảng lặp.")
                            content = "\n".join(curr_lines[2:]).strip()

            section = f"{page_marker}\n\n{content}"
            stitched_sections.append(section)

        return "\n\n".join(stitched_sections)


# ==============================================================================
# GIAO DIỆN DÒNG LỆNH (CLI ENTRYPOINT)
# ==============================================================================
def main():
    """Hàm chạy dòng lệnh chính của Ingestion Pipeline."""
    parser = argparse.ArgumentParser(
        description="Hanoi Metro AI Agent - Document Ingestion Pipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Đường dẫn thư mục chứa PDF gốc (mặc định: data/raw_pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Đường dẫn thư mục lưu file Markdown đã xử lý (mặc định: data/processed_md)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Chỉ định đường dẫn xử lý riêng một file PDF cụ thể",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc chạy lại toàn bộ bất kể file đã có trong state.db hay chưa",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ quét và phân loại trang (không gọi VLM/OCR parser)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Hiển thị thống kê tổng quan các tài liệu đã xử lý trong database state.db",
    )

    args = parser.parse_args()

    pipeline = IngestionPipeline(raw_dir=args.input_dir, processed_dir=args.output_dir)

    if args.status:
        summary = pipeline.state_manager.get_summary_stats()
        print("\n=== THỐNG KÊ TRẠNG THÁI INGESTION PIPELINE (state.db) ===")
        print(f"File status counts: {json.dumps(summary['file_status_counts'], indent=2)}")
        print(f"Page type counts:   {json.dumps(summary['page_type_counts'], indent=2)}")
        print("========================================================\n")
        return

    if args.file:
        if not args.file.exists():
            print(f"Lỗi: File '{args.file}' không tồn tại!")
            sys.exit(1)
        pipeline.process_file(args.file, force=args.force, dry_run=args.dry_run)
    else:
        pipeline.process_all(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
