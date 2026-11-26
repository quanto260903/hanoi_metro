"""
Hanoi Metro AI Agent - Document Ingestion Parsers.
Cung cấp 3 parser chuyên biệt cho 3 luồng trang tài liệu:
1. DirectTextParser: Bóc tách trực tiếp từ vector text của PyMuPDF/pdfplumber sang Markdown.
2. StandardOCRParser: Nhận dạng quang học bằng Docling / RapidOCR sang Markdown.
3. VLMParser: Render ảnh 300 DPI và gọi Vision Language Model qua OpenAI-compatible API
   với System Prompt chuyên sâu về tài liệu kỹ thuật & hành chính Đường sắt Hà Nội.
"""

import io
import re
import base64
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict

warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")

import httpx

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.config import settings
from src.ingestion.classifier import PageType

logger = logging.getLogger("hanoi_metro.parsers")

# ==============================================================================
# SYSTEM PROMPT CHUYÊN BIỆT CHO VLM - TÀI LIỆU HÀ NỘI METRO
# ==============================================================================
# Prompt đầy đủ dùng cho model >=7b hoặc khi num_ctx >= 8192
HANOI_METRO_VLM_SYSTEM_PROMPT = """Bạn là chuyên gia OCR tài liệu kỹ thuật & hành chính Hanoi Metro.
Chuyển đổi hình ảnh trang tài liệu sang Markdown. TUÂN THỦ BẮT BUỘC:

1. BẢNG BIỂU → Markdown Table (| cột |). Giữ nguyên STT, số liệu, đơn vị, thành tiền.
2. THUẬT NGỮ: AFC(TVM/AG/BOM/PCA/IC-Card), OCC, Tuyến 2A(Cát Linh-HĐ), Tuyến 3.1(Nhổn-GHN), Cisco/HPE/Dell/Fortigate/APC.
3. CHỮ VIẾT TAY → [Viết tay: ...]. CHỮ KÝ → [Chữ ký: ...]. DẤU ĐỎ → [Dấu đỏ: ...].
4. Thể thức hành chính: Quốc hiệu, Tiêu ngữ, Số/KH văn bản, Ngày tháng, Trích yếu.
5. Chỉ xuất Markdown thuần túy. KHÔNG thêm lời chào hay giải thích."""

# Prompt rút gọn dùng cho model 3b hoặc khi context < 8192
HANOI_METRO_VLM_SYSTEM_PROMPT_COMPACT = """Bạn là chuyên gia OCR tài liệu Hanoi Metro.
Chuyển ảnh trang PDF sang Markdown:
- Bảng biểu → Markdown Table (| cột |), giữ nguyên số liệu.
- Chữ viết tay → [Viết tay: ...]. Chữ ký → [Chữ ký: ...]. Dấu đỏ → [Dấu đỏ: ...].
- Thuật ngữ: AFC, OCC, TVM, Tuyến 2A, Tuyến 3.1, Cisco, HPE, Dell, Fortigate.
- Chỉ xuất Markdown. Không thêm lời giải thích."""


class BaseParser(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các Document Parser."""

    @abstractmethod
    def parse(self, page: Any, page_num: int, pdf_path: Optional[str] = None) -> str:
        """
        Bóc tách trang PDF thành chuỗi Markdown hoàn chỉnh.
        Args:
            page: Trang fitz.Page hiện tại.
            page_num: Số thứ tự trang (1-based).
            pdf_path: Đường dẫn tới file PDF (tùy chọn cho các parser đọc trực tiếp từ disk).
        Returns:
            Nội dung Markdown của trang.
        """
        pass


class DirectTextParser(BaseParser):
    """
    Parser bóc tách trực tiếp lớp text số hóa từ PyMuPDF và pdfplumber.
    Dành cho trang DIRECT_TEXT: Nhanh, không tốn chi phí gọi API, giữ nguyên độ nét chữ.
    """

    def parse(self, page: Any, page_num: int, pdf_path: Optional[str] = None) -> str:
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) chưa được cài đặt.")

        logger.info(f"[DirectTextParser] Đang bóc tách text trực tiếp trang {page_num}")
        lines_output: List[str] = []

        try:
            # 1. Thử bóc tách bảng bằng pdfplumber nếu có và file tồn tại
            tables_md: List[str] = []
            if pdfplumber is not None and pdf_path is not None:
                try:
                    tables_md = self._extract_tables_with_pdfplumber(pdf_path, page_num)
                except Exception as ex:
                    logger.debug(f"pdfplumber không tìm thấy bảng ở trang {page_num}: {ex}")

            # 2. Bóc tách text blocks theo thứ tự đọc tự nhiên từ PyMuPDF
            # page.get_text("blocks") trả về (x0, y0, x1, y1, text, block_no, block_type)
            # block_type == 0: Text block; block_type == 1: Image block
            blocks = page.get_text("blocks")
            # Sắp xếp các block theo trục y từ trên xuống dưới, rồi x từ trái sang phải
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

            for b in sorted_blocks:
                block_type = b[6]
                if block_type != 0:
                    continue  # Bỏ qua image blocks

                block_text = b[4].strip()
                if not block_text:
                    continue

                formatted_block = self._format_text_block(block_text)
                lines_output.append(formatted_block)

            # Ghép nội dung
            content = "\n\n".join(lines_output)

            # Nếu có bảng biểu bóc tách được từ pdfplumber mà chưa có trong text, ghép bổ sung
            if tables_md:
                content += "\n\n### Bảng dữ liệu trích xuất:\n\n" + "\n\n".join(tables_md)

            return content.strip()

        except Exception as e:
            logger.error(f"[DirectTextParser] Lỗi khi xử lý trang {page_num}: {str(e)}", exc_info=True)
            # Fallback đơn giản lấy text thô
            raw_text = page.get_text("text") or ""
            return raw_text.strip()

    def _format_text_block(self, text: str) -> str:
        """Định dạng heading và danh sách từ khối văn bản."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return ""

        first_line = lines[0]
        # Nhận diện tiêu đề viết hoa hoặc bắt đầu bằng số thứ tự La Mã
        is_heading_candidate = (
            first_line.isupper()
            and len(first_line) > 5
            and len(first_line) < 120
            and not first_line.endswith(".")
        )

        if is_heading_candidate and len(lines) == 1:
            return f"### {first_line}"

        # Xử lý gạch đầu dòng và danh sách
        formatted_lines = []
        for line in lines:
            # Bullet points
            if line.startswith(("-", "•", "+", "*")):
                formatted_lines.append(f"- {line.lstrip('-•+* ')}")
            # Số thứ tự: 1., 2., hoặc 1.1, a)
            elif re.match(r"^(\d+(\.\d+)*|[a-z]\))\s+", line):
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        return " ".join(formatted_lines)

    def _extract_tables_with_pdfplumber(self, pdf_path: str, page_num: int) -> List[str]:
        """Bóc tách bảng biểu và định dạng sang GitHub Markdown Table bằng pdfplumber."""
        tables_md = []
        with pdfplumber.open(pdf_path) as pdf:
            if page_num - 1 < len(pdf.pages):
                plumber_page = pdf.pages[page_num - 1]
                tables = plumber_page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    md_table = self._convert_matrix_to_md_table(table)
                    if md_table:
                        tables_md.append(md_table)
        return tables_md

    def _convert_matrix_to_md_table(self, table: List[List[Any]]) -> str:
        """Chuyển đổi ma trận 2D thành Markdown table."""
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
            cleaned_table.append(cleaned_row)

        if not cleaned_table:
            return ""

        header = cleaned_table[0]
        separator = ["---"] * len(header)
        rows = cleaned_table[1:]

        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rows:
            # Đảm bảo số cột đồng nhất
            padded_row = row + [""] * (len(header) - len(row))
            md_lines.append("| " + " | ".join(padded_row[: len(header)]) + " |")

        return "\n".join(md_lines)


class StandardOCRParser(BaseParser):
    """
    Parser sử dụng công nghệ OCR truyền thống (Docling / RapidOCR / Tesseract).
    Dành cho trang STANDARD_OCR: Trang scan đánh máy ngay ngắn, bố cục phẳng, không bảng phức tạp.
    """

    def __init__(self):
        self._ocr_engine = None
        self._init_ocr_engine()

    def _init_ocr_engine(self) -> None:
        """Khởi tạo engine OCR có sẵn trong môi trường."""
        # Thử nạp rapidocr_onnxruntime (gọn nhẹ, hỗ trợ Tiếng Việt & Tiếng Anh tốt)
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr_engine = RapidOCR()
            logger.info("[StandardOCRParser] Đã kích hoạt RapidOCR engine.")
            return
        except ImportError:
            pass

        # Thử nạp Docling nếu đã cài
        try:
            from docling.document_converter import DocumentConverter
            self._ocr_engine = DocumentConverter()
            logger.info("[StandardOCRParser] Đã kích hoạt Docling DocumentConverter engine.")
            return
        except ImportError:
            pass

        logger.warning(
            "[StandardOCRParser] Chưa cài đặt RapidOCR hoặc Docling. Sẽ sử dụng fallback trích xuất."
        )

    def parse(self, page: Any, page_num: int, pdf_path: Optional[str] = None) -> str:
        logger.info(f"[StandardOCRParser] ?ang OCR trang {page_num}...")

        # Trường hợp đã cấu hình Docling
        if hasattr(self._ocr_engine, "convert") and pdf_path:
            try:
                result = self._ocr_engine.convert(pdf_path)
                doc = result.document
                md_text = doc.export_to_markdown()
                if md_text and len(md_text.strip()) > 30:
                    return md_text.strip()
            except Exception as e:
                logger.warning(f"Docling OCR thất bại ở trang {page_num}: {e}")

        # Trường hợp dùng RapidOCR
        if hasattr(self._ocr_engine, "__call__"):
            try:
                # Render trang thành ảnh numpy array 200 DPI
                zoom = 200 / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                import numpy as np
                import cv2
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                if pix.n == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

                ocr_result, _ = self._ocr_engine(img_np)
                if ocr_result:
                    # ocr_result là list các [dt_boxes, text, score]
                    extracted_lines = []
                    # Sắp xếp các dòng theo tọa độ y trên xuống dưới
                    sorted_lines = sorted(ocr_result, key=lambda item: item[0][0][1])
                    for item in sorted_lines:
                        text = item[1].strip()
                        if text:
                            extracted_lines.append(text)
                    return "\n\n".join(extracted_lines)
            except Exception as e:
                logger.error(f"RapidOCR thất bại ở trang {page_num}: {e}", exc_info=True)

        # Fallback: Trích xuất PyMuPDF text thô
        logger.info(f"[StandardOCRParser] ?ang OCR trang {page_num}...")
        raw_text = page.get_text("text") or ""
        if raw_text.strip():
            return raw_text.strip()

        return f"<!-- [StandardOCRParser]: Không thể nhận diện nội dung trang {page_num}. Đề xuất chuyển sang VLM. -->"


class VLMParser(BaseParser):
    """
    Parser sử dụng mô hình Vision Language Model (VLM) tự host nội bộ qua Ollama.
    Hỗ trợ: qwen2.5vl:3b, qwen2.5-vl:7b, llama3.2-vision:11b, llava:13b, minicpm-v.
    Dành cho trang VLM: Bảng biểu phức tạp (AFC/OCC), scan mờ/nghiêng, chữ viết tay, CamScanner.

    Cơ chế chống lỗi context:
    - Tự động dùng prompt rút gọn khi num_ctx < 8192 hoặc model 3b.
    - Render ảnh theo IMAGE_DPI_VLM (mặc định 150 DPI) thay vì 300 DPI.
    - Truyền num_ctx vào Ollama options để ép mở rộng context window.
    - Không fallback sang OpenAI endpoint khi lỗi là context overflow (tránh lặp vô ích).
    """

    # Ngưỡng num_ctx tối thiểu để dùng prompt đầy đủ
    _FULL_PROMPT_MIN_CTX = 8192

    def __init__(self):
        self.provider = settings.vlm.provider.lower()
        self.base_url = settings.vlm.base_url.rstrip("/")
        self.api_key = settings.vlm.api_key
        self.model_name = settings.vlm.model_name
        self.temperature = settings.vlm.temperature
        self.max_tokens = settings.vlm.max_tokens
        self.timeout_seconds = float(settings.vlm.timeout_seconds)
        self.num_ctx = settings.vlm.num_ctx
        # Dùng IMAGE_DPI_VLM (nhỏ hơn) để render cho VLM — ít token hơn IMAGE_DPI của classifier
        self.dpi = settings.ingestion.image_dpi_vlm

        # Chọn System Prompt phù hợp dựa vào context window và tên model
        is_small_model = any(tag in self.model_name.lower() for tag in ["3b", "1b", "2b"])
        if self.num_ctx < self._FULL_PROMPT_MIN_CTX or is_small_model:
            self.system_prompt = HANOI_METRO_VLM_SYSTEM_PROMPT_COMPACT
            logger.info(
                f"[VLMParser] Dùng prompt RÚT GỌN (num_ctx={self.num_ctx}, model={self.model_name})"
            )
        else:
            self.system_prompt = HANOI_METRO_VLM_SYSTEM_PROMPT
            logger.info(
                f"[VLMParser] Dùng prompt ĐẦY ĐỦ (num_ctx={self.num_ctx}, model={self.model_name})"
            )

        # Chuẩn hóa Ollama host URL (loại bỏ hậu tố /v1 nếu có)
        if self.base_url.endswith("/v1"):
            self.ollama_host = self.base_url[:-3]
        else:
            self.ollama_host = self.base_url

        # Khởi tạo OpenAI client cho fallback non-Ollama
        self._openai_client = None
        if OpenAI is not None and not settings.vlm.is_ollama:
            try:
                self._openai_client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout_seconds,
                )
            except Exception:
                pass

        self._check_ollama_readiness()

    def _check_ollama_readiness(self) -> None:
        """Kiểm tra kết nối và sự tồn tại của model trên Ollama server khi khởi tạo."""
        if not settings.vlm.is_ollama:
            return

        try:
            with httpx.Client(timeout=4.0) as client:
                resp = client.get(f"{self.ollama_host}/api/tags")
                if resp.status_code == 200:
                    models_data = resp.json().get("models", [])
                    available_names = [m.get("name") for m in models_data if "name" in m]
                    logger.info(
                        f"[VLMParser] Kết nối thành công tới Ollama tại '{self.ollama_host}'. "
                        f"Models hiện có trên máy: {available_names}"
                    )
                    # Kiểm tra xem model đã được pull về chưa
                    is_available = any(
                        self.model_name in name or name.startswith(self.model_name.split(":")[0])
                        for name in available_names
                    )
                    if not is_available and available_names:
                        logger.warning(
                            f"[VLMParser] Model '{self.model_name}' chưa có trong Ollama! "
                            f"Hãy tải model: ollama run {self.model_name}"
                        )
        except Exception as e:
            logger.warning(
                f"[VLMParser] Chưa thể kết nối tới Ollama tại '{self.ollama_host}' ({e}). "
                f"Đảm bảo Ollama đang chạy ('ollama serve' hoặc 'docker compose up -d')."
            )

    def parse(self, page: Any, page_num: int, pdf_path: Optional[str] = None) -> str:
        logger.info(
            f"[VLMParser] Render {self.dpi} DPI → gửi Ollama ({self.model_name}) | "
            f"num_ctx={self.num_ctx} | trang {page_num}"
        )
        # Render ảnh ở DPI dành cho VLM (thường thấp hơn classifier để tiết kiệm context)
        base64_image = self._render_page_to_base64(page, dpi=self.dpi)
        return self._call_vlm(base64_image, page_num)

    def _render_page_to_base64(self, page: Any, dpi: int = 150) -> str:
        """Render trang PDF thành ảnh PNG Base64."""
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=retry_if_exception_type((httpx.RequestError, ConnectionError, ValueError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_vlm(self, base64_image: str, page_num: int) -> str:
        """Điều phối cuộc gọi đến Ollama. Lỗi context overflow không retry."""
        if settings.vlm.is_ollama:
            return self._call_ollama_native_api(base64_image, page_num)
        else:
            return self._call_openai_compat_api(base64_image, page_num)

    def _call_ollama_native_api(self, base64_image: str, page_num: int) -> str:
        """
        Gọi trực tiếp Ollama REST API (/api/chat).
        Truyền num_ctx vào options để ép Ollama mở rộng context window cho model.
        Ảnh được đặt ở cấp message.images[], chuẩn native Ollama multimodal.
        """
        user_prompt = (
            f"Bóc tách toàn bộ nội dung trang {page_num} sang Markdown. "
            f"Giữ nguyên bảng biểu (| cột |), ghi chú chữ viết tay và thuật ngữ: "
            f"AFC, OCC, Tuyến 2A, Tuyến 3.1, Cisco, HPE, Dell, Fortigate."
        )

        chat_url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    # Ollama native multimodal: ảnh đặt vào images[] cùng cấp với role/content
                    "role": "user",
                    "content": user_prompt,
                    "images": [base64_image],
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                # Mở rộng context window — đây là key fix cho lỗi "exceeds context size"
                "num_ctx": self.num_ctx,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(chat_url, json=payload)

            # Phân loại lỗi rõ ràng để tránh retry vô ích
            if response.status_code == 404:
                raise RuntimeError(
                    f"Model '{self.model_name}' chưa cài trên Ollama. "
                    f"Tải model: ollama run {self.model_name}"
                )
            if response.status_code == 400:
                # Lấy chi tiết lỗi từ body để log
                try:
                    err_body = response.json()
                    err_msg = err_body.get("error", response.text)
                except Exception:
                    err_msg = response.text
                # Lỗi context overflow: tăng num_ctx hoặc giảm DPI
                if "context" in err_msg.lower() or "token" in err_msg.lower():
                    raise RuntimeError(
                        f"[Context Overflow] Ollama 400: {err_msg}. "
                        f"Giải pháp: Tăng VLM_NUM_CTX (hiện={self.num_ctx}) hoặc giảm IMAGE_DPI_VLM (hiện={self.dpi}). "
                        f"Thử: VLM_NUM_CTX=16384 hoặc IMAGE_DPI_VLM=100"
                    )
                raise RuntimeError(f"Ollama 400 Bad Request: {err_msg}")

            response.raise_for_status()
            res_json = response.json()
            message_obj = res_json.get("message", {})
            raw_content = message_obj.get("content", "")

            if not raw_content.strip():
                raise ValueError("Ollama trả về nội dung rỗng.")

            cleaned_md = self._strip_markdown_code_fences(raw_content)
            logger.info(
                f"[VLMParser] Ollama ({self.model_name}) bóc tách trang {page_num} thành công "
                f"({len(cleaned_md)} ký tự)."
            )
            return cleaned_md

        except httpx.ConnectError as conn_err:
            raise ConnectionError(
                f"Không thể kết nối tới Ollama tại '{chat_url}'. "
                f"Đảm bảo Ollama đang chạy. Chi tiết: {conn_err}"
            )
        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            logger.error(f"[VLMParser] Lỗi khi gọi Ollama cho trang {page_num}: {str(e)}")
            raise e

    def _call_openai_compat_api(self, base64_image: str, page_num: int) -> str:
        """Dành cho provider không phải Ollama (OpenAI, vLLM...) qua /v1/chat/completions."""
        if self._openai_client is None:
            raise RuntimeError("OpenAI-compatible client chưa khởi tạo.")

        user_prompt = (
            f"Bóc tách toàn bộ nội dung trang {page_num} sang Markdown. "
            f"Giữ nguyên bảng biểu, ghi chú chữ viết tay, thuật ngữ Hanoi Metro: "
            f"AFC, OCC, Tuyến 2A, Tuyến 3.1, Cisco, HPE, Dell, Fortigate."
        )

        try:
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
            )
            result_content = response.choices[0].message.content or ""
            cleaned_md = self._strip_markdown_code_fences(result_content)
            logger.info(
                f"[VLMParser] OpenAI-compat bóc tách trang {page_num} thành công ({len(cleaned_md)} ký tự)."
            )
            return cleaned_md

        except Exception as e:
            logger.error(f"[VLMParser] Lỗi OpenAI-compat endpoint trang {page_num}: {str(e)}")
            raise e

    def _strip_markdown_code_fences(self, text: str) -> str:
        """Loại bỏ wrapper ```markdown``` nếu VLM sinh ra."""
        cleaned = text.strip()
        if cleaned.startswith("```markdown") and cleaned.endswith("```"):
            cleaned = cleaned[len("```markdown"):-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
        return cleaned


def create_parser(page_type: PageType) -> BaseParser:
    """Factory method khởi tạo parser tương ứng theo luồng phân loại."""
    if page_type == PageType.DIRECT_TEXT:
        return DirectTextParser()
    elif page_type == PageType.STANDARD_OCR:
        return StandardOCRParser()
    elif page_type == PageType.VLM:
        if settings.vlm.is_cloud:
            from src.ingestion.cloud_vlm import CloudVLMParser
            return CloudVLMParser()
        return VLMParser()
    else:
        logger.warning(f"Loại trang không xác định: {page_type}. Mặc định sử dụng DirectTextParser.")
        return DirectTextParser()
