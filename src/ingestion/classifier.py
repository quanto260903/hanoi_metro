"""
Hanoi Metro AI Agent - Page-Level Document Classifier.
Phân loại từng trang tài liệu PDF bằng PyMuPDF (fitz) và OpenCV (cv2) thành 3 luồng:
1. DIRECT_TEXT: Trang PDF gốc số hóa, text tiếng Việt chuẩn, ít ảnh (<15%), không bảng phức tạp.
2. STANDARD_OCR: Trang scan đánh máy thuần văn bản, rõ nét, không bảng nhiều cột, không điền tay.
3. VLM: Trang scan có bảng biểu (>= 4 cột hoặc gộp ô), biểu mẫu điền tay/CamScanner, hoặc mờ/nghiêng.
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from src.config import settings

logger = logging.getLogger("hanoi_metro.classifier")

VIETNAMESE_CHARS_PATTERN = re.compile(
    r"[a-zA-Z0-9\s.,;:!?'\"()/\-_àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
    r"ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ%&+=<>@#$*\[\]{}|\\]"
)

FONT_ERROR_CHARS_PATTERN = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\u007f]")

MOBILE_SCAN_KEYWORDS = [
    "camscanner",
    "scanned with camscanner",
    "scanned by tapscanner",
    "được quét bởi camscanner",
    "được quét bằng",
    "adobe scan",
    "vflat",
]


class PageType(str, Enum):
    """Phân loại luồng xử lý cho từng trang tài liệu."""
    DIRECT_TEXT = "DIRECT_TEXT"
    STANDARD_OCR = "STANDARD_OCR"
    VLM = "VLM"


@dataclass
class PageClassificationResult:
    """Kết quả phân tích và phân loại của một trang PDF."""
    page_number: int
    page_type: PageType
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "page_type": self.page_type.value,
            "confidence": round(self.confidence, 3),
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


class PageClassifier:
    """Bộ phân loại trang tài liệu đa tiêu chí kết hợp PyMuPDF và OpenCV (Weighted Scoring)."""

    def __init__(self):
        self.image_ratio_threshold = settings.ingestion.image_ratio_threshold
        self.blur_threshold = settings.ingestion.classifier_blur_threshold
        self.skew_threshold = settings.ingestion.classifier_skew_threshold
        self.min_text_chars = settings.ingestion.min_text_chars_direct
        self.max_font_error_ratio = settings.ingestion.max_font_error_ratio
        self.complex_table_cell_threshold = settings.ingestion.complex_table_cell_threshold

        # Chỉ cho phép bảng cực kỳ đơn giản (<= 3 cột, <= 6 hàng, không gộp ô) chạy OCR thường
        self.max_simple_table_cols = 3
        self.max_simple_table_rows = 6
        self.vlm_score_threshold = 5

    def classify_page(self, page: Any, page_num: int) -> PageClassificationResult:
        """Phân loại một trang tài liệu fitz.Page theo 3 luồng: DIRECT_TEXT, STANDARD_OCR, VLM."""
        if fitz is None:
            raise ImportError(
                "Thư viện PyMuPDF (fitz) chưa được cài đặt. Vui lòng chạy: pip install pymupdf"
            )

        metrics: Dict[str, Any] = {}

        try:
            # 1. Phân tích lớp Text số hóa
            text_metrics = self._analyze_text_layer(page)
            metrics.update(text_metrics)

            # 2. Tỉ lệ diện tích ảnh bitmap trên trang
            image_ratio = self._calculate_image_ratio(page)
            metrics["image_ratio"] = round(image_ratio, 4)

            # 3. Render ảnh grayscale 100 DPI để phân tích thị giác
            img_gray = self._render_page_to_opencv_gray(page, dpi=100)

            # 4. Kiểm tra độ mờ (Blur) & góc nghiêng (Skew)
            blur_metrics = self._detect_blur(img_gray)
            skew_metrics = self._detect_skew(img_gray)
            metrics.update(blur_metrics)
            metrics.update(skew_metrics)

            # 5. Phát hiện cấu trúc bảng (n_rows, n_cols, has_merged)
            table_metrics = self._detect_table_complexity(page, img_gray)
            metrics.update(table_metrics)

            # 6. Phát hiện dấu hiệu chụp điện thoại / CamScanner (bao gồm dải watermark đáy trang)
            mobile_metrics = self._detect_mobile_camscanner(page, img_gray)
            metrics.update(mobile_metrics)

            # 7. Phát hiện biểu mẫu có dòng kẻ chấm điền tay / chữ viết tay
            form_metrics = self._detect_form_and_handwriting(img_gray)
            metrics.update(form_metrics)

            # ------------------------------------------------------------------
            # HỆ THỐNG CHẤM ĐIỂM ĐỘ PHỨC TẠP VLM (WEIGHTED SCORING)
            # ------------------------------------------------------------------
            vlm_score = 0
            vlm_reasons: List[str] = []

            # A. Đánh giá bảng biểu
            if table_metrics["has_complex_table"]:
                vlm_score += 5
                vlm_reasons.append(
                    f"Bảng phức tạp ({table_metrics['n_rows']} hàng x {table_metrics['n_cols']} cột, "
                    f"gộp ô: {table_metrics['has_merged']}, {table_metrics['table_intersections']} điểm giao)"
                )
            elif table_metrics["n_rows"] > 0 and table_metrics["n_cols"] > 0:
                # Có kẻ bảng nhỏ trên trang scan: cộng +3 điểm (dễ dàng đẩy sang VLM nếu kèm form/scan điện thoại)
                vlm_score += 3
                vlm_reasons.append(
                    f"Trang scan có kẻ bảng ({table_metrics['n_rows']}x{table_metrics['n_cols']})"
                )

            # B. Đánh giá biểu mẫu điền tay / dòng chấm (....)
            if form_metrics["has_handwritten_form"]:
                vlm_score += 4
                vlm_reasons.append(form_metrics["form_reason"])

            # C. Đánh giá độ mờ (mờ nặng +5, mờ nhẹ +3)
            lap_var = blur_metrics["laplacian_variance"]
            if lap_var < (self.blur_threshold * 0.6):
                vlm_score += 5
                vlm_reasons.append(f"Trang scan rất mờ (Laplacian: {lap_var} < {self.blur_threshold * 0.6:.1f})")
            elif blur_metrics["is_blurry"]:
                vlm_score += 3
                vlm_reasons.append(f"Trang scan hơi mờ (Laplacian: {lap_var} < {self.blur_threshold})")

            # D. Đánh giá góc nghiêng
            abs_skew = abs(skew_metrics["skew_angle"])
            if abs_skew > (self.skew_threshold * 2.0):
                vlm_score += 4
                vlm_reasons.append(f"Trang nghiêng nặng {skew_metrics['skew_angle']}°")
            elif skew_metrics["is_skewed"]:
                vlm_score += 2
                vlm_reasons.append(f"Trang nghiêng nhẹ {skew_metrics['skew_angle']}°")

            # E. Dấu hiệu chụp điện thoại / CamScanner
            if mobile_metrics["has_watermark"]:
                vlm_score += 5
                vlm_reasons.extend(mobile_metrics["mobile_reasons"])
            elif mobile_metrics["has_corner_shadow"] or mobile_metrics["has_footer_badge"]:
                vlm_score += 3
                vlm_reasons.extend(mobile_metrics["mobile_reasons"])

            # F. Lỗi font mã hóa lớp text
            if text_metrics["font_error_ratio"] > self.max_font_error_ratio and text_metrics["char_count"] > 50:
                vlm_score += 5
                vlm_reasons.append(
                    f"Lỗi font chữ lớp text ({text_metrics['font_error_ratio'] * 100:.1f}% ký tự lỗi)"
                )

            metrics["vlm_score"] = vlm_score

            # ------------------------------------------------------------------
            # QUYẾT ĐỊNH LUỒNG XỬ LÝ
            # ------------------------------------------------------------------
            is_direct_text = (
                text_metrics["has_valid_text"]
                and image_ratio < self.image_ratio_threshold
                and vlm_score < self.vlm_score_threshold
                and not table_metrics["has_complex_table"]
            )

            if is_direct_text:
                return PageClassificationResult(
                    page_number=page_num,
                    page_type=PageType.DIRECT_TEXT,
                    confidence=0.96,
                    reasons=[
                        f"Text tiếng Việt chuẩn số hóa ({text_metrics['char_count']} ký tự)",
                        f"Tỉ lệ ảnh thấp ({image_ratio:.2%}), điểm VLM: {vlm_score}/{self.vlm_score_threshold}",
                    ],
                    metrics=metrics,
                )

            if vlm_score >= self.vlm_score_threshold:
                return PageClassificationResult(
                    page_number=page_num,
                    page_type=PageType.VLM,
                    confidence=min(0.98, 0.75 + 0.04 * vlm_score),
                    reasons=[f"[Điểm VLM: {vlm_score}/{self.vlm_score_threshold}] " + "; ".join(vlm_reasons)],
                    metrics=metrics,
                )

            return PageClassificationResult(
                page_number=page_num,
                page_type=PageType.STANDARD_OCR,
                confidence=0.90,
                reasons=[
                    f"Trang scan thuần văn bản rõ nét (Laplacian: {lap_var:.1f}, nghiêng: {skew_metrics['skew_angle']}°)",
                    f"Không bảng phức tạp/điền tay (Điểm VLM: {vlm_score}/{self.vlm_score_threshold})",
                ],
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Lỗi khi phân loại trang {page_num}: {str(e)}", exc_info=True)
            return PageClassificationResult(
                page_number=page_num,
                page_type=PageType.VLM,
                confidence=0.50,
                reasons=[f"Fallback sang VLM do ngoại lệ trong quá trình phân loại: {str(e)}"],
                metrics={"error": str(e)},
            )

    def _analyze_text_layer(self, page: Any) -> Dict[str, Any]:
        """Phân tích lớp text có sẵn trong PDF."""
        text = page.get_text("text") or ""
        stripped_text = text.strip()
        char_count = len(stripped_text)

        if char_count == 0:
            return {
                "char_count": 0,
                "has_valid_text": False,
                "vietnamese_ratio": 0.0,
                "font_error_ratio": 0.0,
            }

        valid_chars = len(VIETNAMESE_CHARS_PATTERN.findall(stripped_text))
        error_chars = len(FONT_ERROR_CHARS_PATTERN.findall(stripped_text))

        vietnamese_ratio = valid_chars / char_count if char_count > 0 else 0.0
        font_error_ratio = error_chars / char_count if char_count > 0 else 0.0

        has_valid_text = (
            char_count >= self.min_text_chars
            and vietnamese_ratio >= 0.85
            and font_error_ratio <= self.max_font_error_ratio
        )

        return {
            "char_count": char_count,
            "has_valid_text": has_valid_text,
            "vietnamese_ratio": round(vietnamese_ratio, 4),
            "font_error_ratio": round(font_error_ratio, 4),
        }

    def _calculate_image_ratio(self, page: Any) -> float:
        """Tính tổng tỉ lệ diện tích các ảnh bitmap chiếm trên trang PDF."""
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        if page_area <= 0:
            return 0.0

        image_list = page.get_images(full=True)
        if not image_list:
            return 0.0

        total_image_area = 0.0
        for img_info in image_list:
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            for r in rects:
                total_image_area += (r.width * r.height)

        return min(1.0, total_image_area / page_area)

    def _render_page_to_opencv_gray(self, page: Any, dpi: int = 100) -> np.ndarray:
        """Render trang PDF thành ảnh grayscale OpenCV."""
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        elif pix.n == 4:
            return cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        return img_np

    def _detect_blur(self, gray: np.ndarray) -> Dict[str, Any]:
        """Đo độ sắc nét bằng phương sai Laplacian."""
        h, w = gray.shape[:2]
        if w > 800:
            scale = 800.0 / w
            resized = cv2.resize(gray, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized = gray

        variance = cv2.Laplacian(resized, cv2.CV_64F).var()
        return {
            "laplacian_variance": round(float(variance), 2),
            "is_blurry": variance < self.blur_threshold,
        }

    def _detect_skew(self, gray: np.ndarray) -> Dict[str, Any]:
        """Phát hiện góc nghiêng của trang tài liệu scan."""
        h, w = gray.shape[:2]
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        lines = cv2.HoughLinesP(
            thresh,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=w // 6,
            maxLineGap=15,
        )

        angles: List[float] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = np.ravel(line)[:4]
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0:
                    angle_deg = np.degrees(np.arctan2(dy, dx))
                    if -30.0 < angle_deg < 30.0:
                        angles.append(angle_deg)

        median_angle = float(np.median(angles)) if angles else 0.0
        return {
            "skew_angle": round(median_angle, 2),
            "is_skewed": abs(median_angle) > self.skew_threshold,
        }

    def _cluster_line_positions(self, coords: np.ndarray, min_gap: int = 12) -> List[float]:
        """Gom cụm các tọa độ đường kẻ gần nhau."""
        if len(coords) == 0:
            return []
        sorted_coords = np.sort(coords)
        clusters = [[float(sorted_coords[0])]]
        for c in sorted_coords[1:]:
            if c - clusters[-1][-1] <= min_gap:
                clusters[-1].append(float(c))
            else:
                clusters.append([float(c)])
        return [float(np.mean(cluster)) for cluster in clusters]

    def _detect_table_complexity(self, page: Any, gray: np.ndarray) -> Dict[str, Any]:
        """Phân tích cấu trúc bảng thực tế: số hàng (n_rows), số cột (n_cols), gộp ô (has_merged)."""
        vector_h_lines = 0
        vector_v_lines = 0
        for draw in page.get_drawings():
            rect = draw.get("rect")
            if rect:
                if rect.width > 50 and rect.height < 4:
                    vector_h_lines += 1
                elif rect.height > 50 and rect.width < 4:
                    vector_v_lines += 1

        h, w = gray.shape[:2]
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        h_kernel_len = max(w // 12, 50)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

        v_kernel_len = max(h // 18, 38)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)

        kernel_dilate = np.ones((3, 3), np.uint8)
        table_intersections = cv2.bitwise_and(
            cv2.dilate(h_lines_img, kernel_dilate, iterations=1),
            cv2.dilate(v_lines_img, kernel_dilate, iterations=1),
        )
        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(table_intersections)
        num_intersections = max(0, num_labels - 1)

        if num_intersections >= 4:
            pts = centroids[1:]
            unique_y = self._cluster_line_positions(pts[:, 1], min_gap=12)
            unique_x = self._cluster_line_positions(pts[:, 0], min_gap=15)
            n_rows = max(0, len(unique_y) - 1)
            n_cols = max(0, len(unique_x) - 1)
        else:
            n_rows = 0
            n_cols = 0

        table_grid = cv2.add(h_lines_img, v_lines_img)
        contours, hierarchy = cv2.findContours(table_grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        cell_count = 0
        if hierarchy is not None:
            for idx, cnt in enumerate(contours):
                if hierarchy[0][idx][3] != -1:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw > 25 and ch > 12 and (cw * ch) < (w * h * 0.7):
                        cell_count += 1

        expected_intersections = (n_rows + 1) * (n_cols + 1) if (n_rows > 0 and n_cols > 0) else 0
        has_merged = False
        if expected_intersections >= 9 and num_intersections < int(expected_intersections * 0.75):
            has_merged = True

        total_grid_lines = vector_h_lines + vector_v_lines

        has_complex_table = (
            (n_cols > self.max_simple_table_cols)
            or (n_rows > self.max_simple_table_rows)
            or has_merged
            or (vector_v_lines > self.max_simple_table_cols + 1 and vector_h_lines > 3)
        )

        return {
            "has_complex_table": has_complex_table,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "has_merged": has_merged,
            "table_cell_count": cell_count,
            "grid_lines_count": total_grid_lines,
            "table_intersections": num_intersections,
        }

    def _detect_form_and_handwriting(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Phát hiện biểu mẫu biên bản có dòng chấm điền tay (.......) và nét chữ viết tay
        bằng cách đếm các cụm chấm ngang nhỏ xếp thẳng hàng trên trang.
        """
        h, w = gray.shape[:2]
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)

        dot_count = 0
        for i in range(1, num_labels):
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # Dấu chấm của dòng điền tay (.......) có kích thước nhỏ 1x1 đến 3x3 px ở 100 DPI
            if 1 <= cw <= 3 and 1 <= ch <= 3 and 1 <= area <= 8:
                dot_count += 1

        # Nếu có nhiều dấu chấm điền tay (>= 35 chấm nhỏ) -> biểu mẫu biên bản điền tay
        has_handwritten_form = dot_count >= 35
        reason = f"Biểu mẫu có dòng điền tay / chữ viết tay ({dot_count} dấu chấm form)" if has_handwritten_form else ""
        return {
            "has_handwritten_form": has_handwritten_form,
            "form_dot_count": dot_count,
            "form_reason": reason,
        }

    def _detect_mobile_camscanner(self, page: Any, gray: np.ndarray) -> Dict[str, Any]:
        """Phát hiện chụp điện thoại / CamScanner (kiểm tra text ẩn, bóng góc và dải watermark đáy trang)."""
        mobile_reasons: List[str] = []
        has_watermark = False
        has_corner_shadow = False
        has_footer_badge = False

        text = (page.get_text("text") or "").lower()
        for kw in MOBILE_SCAN_KEYWORDS:
            if kw in text:
                has_watermark = True
                mobile_reasons.append(f"Chứa nhãn watermark ứng dụng chụp ảnh: '{kw}'")
                break

        h, w = gray.shape[:2]
        margin_y = max(int(h * 0.05), 5)
        margin_x = max(int(w * 0.05), 5)
        cropped = gray[margin_y : h - margin_y, margin_x : w - margin_x]

        ch, cw = cropped.shape[:2]
        corner_size_h = max(ch // 8, 10)
        corner_size_w = max(cw // 8, 10)

        top_left = cropped[0:corner_size_h, 0:corner_size_w].mean()
        top_right = cropped[0:corner_size_h, cw - corner_size_w : cw].mean()
        bottom_left = cropped[ch - corner_size_h : ch, 0:corner_size_w].mean()
        bottom_right = cropped[ch - corner_size_h : ch, cw - corner_size_w : cw].mean()
        center = cropped[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4].mean()

        min_corner = min(top_left, top_right, bottom_left, bottom_right)
        if center > 150 and min_corner < (center * 0.65):
            has_corner_shadow = True
            mobile_reasons.append(
                f"Bóng tối góc sau khi bỏ viền (Góc: {min_corner:.1f} vs Tâm: {center:.1f})"
            )

        # Kiểm tra góc phải dưới cùng (vị trí đặc trưng của nhãn "Được quét bằng CamScanner")
        footer_strip = gray[int(h * 0.93) : h, int(w * 0.55) : w]
        dark_pixels = np.sum(footer_strip < 160)
        if 30 <= dark_pixels <= 1500:
            has_footer_badge = True
            mobile_reasons.append("Phát hiện nhãn watermark góc dưới bên phải (CamScanner/Mobile Scan)")

        return {
            "is_mobile_scan": has_watermark or has_corner_shadow or has_footer_badge,
            "has_watermark": has_watermark,
            "has_corner_shadow": has_corner_shadow,
            "has_footer_badge": has_footer_badge,
            "mobile_reasons": mobile_reasons,
        }
