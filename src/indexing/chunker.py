"""
Hanoi Metro AI Agent - Markdown Semantic Chunker Module.
Thực hiện bóc tách, phân đoạn tài liệu Markdown theo cấu trúc tiêu đề (#, ##, ###),
bảo toàn nguyên vẹn 100% các bảng biểu Markdown (|---|), và áp dụng kỹ thuật
Contextual Header (Anthropic Contextual Retrieval) để đảm bảo từng chunk giữ trọn vẹn ngữ cảnh.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("hanoi_metro.chunker")


@dataclass
class DocumentChunk:
    """Đại diện cho một đoạn văn bản (chunk) sau khi cắt và gắn ngữ cảnh."""
    chunk_id: str
    content: str               # Nội dung hoàn chỉnh bao gồm Contextual Header (dùng để embedding & retrieval)
    raw_content: str           # Nội dung gốc của chunk trước khi ghép Contextual Header
    context_header: str        # Phần Contextual Header chèn ở đầu
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarkdownTablePreserver:
    """Nhận diện và bảo toàn nguyên khối các bảng biểu Markdown."""

    @classmethod
    def is_table_row(cls, line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    @classmethod
    def parse_blocks(cls, text: str) -> List[Tuple[str, str]]:
        """
        Phân rã văn bản thành chuỗi các block ('text' hoặc 'table' hoặc 'code').
        Bảo toàn tuyệt đối 100% nguyên khối bảng biểu Markdown.
        """
        lines = text.splitlines()
        blocks: List[Tuple[str, str]] = []
        current_lines: List[str] = []
        state = "text"  # 'text', 'table', 'code'

        for line in lines:
            stripped = line.strip()

            # Xử lý khối fenced code block (ví dụ ```markdown ... ```)
            if stripped.startswith("```"):
                if state == "code":
                    current_lines.append(line)
                    blocks.append(("code", "\n".join(current_lines)))
                    current_lines = []
                    state = "text"
                    continue
                else:
                    if current_lines:
                        blocks.append((state, "\n".join(current_lines)))
                        current_lines = []
                    state = "code"
                    current_lines.append(line)
                    continue

            if state == "code":
                current_lines.append(line)
                continue

            # Xử lý khối Table
            if cls.is_table_row(line):
                if state != "table":
                    if current_lines:
                        blocks.append((state, "\n".join(current_lines)))
                        current_lines = []
                    state = "table"
                current_lines.append(line)
            else:
                if state == "table":
                    # Kết thúc table
                    if current_lines:
                        blocks.append(("table", "\n".join(current_lines)))
                        current_lines = []
                    state = "text"
                current_lines.append(line)

        if current_lines:
            blocks.append((state, "\n".join(current_lines)))

        return [(b_type, b_content.strip()) for b_type, b_content in blocks if b_content.strip()]


class MarkdownDocumentChunker:
    """
    Trình phân tách tài liệu Markdown thông minh cho hệ thống RAG Hanoi Metro:
    - Đọc metadata từ YAML frontmatter (nếu có).
    - Tự động nhận diện loại tài liệu (doc_type).
    - Tách theo cấu trúc cây tiêu đề (Headings H1, H2, H3, H4).
    - Bảo toàn tuyệt đối các bảng biểu (Markdown tables), không bao giờ cắt đôi hàng/bảng.
    - Chèn Contextual Header ở đầu mỗi chunk.
    - Gắn đầy đủ metadata theo chuẩn: source_file, file_path, doc_type, chunk_index.
    """

    HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    PAGE_TAG_REGEX = re.compile(r"<!--\s*TRANG\s+(\d+)\s*(?:\[(.*?)\])?\s*-->", re.IGNORECASE)

    def __init__(
        self,
        max_chunk_size: int = 1200,      # Ký tự tối đa cho mỗi chunk (khoảng 300-400 từ)
        min_chunk_size: int = 150,       # Ngưỡng tối thiểu gộp chunk nhỏ
        chunk_overlap: int = 150,        # Độ chồng lấn giữa các chunk dài (nếu cần tách đoạn)
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def infer_doc_type(filename: str, title: str, text: str) -> str:
        """Suy luận loại tài liệu nội bộ đường sắt dựa trên tiêu đề và nội dung."""
        combined = f"{filename} {title} {text[:500]}".lower()
        if any(kw in combined for kw in ["biên bản", "bien ban", "kiểm đếm"]):
            return "BIEN_BAN"
        elif any(kw in combined for kw in ["kế hoạch", "ke hoach"]):
            return "KE_HOACH"
        elif any(kw in combined for kw in ["công văn", "cong van", "cv-", "số:"]):
            return "CONG_VAN"
        elif any(kw in combined for kw in ["tờ trình", "to trinh"]):
            return "TO_TRINH"
        elif any(kw in combined for kw in ["báo cáo", "bao cao"]):
            return "BAO_CAO"
        elif any(kw in combined for kw in ["quy chuẩn", "quy định", "hướng dẫn"]):
            return "QUY_DINH"
        return "TAI_LIEU_KY_THUAT"

    def parse_frontmatter(self, text: str) -> Tuple[Dict[str, Any], str]:
        """Tách phần YAML frontmatter và nội dung Markdown chính một cách an toàn."""
        frontmatter: Dict[str, Any] = {}
        content = text

        if text.startswith("---"):
            match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", text, re.DOTALL)
            if match:
                fm_text = match.group(1)
                content = match.group(2).strip()

                # Thử parse bằng pyyaml trước
                try:
                    loaded = yaml.safe_load(fm_text)
                    if isinstance(loaded, dict):
                        frontmatter = loaded
                except Exception:
                    # Parse dòng đơn giản bằng regex nếu YAML có syntax lỗi
                    for line in fm_text.splitlines():
                        kv = re.match(r"^([a-zA-Z0-9_\-]+)\s*:\s*(.*)$", line.strip())
                        if kv:
                            k = kv.group(1).strip()
                            v = kv.group(2).strip().strip('"').strip("'")
                            if k not in frontmatter:
                                frontmatter[k] = v

        return frontmatter, content

    def _split_into_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Phân rã văn bản Markdown thành các Section phân cấp dựa vào tiêu đề (#, ##, ###)
        và theo dõi số trang qua thẻ ghi chú HTML (<!-- TRANG x -->).
        """
        lines = content.splitlines()
        sections: List[Dict[str, Any]] = []

        current_heading_stack: List[Tuple[int, str]] = []  # [(level, title)]
        current_lines: List[str] = []
        current_page = 1
        current_route_type = "UNKNOWN"

        def commit_section():
            if not current_lines:
                return
            section_text = "\n".join(current_lines).strip()
            if section_text:
                headings = [h[1] for h in current_heading_stack]
                sections.append({
                    "headings": headings,
                    "heading_path": " > ".join(headings) if headings else "",
                    "text": section_text,
                    "page": current_page,
                    "route_type": current_route_type,
                })
            current_lines.clear()

        for line in lines:
            # Kiểm tra thẻ đổi trang
            page_match = self.PAGE_TAG_REGEX.search(line)
            if page_match:
                current_page = int(page_match.group(1))
                if page_match.group(2):
                    current_route_type = page_match.group(2).strip()
                continue

            # Bỏ qua các đường kẻ phân cách comment HTML trang
            if line.strip().startswith("<!-- ===") or line.strip().endswith("=== -->"):
                continue

            # Kiểm tra tiêu đề Markdown
            heading_match = self.HEADING_REGEX.match(line)
            if heading_match:
                # Tiêu đề mới -> commit section trước đó
                commit_section()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Cập nhật stack phân cấp tiêu đề
                while current_heading_stack and current_heading_stack[-1][0] >= level:
                    current_heading_stack.pop()
                current_heading_stack.append((level, title))
                continue

            current_lines.append(line)

        # Commit section cuối cùng
        commit_section()
        return sections

    def _subdivide_section_preserving_tables(self, section_text: str) -> List[str]:
        """
        Chia nhỏ một Section dài mà BẢO TOÀN NGUYÊN VẸN các bảng Markdown.
        Tuyệt đối không cắt ngang giữa các dòng của bảng.
        """
        if len(section_text) <= self.max_chunk_size:
            return [section_text]

        raw_blocks = MarkdownTablePreserver.parse_blocks(section_text)
        if not raw_blocks:
            return [section_text]

        flattened_blocks: List[Tuple[str, str]] = []
        for b_type, b_content in raw_blocks:
            if b_type == "text" and len(b_content) > self.max_chunk_size:
                # Tách text thông thường theo đoạn văn
                paras = [p.strip() for p in b_content.split("\n\n") if p.strip()]
                for p in paras:
                    flattened_blocks.append(("text", p))
            else:
                flattened_blocks.append((b_type, b_content))

        chunks: List[str] = []
        current_chunk_blocks: List[str] = []
        current_len = 0

        for b_type, b_content in flattened_blocks:
            b_len = len(b_content)

            # Nếu là bảng biểu hoặc code block dung lượng lớn, giữ nguyên thành 1 chunk độc lập
            if b_type in ("table", "code") and b_len > self.max_chunk_size * 0.6:
                if current_chunk_blocks:
                    chunks.append("\n\n".join(current_chunk_blocks))
                    current_chunk_blocks = []
                    current_len = 0
                chunks.append(b_content)
                continue

            if current_len + b_len + 2 > self.max_chunk_size and current_chunk_blocks:
                chunks.append("\n\n".join(current_chunk_blocks))
                current_chunk_blocks = [b_content]
                current_len = b_len
            else:
                current_chunk_blocks.append(b_content)
                current_len += b_len + 2

        if current_chunk_blocks:
            chunks.append("\n\n".join(current_chunk_blocks))

        return chunks

    def chunk_markdown_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Thực hiện chunking hoàn chỉnh cho một file Markdown đã tiền xử lý.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        raw_text = file_path.read_text(encoding="utf-8")
        frontmatter, content = self.parse_frontmatter(raw_text)

        # Lấy thông tin cơ bản từ frontmatter hoặc tên file
        source_file = frontmatter.get("source_file") or file_path.name
        doc_title = frontmatter.get("title") or file_path.stem
        raw_source_path = frontmatter.get("source_path") or str(file_path.resolve())
        doc_type = self.infer_doc_type(file_path.name, doc_title, content)

        # Tách cấu trúc theo tiêu đề
        sections = self._split_into_sections(content)
        if not sections:
            # Trường hợp văn bản không có heading nào
            sections = [{
                "headings": [],
                "heading_path": "",
                "text": content,
                "page": 1,
                "route_type": "DIRECT_TEXT",
            }]

        chunks: List[DocumentChunk] = []
        chunk_index = 0

        for sec in sections:
            sec_text = sec["text"]
            headings = sec["headings"]
            heading_path = sec["heading_path"]
            page_num = sec["page"]

            sub_chunks = self._subdivide_section_preserving_tables(sec_text)

            for sub_text in sub_chunks:
                clean_sub_text = sub_text.strip()
                if not clean_sub_text:
                    continue

                # Kỹ thuật Contextual Header:
                # Tự động tạo ngữ cảnh đầu chunk gồm: Tên tài liệu, phân cấp mục cha, trang
                header_parts = [f"Tài liệu: {source_file}"]
                if doc_title and doc_title != source_file:
                    header_parts.append(f"Tiêu đề: {doc_title}")
                if heading_path:
                    header_parts.append(f"Mục: {heading_path}")
                if page_num:
                    header_parts.append(f"Trang: {page_num}")

                context_header = f"[{' | '.join(header_parts)}]\n"
                enriched_content = f"{context_header}\n{clean_sub_text}"

                # Kiểm tra có chứa bảng biểu hay không
                has_table = any(MarkdownTablePreserver.is_table_row(l) for l in clean_sub_text.splitlines()) or "```markdown" in clean_sub_text

                chunk_obj = DocumentChunk(
                    chunk_id=f"{file_path.stem}_chunk_{chunk_index:04d}",
                    content=enriched_content,
                    raw_content=clean_sub_text,
                    context_header=context_header.strip(),
                    metadata={
                        "source_file": source_file,
                        "file_path": str(file_path.resolve()),
                        "raw_source_path": raw_source_path,
                        "doc_type": doc_type,
                        "chunk_index": chunk_index,
                        "title": doc_title,
                        "headings": headings,
                        "heading_path": heading_path,
                        "page_number": page_num,
                        "has_table": has_table,
                        "char_count": len(clean_sub_text),
                    }
                )
                chunks.append(chunk_obj)
                chunk_index += 1

        logger.info(f"Đã chunk file '{file_path.name}': sinh {len(chunks)} chunks (doc_type={doc_type}).")
        return chunks

    def chunk_directory(self, directory_path: Path) -> List[DocumentChunk]:
        """Đọc và chunk toàn bộ file .md trong thư mục data/processed_md/."""
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Thư mục không tồn tại: {dir_path}")

        md_files = sorted(list(dir_path.glob("*.md")))
        all_chunks: List[DocumentChunk] = []

        logger.info(f"Bắt đầu chunking thư mục: {dir_path} ({len(md_files)} file Markdown)")
        for md_file in md_files:
            file_chunks = self.chunk_markdown_file(md_file)
            all_chunks.extend(file_chunks)

        logger.info(f"Hoàn thành chunking toàn bộ: Tổng cộng {len(all_chunks)} chunks từ {len(md_files)} file.")
        return all_chunks
