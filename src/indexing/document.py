"""
Hanoi Metro AI Agent - Processed Markdown Loader.
Đọc file Markdown do Ingestion Pipeline sinh ra (YAML frontmatter + thẻ <!-- TRANG n [LUỒNG] -->)
thành danh sách trang, để các bước làm sạch / chunking biết từng dòng thuộc trang nào.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger("hanoi_metro.document")

_RX_FRONTMATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", re.DOTALL)
_RX_PAGE_TAG = re.compile(r"<!--\s*TRANG\s+(\d+)\s*(?:\[(.*?)\])?\s*-->", re.IGNORECASE)
_RX_RULE_COMMENT = re.compile(r"^\s*<!--\s*=+\s*-->\s*$")


@dataclass
class Page:
    number: int
    route: str
    text: str


@dataclass
class SourceDocument:
    doc_id: str
    source_file: str
    source_path: str
    title: str
    pages: List[Page]
    frontmatter: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return max((p.number for p in self.pages), default=0)


def parse_frontmatter(text: str) -> tuple:
    """Tách YAML frontmatter; frontmatter cũ có dòng note thiếu dấu nháy nên có nhánh đọc từng dòng."""
    match = _RX_FRONTMATTER.match(text)
    if not match:
        return {}, text
    fm_text, body = match.group(1), match.group(2)
    try:
        loaded = yaml.safe_load(fm_text)
        if isinstance(loaded, dict):
            return loaded, body
    except yaml.YAMLError:
        pass
    meta: Dict[str, Any] = {}
    for line in fm_text.splitlines():
        kv = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.+)$", line.strip())
        if kv and kv.group(1) not in meta:
            meta[kv.group(1)] = kv.group(2).strip().strip('"').strip("'")
    return meta, body


def load_markdown(path: Path) -> SourceDocument:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    doc_id = str(meta.get("md5_hash") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{32}", doc_id):
        doc_id = hashlib.md5(raw.encode("utf-8")).hexdigest()
        logger.warning(f"'{path.name}' không có md5_hash của file gốc; dùng MD5 nội dung Markdown làm doc_id.")

    pages: List[Page] = []
    number, route, buf = 1, "UNKNOWN", []
    for line in body.splitlines():
        if _RX_RULE_COMMENT.match(line):
            continue
        tag = _RX_PAGE_TAG.search(line)
        if tag:
            if any(x.strip() for x in buf):
                pages.append(Page(number, route, "\n".join(buf)))
            number, route, buf = int(tag.group(1)), (tag.group(2) or "UNKNOWN").strip(), []
            continue
        buf.append(line)
    if any(x.strip() for x in buf):
        pages.append(Page(number, route, "\n".join(buf)))

    source_file = str(meta.get("source_file") or path.with_suffix(".pdf").name)
    return SourceDocument(
        doc_id=doc_id,
        source_file=source_file,
        source_path=str(meta.get("source_path") or path.resolve()),
        title=str(meta.get("title") or path.stem),
        pages=pages,
        frontmatter=meta,
    )
