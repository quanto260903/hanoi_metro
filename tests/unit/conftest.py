import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.document import Page, SourceDocument  # noqa: E402
from src.indexing.structured import StructuredChunker  # noqa: E402
from src.store.sqlite import SQLiteStore  # noqa: E402

PROCESSED = ROOT / "tests" / "fixtures" / "processed_md"  # đóng băng: không phụ thuộc lần bóc tách gần nhất
CV_916 = PROCESSED / "916.250926CV-VT&QLPT. Kế hoạch khảo sát hệ thống AFC.md"
BB_DUY_TAN = PROCESSED / "Biên bản thiết bị DC ở Duy Tân.md"


def make_doc(pages: List[str], name: str = "tai_lieu.pdf", doc_id: str = "a" * 32, path: str = "") -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        source_file=name,
        source_path=path or f"D:/kho/{name}",
        title=name.rsplit(".", 1)[0],
        pages=[Page(i + 1, "DIRECT_TEXT", text) for i, text in enumerate(pages)],
    )


class FakeLLM:
    """LLM là dịch vụ HTTP bên ngoài: thay bằng bản ghi lại prompt và trả lời theo kịch bản."""

    def __init__(self, reply: str = "Trả lời [1].", external: bool = False):
        self.reply = reply
        self.external = external
        self.calls: List[Tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        self.calls.append((system, user))
        return self.reply(user) if callable(self.reply) else self.reply

    def stream(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        text = self.complete(system, user, max_tokens, temperature)
        for i in range(0, len(text), 7):
            yield text[i:i + 7]


@pytest.fixture
def store(tmp_path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "index.sqlite")


@pytest.fixture
def indexed_store(store) -> SQLiteStore:
    chunker = StructuredChunker()
    for path in (CV_916, BB_DUY_TAN):
        doc, parents, chunks = chunker.chunk_file(path)
        store.upsert_document(doc, parents, chunks)
    return store
