"""
Hanoi Metro AI Agent - Runtime Assembly.
Ghép các thành phần theo cấu hình .env: kho văn bản + nhánh từ khóa (SQLite/PostgreSQL), nhánh Qdrant
(server, hoặc Qdrant nhúng khi chưa chạy Docker), bộ máy tìm kiếm, trợ lý và bộ nạp chỉ mục.
"""

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from src.config import AppSettings, settings as default_settings
from src.store.base import DocumentStore

logger = logging.getLogger("hanoi_metro.runtime")


@dataclass
class Runtime:
    settings: AppSettings

    @cached_property
    def store(self) -> DocumentStore:
        r = self.settings.retrieval
        if r.keyword_backend == "postgres":
            from src.store.postgres import PostgresStore
            if not r.postgres_dsn:
                raise RuntimeError("KEYWORD_BACKEND=postgres nhưng chưa đặt POSTGRES_DSN")
            return PostgresStore(r.postgres_dsn)
        from src.store.sqlite import SQLiteStore
        return SQLiteStore(r.sqlite_path)

    @cached_property
    def qdrant(self):
        from qdrant_client import QdrantClient
        try:
            client = self.settings.get_qdrant_client()
            client.get_collections()
            logger.info(f"Kết nối Qdrant server {self.settings.qdrant.url}")
            return client
        except Exception as e:
            path = self.settings.retrieval.qdrant_local_path
            path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Không kết nối được Qdrant server ({e}); dùng Qdrant nhúng tại {path} (một tiến trình).")
            return QdrantClient(path=str(path))

    @cached_property
    def vector(self):
        from src.indexing.embedder import HybridEmbedder
        from src.store.vector import QdrantBranch
        return QdrantBranch(
            self.qdrant,
            HybridEmbedder(strict=True),
            self.settings.retrieval.chunks_collection,
            prefetch=self.settings.retrieval.branch_k,
        )

    @cached_property
    def engine(self):
        from src.retrieval.engine import SearchConfig, SearchEngine
        from src.text.abbrev import AbbreviationDictionary
        r = self.settings.retrieval
        return SearchEngine(
            store=self.store,
            branches=[self.vector, self.store],
            abbrev=AbbreviationDictionary.load(r.synonym_dict_path),
            reranker=self._reranker(),
            config=SearchConfig(rrf_k=r.rrf_k, branch_k=r.branch_k, max_per_doc=r.max_chunks_per_doc,
                                final_k=r.final_k, parent_expand=r.parent_expand),
        )

    def _reranker(self) -> Optional[object]:
        mode = self.settings.retrieval.rerank.lower()
        if mode == "off":
            return None
        from src.retrieval.hybrid_search import BGERerankerWrapper
        wrapper = BGERerankerWrapper()
        if not wrapper.is_flag_reranker:
            if mode == "on":
                raise RuntimeError("RERANK=on nhưng chưa cài FlagEmbedding / bge-reranker-v2-m3")
            return None
        return wrapper

    @cached_property
    def blocklist(self):
        from src.indexing.indexer import Blocklist
        return Blocklist.from_dir(self.settings.retrieval.blocklist_dir)

    @cached_property
    def assistant(self):
        from src.agent.assistant import DocumentAssistant
        from src.agent.llm import create_chat_model
        return DocumentAssistant(self.engine, self.store, create_chat_model(), self.settings.agent,
                                 outbound_guard=self.blocklist.assert_allowed)

    @cached_property
    def indexer(self):
        from src.indexing.indexer import DocumentIndexer
        from src.indexing.structured import ChunkingParams, StructuredChunker
        params = ChunkingParams(**self.settings.chunking.model_dump())
        return DocumentIndexer(StructuredChunker(params), self.store, self.vector, self.blocklist)


def build_runtime(app_settings: Optional[AppSettings] = None) -> Runtime:
    return Runtime(app_settings or default_settings)
