"""
Hanoi Metro AI Agent - Hybrid Vector Embedder & Qdrant Indexing Module.
Sinh đồng thời Dense Vector (ngữ nghĩa) và Sparse Vector (từ khóa/trọng số từ vựng) cho từng chunk:
- Hỗ trợ thư viện FlagEmbedding (BAAI/bge-m3) chạy local/GPU.
- Hỗ trợ chế độ API Embedding (Ollama / OpenAI-compatible cấu hình trong .env) kết hợp
  BM25 Lexical Sparse Vectorizer tối ưu cho tiếng Việt và máy cá nhân.
- Khởi tạo Qdrant Collection hỗ trợ Hybrid Search (Dense + Sparse) và Upsert dữ liệu kèm đầy đủ Metadata.
"""

import os
import re
import zlib
import math
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient, models
from src.config import settings
from src.indexing.chunker import DocumentChunk

logger = logging.getLogger("hanoi_metro.embedder")


class BM25LexicalSparseVectorizer:
    """
    Bộ mã hóa Sparse Vector từ vựng (Lexical/BM25) thuần Python tối ưu hóa cao:
    - Bóc tách từ tiếng Việt và thuật ngữ kỹ thuật đường sắt (mã thiết bị, số hiệu công văn).
    - Ánh xạ từ khóa sang chỉ số nguyên 31-bit (CRC32 hashing) tương thích chuẩn Qdrant Sparse Index.
    - Tính trọng số từ theo thuật toán BM25 Term Weighting để phản ánh chính xác tần suất và độ hiếm.
    """

    # Regex nhận diện từ tiếng Việt, ký tự chữ và số, mã hiệu kỹ thuật
    TOKEN_REGEX = re.compile(r"[\w\d]+", re.UNICODE)

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 150.0):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len

    @classmethod
    def hash_token(cls, token: str) -> int:
        """Băm từ thành số nguyên dương 31-bit [1, 2^31 - 1] làm Sparse Index."""
        return (zlib.crc32(token.lower().encode("utf-8")) & 0x7FFFFFFF) or 1

    def tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self.TOKEN_REGEX.findall(text) if len(t) > 1]

    def encode(self, text: str) -> models.SparseVector:
        """Mã hóa văn bản thành Qdrant SparseVector."""
        tokens = self.tokenize(text)
        if not tokens:
            return models.SparseVector(indices=[1], values=[0.0])

        doc_len = len(tokens)
        counts = Counter(tokens)

        indices: List[int] = []
        values: List[float] = []

        # BM25 Term Frequency Weighting
        for token, tf in counts.items():
            token_id = self.hash_token(token)
            # Công thức BM25 Term Weight
            weight = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)))
            # Log transform nhẹ để tránh giá trị quá lệch
            score = float(math.log1p(weight))
            indices.append(token_id)
            values.append(score)

        # Sắp xếp indices tăng dần theo yêu cầu của Qdrant SparseVector
        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
        sorted_indices = [p[0] for p in sorted_pairs]
        sorted_values = [p[1] for p in sorted_pairs]

        return models.SparseVector(indices=sorted_indices, values=sorted_values)

    def encode_batch(self, texts: List[str]) -> List[models.SparseVector]:
        return [self.encode(t) for t in texts]


class HybridEmbedder:
    """
    Module sinh Hybrid Vector (Dense + Sparse) cho Hanoi Metro RAG:
    - Nếu có sẵn thư viện FlagEmbedding và mô hình BAAI/bge-m3: sử dụng trực tiếp để sinh cả 2 vector.
    - Nếu không (hoặc cấu hình API qua .env): sử dụng OpenAI/Ollama API cho Dense Vector + BM25 cho Sparse Vector.
    - Hoạt động nhẹ nhàng, tối ưu bộ nhớ RAM/VRAM để chạy ổn định trên máy cá nhân.
    """

    def __init__(self, engine: str = "auto", strict: bool = False):
        self.engine = engine.lower()
        # strict=True: lỗi API embedding thì ném lỗi thay vì lặng lẽ nạp vector băm vào chỉ mục
        self.strict = strict
        self.dimension = settings.embedding.dimension
        self.bge_model = None
        self.lexical_vectorizer = BM25LexicalSparseVectorizer()
        self._init_engine()

    def _init_engine(self) -> None:
        """Khởi tạo engine embedding tương ứng."""
        if self.engine in ("auto", "flag_embedding", "bge-m3"):
            try:
                from FlagEmbedding import BGEM3FlagModel
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                use_fp16 = torch.cuda.is_available()
                logger.info(f"Đang tải mô hình BAAI/bge-m3 qua FlagEmbedding (device={device}, fp16={use_fp16})...")
                self.bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)
                self.engine = "flag_embedding"
                logger.info("Đã nạp thành công BAAI/bge-m3 FlagEmbedding!")
                return
            except Exception as e:
                if self.engine == "flag_embedding":
                    raise RuntimeError(f"Không thể khởi tạo FlagEmbedding: {e}")
                logger.info(f"FlagEmbedding chưa sẵn sàng ({e}). Chuyển sang chế độ API Embedding ({settings.embedding.model_name}) + BM25 Lexical Sparse.")

        self.engine = "api"
        logger.info(f"Sử dụng API Embedding: URL={settings.embedding.base_url}, Model={settings.embedding.model_name}, Dim={self.dimension}")

    def _generate_fallback_dense_vector(self, text: str) -> List[float]:
        """Tạo dense vector dự phòng bằng Feature Hashing chuẩn hóa khi API chưa bật."""
        vec = [0.0] * self.dimension
        words = re.findall(r"[\w\d]+", text.lower())
        if not words:
            return vec
        for word in words:
            h = zlib.crc32(word.encode("utf-8"))
            idx = h % self.dimension
            sign = 1.0 if (h & 0x10000) else -1.0
            vec[idx] += sign

        # Chuẩn hóa L2 norm
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def encode_dense_api(self, texts: List[str]) -> List[List[float]]:
        """Sinh Dense Vector qua OpenAI-compatible API client (Ollama/vLLM/OpenAI)."""
        client = settings.get_embedding_client()
        try:
            response = client.embeddings.create(
                input=texts,
                model=settings.embedding.model_name
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"Embedding API ({settings.embedding.model_name}) lỗi: {e}") from e
            logger.warning(f"Lỗi khi gọi embedding API: {e}. Sử dụng vector mã hóa dự phòng.")
            return [self._generate_fallback_dense_vector(t) for t in texts]

    def encode(self, texts: Union[str, List[str]]) -> Tuple[List[List[float]], List[models.SparseVector]]:
        """
        Sinh đồng thời Dense Vector và Sparse Vector cho danh sách văn bản.
        Trả về: (dense_vectors, sparse_vectors)
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        if not text_list:
            return [], []

        # 1. Chế độ BAAI/bge-m3 qua FlagEmbedding
        if self.engine == "flag_embedding" and self.bge_model is not None:
            try:
                output = self.bge_model.encode(
                    text_list,
                    return_dense=True,
                    return_sparse=True,
                    max_length=4096
                )
                dense_vecs = [v.tolist() for v in output["dense_vecs"]]
                sparse_vecs: List[models.SparseVector] = []

                for weights_dict in output["lexical_weights"]:
                    indices: List[int] = []
                    values: List[float] = []
                    for k, val in weights_dict.items():
                        # Key có thể là token id dạng string hoặc int
                        try:
                            t_idx = int(k)
                        except ValueError:
                            t_idx = (zlib.crc32(str(k).encode("utf-8")) & 0x7FFFFFFF) or 1
                        indices.append(t_idx)
                        values.append(float(val))

                    # Sắp xếp indices theo thứ tự tăng dần
                    sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
                    s_indices = [p[0] for p in sorted_pairs]
                    s_values = [p[1] for p in sorted_pairs]
                    sparse_vecs.append(models.SparseVector(indices=s_indices, values=s_values))

                return dense_vecs, sparse_vecs
            except Exception as e:
                logger.warning(f"Lỗi khi encode bằng FlagEmbedding: {e}. Chuyển sang fallback API.")

        # 2. Chế độ API Embedding + BM25 Lexical Sparse
        dense_vecs = self.encode_dense_api(text_list)
        sparse_vecs = self.lexical_vectorizer.encode_batch(text_list)
        return dense_vecs, sparse_vecs

    def encode_query(self, query: str) -> Tuple[List[float], models.SparseVector]:
        """Sinh Dense và Sparse Vector cho một câu truy vấn."""
        dense_vecs, sparse_vecs = self.encode([query])
        return dense_vecs[0], sparse_vecs[0]


class QdrantIndexer:
    """
    Quản lý khởi tạo Collection và Upsert Hybrid Vectors vào Qdrant.
    Tối ưu hóa bộ nhớ và hỗ trợ kết nối linh hoạt giữa Docker Qdrant hoặc Local Storage.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[HybridEmbedder] = None,
        collection_name: Optional[str] = None
    ):
        self.collection_name = collection_name or settings.qdrant.collection_name
        self.embedder = embedder or HybridEmbedder()
        self.client = client or self._get_resilient_qdrant_client()

    def _get_resilient_qdrant_client(self) -> QdrantClient:
        """
        Khởi tạo Qdrant Client với cơ chế fallback tự động:
        Nếu kết nối tới Qdrant server (Docker/localhost:6333) thất bại,
        tự động chuyển sang lưu trữ file cục bộ tại data/qdrant_storage để không gián đoạn công việc.
        """
        try:
            client = settings.get_qdrant_client()
            # Kiểm tra kết nối nhanh
            client.get_collections()
            logger.info(f"Kết nối thành công tới Qdrant Server tại {settings.qdrant.url}")
            return client
        except Exception as e:
            fallback_dir = settings.project_root / "data" / "qdrant_storage"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Không thể kết nối Qdrant Server ({e}). Chuyển sang chế độ Local Qdrant Storage: {fallback_dir}")
            return QdrantClient(path=str(fallback_dir))

    def create_collection(self, recreate: bool = False) -> None:
        """
        Tạo mới hoặc khởi tạo lại Collection hỗ trợ Hybrid Search (Dense + Sparse).
        """
        exists = False
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
        except Exception:
            pass

        if exists:
            if recreate:
                logger.info(f"Đang xóa collection cũ '{self.collection_name}' để tạo lại...")
                self.client.delete_collection(collection_name=self.collection_name)
            else:
                logger.info(f"Collection '{self.collection_name}' đã tồn tại và sẵn sàng sử dụng.")
                return

        # Cấu hình Dense Vector và Sparse Vector theo chuẩn Qdrant Hybrid Search
        vectors_config = {
            "dense": models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE
            )
        }
        sparse_vectors_config = {
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }

        logger.info(f"Đang tạo Collection '{self.collection_name}' (Dense Dim={self.embedder.dimension}, Sparse=Enabled)...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )
        logger.info(f"Đã tạo thành công Hybrid Collection '{self.collection_name}'!")

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 16
    ) -> int:
        """
        Mã hóa và Upsert danh sách chunks vào Qdrant theo từng batch tối ưu bộ nhớ.
        """
        if not chunks:
            logger.warning("Không có chunk nào để upsert.")
            return 0

        self.create_collection(recreate=False)

        total_chunks = len(chunks)
        total_upserted = 0
        logger.info(f"Bắt đầu upsert {total_chunks} chunks vào Qdrant (batch_size={batch_size})...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            # Sinh Dense & Sparse Vectors
            dense_vecs, sparse_vecs = self.embedder.encode(texts)

            points: List[models.PointStruct] = []
            for chunk_obj, dense_v, sparse_v in zip(batch, dense_vecs, sparse_vecs):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_obj.chunk_id))

                # Chuẩn bị metadata payload
                payload = {
                    "chunk_id": chunk_obj.chunk_id,
                    "content": chunk_obj.content,
                    "raw_content": chunk_obj.raw_content,
                    "context_header": chunk_obj.context_header,
                    **chunk_obj.metadata,
                }

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_v,
                        "sparse": sparse_v,
                    },
                    payload=payload,
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            total_upserted += len(points)
            logger.info(f"Đã upsert: {total_upserted}/{total_chunks} chunks.")

        logger.info(f"Hoàn thành xuất sắc: Đã index toàn bộ {total_upserted} chunks vào '{self.collection_name}'.")
        return total_upserted
