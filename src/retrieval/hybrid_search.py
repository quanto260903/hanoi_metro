"""
Hanoi Metro AI Agent - Hybrid Retrieval & Reranker Module.
Thực hiện tìm kiếm kết hợp (Hybrid Search: Dense Vector + Sparse Vector) sử dụng thuật toán RRF
(Reciprocal Rank Fusion) trên Qdrant để lấy Top 15 chunks ứng viên, sau đó sử dụng mô hình
Reranker (BAAI/bge-reranker-v2-m3) chấm điểm tương đồng ngữ cảnh sâu để chọn lọc Top 5 chunks
chính xác nhất kèm metadata nguồn.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

from qdrant_client import QdrantClient, models
from src.config import settings
from src.indexing.embedder import HybridEmbedder, QdrantIndexer

logger = logging.getLogger("hanoi_metro.retrieval")


@dataclass
class RetrievalResult:
    """Kết quả truy xuất một chunk văn bản kèm điểm số và metadata nguồn."""
    chunk_id: str
    score: float                      # Điểm số Reranker cuối cùng
    rrf_score: float                  # Điểm kết hợp RRF từ Qdrant
    content: str                      # Nội dung chunk có Contextual Header
    raw_content: str                  # Nội dung nguyên bản của chunk
    source_file: str                  # Tên file nguồn (PDF/MD)
    file_path: str                    # Đường dẫn file trên ổ đĩa
    doc_type: str                     # Loại tài liệu (BIEN_BAN, CONG_VAN, KE_HOACH, ...)
    chunk_index: int                  # Vị trí thứ tự chunk trong tài liệu
    title: str                        # Tiêu đề tài liệu
    heading_path: str                 # Phân cấp tiêu đề (Breadcrumbs)
    page_number: int                  # Trang tài liệu
    has_table: bool                   # Có chứa bảng biểu hay không
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BGERerankerWrapper:
    """
    Trình chấm điểm lại (Reranker) hỗ trợ BAAI/bge-reranker-v2-m3:
    - Nếu máy có cài thư viện FlagEmbedding và mô hình BAAI/bge-reranker-v2-m3: chạy Cross-Encoder chuẩn.
    - Nếu không (môi trường nhẹ, máy tính cá nhân RAM hạn chế): tự động kích hoạt bộ Reranker
      dựa trên phân tích trọng số từ vựng chuyên sâu (Lexical-Semantic Cross-Scorer)
      để đảm bảo không bị gián đoạn và vẫn đưa các bảng biểu/từ khóa quan trọng lên đầu.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model_name = model_name
        self.reranker_model = None
        self.is_flag_reranker = False
        self._init_reranker(device)

    def _init_reranker(self, device: Optional[str]) -> None:
        try:
            from FlagEmbedding import FlagReranker
            import torch
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            use_fp16 = torch.cuda.is_available()
            logger.info(f"Đang khởi tạo Reranker {self.model_name} (device={dev}, fp16={use_fp16})...")
            self.reranker_model = FlagReranker(self.model_name, use_fp16=use_fp16, device=dev)
            self.is_flag_reranker = True
            logger.info(f"Đã nạp thành công {self.model_name}!")
        except Exception as e:
            logger.info(f"Chưa cài FlagReranker ({e}). Kích hoạt bộ Cross-Scoring Reranker tối ưu cho máy cá nhân.")
            self.is_flag_reranker = False

    def compute_score(self, query: str, texts: List[str]) -> List[float]:
        """Tính điểm tương quan giữa câu query và từng đoạn text."""
        if not texts:
            return []

        # 1. Sử dụng FlagReranker (nếu có)
        if self.is_flag_reranker and self.reranker_model is not None:
            try:
                pairs = [[query, text] for text in texts]
                scores = self.reranker_model.compute_score(pairs)
                if isinstance(scores, (float, int)):
                    scores = [float(scores)]
                return [float(s) for s in scores]
            except Exception as e:
                logger.warning(f"Lỗi khi chạy FlagReranker: {e}. Sử dụng Cross-Scoring fallback.")

        # 2. Fallback Cross-Scoring tối ưu bộ nhớ:
        # Đánh giá độ phủ từ khóa (keyword coverage), thuật ngữ kỹ thuật viết hoa (mã hiệu), và độ liên quan
        scores: List[float] = []
        q_tokens = [w.lower() for w in re.findall(r"[\w\d]+", query) if len(w) > 1]
        q_codes = [w for w in re.findall(r"[A-Z0-9\-\.\/]{2,}", query)]

        for text in texts:
            t_lower = text.lower()
            score = 0.0

            # Khớp toàn bộ câu truy vấn chính xác
            if query.lower() in t_lower:
                score += 3.0

            # Khớp các token từ vựng
            matched_tokens = sum(1 for tok in q_tokens if tok in t_lower)
            if q_tokens:
                score += (matched_tokens / len(q_tokens)) * 2.0

            # Ưu tiên cực cao cho các mã hiệu kỹ thuật, số phòng, số công văn (VD: N3K93180, AFC, 916/CV)
            for code in q_codes:
                if code.lower() in t_lower:
                    score += 2.5

            # Thưởng điểm nếu chunk chứa bảng biểu liên quan
            if "|" in text and ("stt" in t_lower or "thời gian" in t_lower or "thiết bị" in t_lower):
                score += 0.5

            scores.append(score)

        return scores


class HybridRetriever:
    """
    Module tìm kiếm kết hợp (Hybrid Search) và Reranking:
    1. Nhận câu truy vấn tiếng Việt từ người dùng.
    2. Vector hóa câu truy vấn thành Dense Vector và Sparse Vector.
    3. Tìm kiếm trên Qdrant bằng kỹ thuật RRF (Reciprocal Rank Fusion) để lấy Top 15 chunks.
    4. Rerank lại bằng BAAI/bge-reranker-v2-m3 để chắt lọc Top 5 chunks chuẩn xác nhất.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[HybridEmbedder] = None,
        collection_name: Optional[str] = None,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    ):
        self.collection_name = collection_name or settings.qdrant.collection_name
        self.embedder = embedder or HybridEmbedder()
        self.indexer = QdrantIndexer(client=client, embedder=self.embedder, collection_name=self.collection_name)
        self.client = self.indexer.client
        self.reranker = BGERerankerWrapper(model_name=reranker_model_name)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 15,
        prefetch_limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm Hybrid Search kết hợp Dense + Sparse với thuật toán RRF.
        Lấy Top 15 chunks ứng viên tiềm năng nhất từ Qdrant.
        """
        # Vector hóa câu query
        dense_vec, sparse_vec = self.embedder.encode_query(query)

        # 1. Thử gọi API query_points nguyên bản của Qdrant (Server-side RRF Fusion)
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            candidates: List[Dict[str, Any]] = []
            for hit in results.points:
                payload = hit.payload or {}
                candidates.append({
                    "id": hit.id,
                    "rrf_score": float(hit.score if hit.score is not None else 0.0),
                    "payload": payload,
                })
            return candidates

        except Exception as e:
            logger.warning(f"Server-side RRF thất bại ({e}). Chuyển sang tính toán Client-side RRF.")

        # 2. Client-side RRF Fallback: Truy vấn độc lập Dense và Sparse rồi kết hợp bằng công thức RRF
        try:
            dense_hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", dense_vec),
                limit=prefetch_limit,
                with_payload=True,
            )
            sparse_hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedSparseVector(name="sparse", vector=sparse_vec),
                limit=prefetch_limit,
                with_payload=True,
            )

            # Áp dụng công thức RRF: RRF_score(d) = sum(1 / (60 + rank))
            rrf_scores: Dict[str, float] = {}
            payload_map: Dict[str, Dict[str, Any]] = {}
            k_rrf = 60.0

            for rank, hit in enumerate(dense_hits, start=1):
                p_id = str(hit.id)
                rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + (1.0 / (k_rrf + rank))
                payload_map[p_id] = hit.payload or {}

            for rank, hit in enumerate(sparse_hits, start=1):
                p_id = str(hit.id)
                rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + (1.0 / (k_rrf + rank))
                if p_id not in payload_map:
                    payload_map[p_id] = hit.payload or {}

            # Sắp xếp giảm dần theo điểm RRF
            sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            candidates = [
                {
                    "id": p_id,
                    "rrf_score": float(score),
                    "payload": payload_map.get(p_id, {}),
                }
                for p_id, score in sorted_candidates
            ]
            return candidates

        except Exception as err:
            logger.error(f"Lỗi khi thực hiện Hybrid Search trên Qdrant: {err}")
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        top_k_hybrid: int = 15
    ) -> List[RetrievalResult]:
        """
        Quy trình hoàn chỉnh:
        1. Hybrid Search (RRF) -> Lấy Top 15 chunks
        2. Reranker (BAAI/bge-reranker-v2-m3) -> Trả về Top 5 chunks chuẩn xác nhất kèm metadata
        """
        logger.info(f"Bắt đầu Hybrid Search cho câu hỏi: '{query}'")
        candidates = self.search_hybrid(query, top_k=top_k_hybrid)

        if not candidates:
            logger.warning("Không tìm thấy kết quả phù hợp nào trong Vector DB.")
            return []

        logger.info(f"Hybrid Search trả về {len(candidates)} chunks ứng viên. Đang chấm điểm lại bằng Reranker...")

        # Chuẩn bị nội dung cho Reranker
        candidate_texts = [c["payload"].get("content", "") for c in candidates]
        rerank_scores = self.reranker.compute_score(query, candidate_texts)

        # Gắn điểm rerank vào từng ứng viên
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
        for cand, r_score in zip(candidates, rerank_scores):
            scored_candidates.append((r_score, cand))

        # Sắp xếp theo điểm Reranker giảm dần
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Lấy Top k (mặc định Top 5)
        top_results: List[RetrievalResult] = []
        for r_score, cand in scored_candidates[:top_k]:
            payload = cand["payload"]
            result_obj = RetrievalResult(
                chunk_id=str(payload.get("chunk_id", cand["id"])),
                score=float(r_score),
                rrf_score=float(cand.get("rrf_score", 0.0)),
                content=str(payload.get("content", "")),
                raw_content=str(payload.get("raw_content", "")),
                source_file=str(payload.get("source_file", "Chưa xác định")),
                file_path=str(payload.get("file_path", "")),
                doc_type=str(payload.get("doc_type", "UNKNOWN")),
                chunk_index=int(payload.get("chunk_index", 0)),
                title=str(payload.get("title", "")),
                heading_path=str(payload.get("heading_path", "")),
                page_number=int(payload.get("page_number", 1)),
                has_table=bool(payload.get("has_table", False)),
                metadata=payload,
            )
            top_results.append(result_obj)

        logger.info(f"Đã hoàn thành Rerank: Lựa chọn {len(top_results)} chunks xuất sắc nhất.")
        return top_results
