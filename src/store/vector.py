"""
Hanoi Metro AI Agent - Qdrant Semantic Branch (Mục 5.3 đề xuất).
Mỗi chunk con lưu 2 vector bge-m3: dense 1.024 chiều và sparse (trọng số từ vựng), nén int8.
Payload chỉ chứa trường lọc (doc_id, loai_vb, nam, don_vi, muc_mat, lang, pair_id, ho_so); nội dung nằm ở
kho văn bản nên hai nhánh luôn trả cùng một bản chunk. Truy vấn lai dense + sparse gộp RRF ngay trên Qdrant,
cùng một bộ lọc quyền ở cả hai prefetch.
"""

import logging
import uuid
from typing import List, Optional, Sequence

from qdrant_client import QdrantClient, models

from src.indexing.embedder import HybridEmbedder
from src.indexing.models import Chunk, DocRecord
from src.store.base import Filters, ParsedQuery, Ranked

logger = logging.getLogger("hanoi_metro.vector")

_PAYLOAD_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "loai_vb": models.PayloadSchemaType.KEYWORD,
    "don_vi": models.PayloadSchemaType.KEYWORD,
    "lang": models.PayloadSchemaType.KEYWORD,
    "pair_id": models.PayloadSchemaType.KEYWORD,
    "ho_so": models.PayloadSchemaType.KEYWORD,
    "nam": models.PayloadSchemaType.INTEGER,
    "muc_mat": models.PayloadSchemaType.INTEGER,
    "phong": models.PayloadSchemaType.KEYWORD,
    "huong_vb": models.PayloadSchemaType.KEYWORD,
    "trang_thai_hieu_luc": models.PayloadSchemaType.KEYWORD,
}
ENRICH_PAYLOAD_KEYS = ("huong_vb", "trang_thai_hieu_luc")   # metadata bổ sung (src.indexing.enrich) dùng để lọc


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hanoi-metro:{chunk_id}"))


class QdrantBranch:
    name = "vector"

    def __init__(self, client: QdrantClient, embedder: HybridEmbedder, collection: str, prefetch: int = 50, batch_size: int = 16):
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.prefetch = prefetch
        self.batch_size = batch_size
        self._ready = False

    def ensure_collection(self) -> None:
        if self._ready:
            return
        if not self.client.collection_exists(self.collection):
            logger.info(f"Tạo collection '{self.collection}' (dense {self.embedder.dimension} + sparse, int8).")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": models.VectorParams(size=self.embedder.dimension, distance=models.Distance.COSINE, on_disk=True)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
                ),
            )
            for field_name, schema in _PAYLOAD_INDEXES.items():
                self.client.create_payload_index(self.collection, field_name=field_name, field_schema=schema)
        self._ready = True

    def index(self, doc: DocRecord, chunks: Sequence[Chunk], phong: Sequence[str] = ()) -> None:
        self.ensure_collection()
        base = {
            "doc_id": doc.doc_id, "loai_vb": doc.loai_vb, "don_vi": doc.don_vi, "lang": doc.lang,
            "pair_id": doc.pair_id, "ho_so": doc.ho_so, "nam": doc.nam, "muc_mat": doc.muc_mat, "phong": list(phong),
        }
        for i in range(0, len(chunks), self.batch_size):
            batch = list(chunks[i:i + self.batch_size])
            dense, sparse = self.embedder.encode([c.search_text for c in batch])
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector={"dense": d, "sparse": s},
                        payload={**base, "chunk_id": c.chunk_id, "parent_id": c.parent_id, "page_from": c.page_from, "page_to": c.page_to},
                    )
                    for c, d, s in zip(batch, dense, sparse)
                ],
                wait=True,
            )

    def delete(self, doc_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
            ])),
            wait=True,
        )

    def search(self, query: ParsedQuery, filters: Filters, k: int) -> Ranked:
        if not self.client.collection_exists(self.collection):
            return []
        text = query.text
        if query.expansions:
            text += " (" + "; ".join(alt for e in query.expansions for alt in e.alternatives) + ")"
        dense, sparse = self.embedder.encode_query(text)
        flt = self._filter(filters)
        res = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=self.prefetch, filter=flt),
                models.Prefetch(query=sparse, using="sparse", limit=self.prefetch, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=["chunk_id"],
        )
        return [(p.payload["chunk_id"], rank) for rank, p in enumerate(res.points, start=1) if p.payload]

    @staticmethod
    def _filter(f: Filters) -> Optional[models.Filter]:
        must: List[models.Condition] = [models.FieldCondition(key="muc_mat", range=models.Range(lte=f.max_muc_mat))]
        for key, values in (("doc_id", f.doc_ids), ("loai_vb", f.loai_vb), ("don_vi", f.don_vi)):
            if values:
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
        if f.year_from or f.year_to:
            must.append(models.FieldCondition(key="nam", range=models.Range(gte=f.year_from, lte=f.year_to)))
        if f.lang:
            must.append(models.FieldCondition(key="lang", match=models.MatchValue(value=f.lang)))
        if f.huong_vb:
            must.append(models.FieldCondition(key="huong_vb", match=models.MatchAny(any=list(f.huong_vb))))
        if f.access is not None:
            # điểm chưa có nhãn "phong" không khớp MatchAny → văn bản chưa gán nhãn bị chặn
            should: List[models.Condition] = []
            if f.access.phong:
                should.append(models.FieldCondition(key="phong", match=models.MatchAny(any=sorted(f.access.phong))))
            if f.access.doc_ids:
                should.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=sorted(f.access.doc_ids))))
            if not should:
                should.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value="__khong_co_quyen__")))
            must.append(models.Filter(should=should))
        return models.Filter(must=must)

    def set_access_payload(self, doc_id: str, phong: Sequence[str]) -> None:
        """Đổi nhãn phòng của mọi đoạn một văn bản: chỉ sửa payload, không tạo lại vector."""
        if not self.client.collection_exists(self.collection):
            return
        self.client.set_payload(
            collection_name=self.collection, payload={"phong": list(phong)},
            points=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]),
        )

    def set_doc_payload(self, doc_id: str, payload: dict) -> None:
        """Ghi trường lọc bổ sung (huong_vb, trang_thai_hieu_luc) cho mọi đoạn của văn bản; không tạo lại vector."""
        if not payload or not self.client.collection_exists(self.collection):
            return
        self.client.set_payload(
            collection_name=self.collection, payload=dict(payload),
            points=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]),
        )

    def ensure_enrich_indexes(self) -> None:
        if self.client.collection_exists(self.collection):
            for key in ENRICH_PAYLOAD_KEYS:
                self.client.create_payload_index(self.collection, field_name=key, field_schema=models.PayloadSchemaType.KEYWORD)

    def ensure_access_index(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.create_payload_index(self.collection, field_name="phong", field_schema=models.PayloadSchemaType.KEYWORD)
