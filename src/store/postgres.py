"""
Hanoi Metro AI Agent - PostgreSQL Keyword Branch & Document Store (Mục 5.4 đề xuất).
- Cấu hình vi_unaccent (parser simple + unaccent): gõ "bao cao van thu" vẫn tìm được "báo cáo văn thư".
- Số hiệu / mã lưu ở mảng ma_trich_xuat có chỉ mục GIN: khớp chính xác tuyệt đối.
- pg_trgm trên so_hieu_norm: tìm gần đúng khi gõ thiếu/sai số hiệu.
- Dòng ngữ cảnh trọng số A, nội dung trọng số B.
Cần: pip install "psycopg[binary]" và POSTGRES_DSN; quyền CREATE EXTENSION cho unaccent, pg_trgm.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from src.indexing.models import Chunk, DocRecord, ParentSection
from src.store.base import Filters, ParsedQuery, Ranked
from src.text.codes import normalize_code
from src.text.normalize import content_terms, fold

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'vi_unaccent') THEN
    CREATE TEXT SEARCH CONFIGURATION vi_unaccent (COPY = simple);
    ALTER TEXT SEARCH CONFIGURATION vi_unaccent ALTER MAPPING FOR word, hword, hword_part WITH unaccent, simple;
  END IF;
END $$;
CREATE TABLE IF NOT EXISTS documents (
  doc_id char(32) PRIMARY KEY, source_file text, so_hieu text, so_hieu_norm text, ngay_ban_hanh date,
  nam int, loai_vb text, don_vi text, trich_yeu text, ho_so text, lang text, pair_id text,
  muc_mat smallint DEFAULT 0, duong_dan text, meta jsonb NOT NULL, chunker_version text, indexed_at timestamptz
);
CREATE TABLE IF NOT EXISTS parents (
  parent_id text PRIMARY KEY, doc_id char(32) REFERENCES documents ON DELETE CASCADE,
  breadcrumb text, noi_dung text, trang_tu int, trang_den int
);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id text PRIMARY KEY, doc_id char(32) REFERENCES documents ON DELETE CASCADE,
  parent_id text, seq int, trang_tu int, trang_den int, breadcrumb text, context text, noi_dung text,
  search_text text, ma_trich_xuat text[], is_table boolean, tokens int,
  tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('vi_unaccent', coalesce(context, '')), 'A') ||
    setweight(to_tsvector('vi_unaccent', coalesce(search_text, '')), 'B')) STORED
);
CREATE TABLE IF NOT EXISTS citations (doc_id char(32) REFERENCES documents ON DELETE CASCADE, cited_so_hieu text);
CREATE TABLE IF NOT EXISTS doc_access (doc_id char(32) NOT NULL, phong text NOT NULL, PRIMARY KEY (doc_id, phong));
CREATE INDEX IF NOT EXISTS ix_doc_access_phong ON doc_access (phong, doc_id);
CREATE TABLE IF NOT EXISTS doc_access_info (doc_id char(32) PRIMARY KEY, nguon text, ghi_chu text, updated timestamptz);
CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS ix_chunks_codes ON chunks USING gin (ma_trich_xuat);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks (doc_id, seq);
CREATE INDEX IF NOT EXISTS ix_documents_so_hieu_trgm ON documents USING gin (so_hieu_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_documents_date ON documents (ngay_ban_hanh, loai_vb);
"""


class PostgresStore:
    name = "keyword"

    def __init__(self, dsn: str):
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as e:
            raise RuntimeError('Thiếu thư viện PostgreSQL: pip install "psycopg[binary]" psycopg_pool') from e
        self._pool = ConnectionPool(dsn, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, open=True)
        with self._pool.connection() as c:
            c.execute(_SCHEMA)

    # ------------------------------------------------------------------ ghi
    def upsert_document(self, doc: DocRecord, parents: Sequence[ParentSection], chunks: Sequence[Chunk]) -> None:
        with self._pool.connection() as c, c.transaction():
            c.execute("DELETE FROM documents WHERE doc_id = %s", (doc.doc_id,))
            c.execute(
                "INSERT INTO documents VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (doc.doc_id, doc.source_file, doc.so_hieu, normalize_code(doc.so_hieu) if doc.so_hieu else "",
                 doc.ngay_ban_hanh or None, doc.nam, doc.loai_vb, doc.don_vi, doc.trich_yeu, doc.ho_so, doc.lang,
                 doc.pair_id, doc.muc_mat, doc.source_path, json.dumps(doc.to_dict(), ensure_ascii=False),
                 doc.chunker_version, datetime.now()),
            )
            with c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO parents VALUES (%s,%s,%s,%s,%s,%s)",
                    [(p.parent_id, p.doc_id, p.breadcrumb, p.text, p.page_from, p.page_to) for p in parents],
                )
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, parent_id, seq, trang_tu, trang_den, breadcrumb, context, "
                    "noi_dung, search_text, ma_trich_xuat, is_table, tokens) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [(ch.chunk_id, ch.doc_id, ch.parent_id, ch.seq, ch.page_from, ch.page_to, ch.breadcrumb, ch.context,
                      ch.text, ch.search_text[len(ch.context):], ch.codes, ch.is_table, ch.tokens) for ch in chunks],
                )
                cur.executemany("INSERT INTO citations VALUES (%s,%s)", [(doc.doc_id, x) for x in doc.cites])

    def delete_document(self, doc_id: str) -> None:
        with self._pool.connection() as c:
            c.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    def index(self, doc: DocRecord, chunks: Sequence[Chunk]) -> None:
        return None

    def delete(self, doc_id: str) -> None:
        return None

    # ------------------------------------------------------------------ đọc
    def indexed_version(self, doc_id: str) -> Optional[str]:
        with self._pool.connection() as c:
            row = c.execute("SELECT chunker_version FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
        return row["chunker_version"] if row else None

    def get_document(self, doc_id: str) -> Optional[DocRecord]:
        with self._pool.connection() as c:
            row = c.execute("SELECT meta FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
        return DocRecord(**row["meta"]) if row else None

    def get_documents(self, doc_ids: Sequence[str]) -> Dict[str, DocRecord]:
        if not doc_ids:
            return {}
        with self._pool.connection() as c:
            rows = c.execute("SELECT meta FROM documents WHERE doc_id = ANY(%s)", (list(doc_ids),)).fetchall()
        return {r["meta"]["doc_id"]: DocRecord(**r["meta"]) for r in rows}

    def find_documents_by_code(self, code: str, limit: int = 5) -> List[DocRecord]:
        norm = normalize_code(code)
        with self._pool.connection() as c:
            rows = c.execute("SELECT meta FROM documents WHERE so_hieu_norm = %s LIMIT %s", (norm, limit)).fetchall()
            if not rows:
                rows = c.execute(
                    "SELECT meta FROM documents WHERE so_hieu_norm %% %s ORDER BY similarity(so_hieu_norm, %s) DESC LIMIT %s",
                    (norm, norm, limit),
                ).fetchall()
        return [DocRecord(**r["meta"]) for r in rows]

    def list_documents(self, filters: Filters, keyword: str = "", limit: int = 200, offset: int = 0) -> Tuple[List[DocRecord], int]:
        where, args = self._filter_sql(filters, alias="")
        for term in content_terms(keyword):
            where.append("unaccent(lower(coalesce(trich_yeu,'') || ' ' || coalesce(so_hieu,'') || ' ' || source_file)) LIKE %s")
            args.append(f"%{fold(term)}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._pool.connection() as c:
            total = c.execute(f"SELECT COUNT(*) AS n FROM documents {clause}", args).fetchone()["n"]
            rows = c.execute(
                f"SELECT meta FROM documents {clause} ORDER BY ngay_ban_hanh NULLS LAST, source_file LIMIT %s OFFSET %s",
                [*args, limit, offset],
            ).fetchall()
        return [DocRecord(**r["meta"]) for r in rows], total

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self._pool.connection() as c:
            rows = c.execute("SELECT * FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),)).fetchall()
        return {r["chunk_id"]: self._row_to_chunk(r) for r in rows}

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        with self._pool.connection() as c:
            rows = c.execute("SELECT * FROM chunks WHERE doc_id = %s ORDER BY seq", (doc_id,)).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_parents(self, parent_ids: Sequence[str]) -> Dict[str, ParentSection]:
        if not parent_ids:
            return {}
        with self._pool.connection() as c:
            rows = c.execute("SELECT * FROM parents WHERE parent_id = ANY(%s)", (list(parent_ids),)).fetchall()
        return {
            r["parent_id"]: ParentSection(r["parent_id"], r["doc_id"].strip(), r["breadcrumb"], r["noi_dung"], r["trang_tu"], r["trang_den"])
            for r in rows
        }

    @staticmethod
    def _row_to_chunk(r: dict) -> Chunk:
        return Chunk(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"].strip(), parent_id=r["parent_id"], seq=r["seq"],
            page_from=r["trang_tu"], page_to=r["trang_den"], breadcrumb=r["breadcrumb"], context=r["context"],
            text=r["noi_dung"], search_text=r["context"] + r["search_text"], codes=list(r["ma_trich_xuat"] or []),
            is_table=bool(r["is_table"]), tokens=r["tokens"],
        )

    # ------------------------------------------------------------------ tìm kiếm
    def search(self, query: ParsedQuery, filters: Filters, k: int) -> Ranked:
        where, args = self._filter_sql(filters, alias="d.")
        clause = (" AND " + " AND ".join(where)) if where else ""
        ordered: List[str] = []
        with self._pool.connection() as c:
            if query.codes:
                rows = c.execute(
                    f"SELECT ch.chunk_id FROM chunks ch JOIN documents d USING (doc_id) "
                    f"WHERE ch.ma_trich_xuat && %s{clause} ORDER BY ch.seq LIMIT %s",
                    [query.codes, *args, k],
                ).fetchall()
                ordered.extend(r["chunk_id"] for r in rows)
                if not rows:
                    for code in query.codes:
                        for doc in self.find_documents_by_code(code, 3):
                            ordered.extend(ch.chunk_id for ch in self.get_document_chunks(doc.doc_id)[:3])
            tsquery = self._websearch(query)
            if tsquery:
                rows = c.execute(
                    f"SELECT ch.chunk_id, ts_rank_cd(ch.tsv, q) AS diem FROM chunks ch JOIN documents d USING (doc_id), "
                    f"websearch_to_tsquery('vi_unaccent', %s) q WHERE ch.tsv @@ q{clause} ORDER BY diem DESC LIMIT %s",
                    [tsquery, *args, k],
                ).fetchall()
                ordered.extend(r["chunk_id"] for r in rows)
        return [(cid, rank) for rank, cid in enumerate(dict.fromkeys(ordered), start=1)][:k]

    @staticmethod
    def _websearch(query: ParsedQuery) -> str:
        parts = list(query.terms)
        for exp in query.expansions:
            parts.extend(f'"{x}"' for x in [exp.matched, *exp.alternatives])
        return " OR ".join(dict.fromkeys(p.replace("-", " ") for p in parts if p))

    @staticmethod
    def _filter_sql(f: Filters, alias: str) -> Tuple[List[str], list]:
        where: List[str] = [f"{alias}muc_mat <= %s"]
        args: list = [f.max_muc_mat]
        for col, values in (("doc_id", f.doc_ids), ("loai_vb", f.loai_vb), ("don_vi", f.don_vi)):
            if values:
                where.append(f"{alias}{col} = ANY(%s)")
                args.append(list(values))
        if f.year_from:
            where.append(f"{alias}nam >= %s")
            args.append(f.year_from)
        if f.year_to:
            where.append(f"{alias}nam <= %s")
            args.append(f.year_to)
        if f.lang:
            where.append(f"{alias}lang = %s")
            args.append(f.lang)
        if f.huong_vb:
            raise NotImplementedError("Bộ lọc huong_vb chưa có cho PostgreSQL (bảng doc_fields mới có trên SQLite)")
        if f.access is not None:
            parts = []
            if f.access.phong:
                parts.append(f"{alias}doc_id IN (SELECT doc_id FROM doc_access WHERE phong = ANY(%s))")
                args.append(sorted(f.access.phong))
            if f.access.doc_ids:
                parts.append(f"{alias}doc_id = ANY(%s)")
                args.append(sorted(f.access.doc_ids))
            where.append("(" + " OR ".join(parts) + ")" if parts else "false")
        return where, args

    def set_access_labels(self, doc_id: str, phong: Sequence[str], nguon: str = "", ghi_chu: str = "") -> None:
        with self._pool.connection() as c, c.transaction():
            c.execute("DELETE FROM doc_access WHERE doc_id = %s", (doc_id,))
            for p in dict.fromkeys(phong):
                c.execute("INSERT INTO doc_access VALUES (%s, %s)", (doc_id, p))
            c.execute("INSERT INTO doc_access_info VALUES (%s, %s, %s, now()) ON CONFLICT (doc_id) DO UPDATE "
                      "SET nguon = EXCLUDED.nguon, ghi_chu = EXCLUDED.ghi_chu, updated = now()", (doc_id, nguon, ghi_chu))

    def access_labels(self, doc_ids: Sequence[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {d: [] for d in doc_ids}
        with self._pool.connection() as c:
            for r in c.execute("SELECT doc_id, phong FROM doc_access WHERE doc_id = ANY(%s)", (list(doc_ids),)).fetchall():
                out[r["doc_id"]].append(r["phong"])
        return out

    # ------------------------------------------------------------------ metadata bổ sung: chưa cài đặt
    # Bảng doc_fields / doc_relations / doc_tasks mới có trên SQLite (src/store/sqlite.py). Khi chuyển sang PostgreSQL:
    # cài đặt các hàm dưới theo giao diện DocumentStore (src/store/base.py) và thêm điều kiện huong_vb vào bộ lọc.
    def _enrich_missing(self, *_, **__):
        raise NotImplementedError("Metadata bổ sung (src.indexing.enrich) chưa có cho PostgreSQL; dùng KEYWORD_BACKEND=sqlite")

    set_doc_fields = doc_fields = set_doc_relations = doc_relations = relations_to = _enrich_missing
    set_doc_tasks = doc_tasks = _enrich_missing
