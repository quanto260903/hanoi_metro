"""
Hanoi Metro AI Agent - SQLite Keyword Branch & Document Store (môi trường thử nghiệm).
Cùng giao diện với nhánh PostgreSQL: FTS5 (BM25, bỏ dấu) cho toàn văn, bảng mã riêng để khớp chính xác
số hiệu/mã thiết bị, so khớp gần đúng số hiệu gõ thiếu/sai, liệt kê đầy đủ theo metadata.
Đề xuất xếp SQLite FTS5 vào loại "chỉ dùng thử nghiệm" (một luồng ghi, không phân quyền) — khi triển khai
nhiều người dùng, đặt KEYWORD_BACKEND=postgres.
"""

import difflib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from src.indexing.models import Chunk, DocRecord, ParentSection
from src.store.base import DocField, DocRelation, DocTask, Filters, ParsedQuery, Ranked, nguon_rank
from src.text.codes import normalize_code
from src.text.normalize import content_terms, fold

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, source_file TEXT, so_hieu TEXT, so_hieu_norm TEXT, ngay_ban_hanh TEXT,
    nam INTEGER, loai_vb TEXT, don_vi TEXT, ho_so TEXT, lang TEXT, pair_id TEXT, muc_mat INTEGER DEFAULT 0,
    meta_fold TEXT, meta_json TEXT NOT NULL, chunker_version TEXT, indexed_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_documents_date ON documents (ngay_ban_hanh, loai_vb);
CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, breadcrumb TEXT, text TEXT, page_from INTEGER, page_to INTEGER
);
CREATE INDEX IF NOT EXISTS ix_parents_doc ON parents (doc_id);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, parent_id TEXT, seq INTEGER, page_from INTEGER, page_to INTEGER,
    breadcrumb TEXT, context TEXT, text TEXT, search_text TEXT, codes_json TEXT, is_table INTEGER, tokens INTEGER
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks (doc_id, seq);
CREATE TABLE IF NOT EXISTS chunk_codes (code TEXT NOT NULL, chunk_id TEXT NOT NULL, doc_id TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_chunk_codes ON chunk_codes (code);
CREATE INDEX IF NOT EXISTS ix_chunk_codes_doc ON chunk_codes (doc_id);
CREATE TABLE IF NOT EXISTS doc_access (doc_id TEXT NOT NULL, phong TEXT NOT NULL, PRIMARY KEY (doc_id, phong));
CREATE INDEX IF NOT EXISTS ix_doc_access_phong ON doc_access (phong, doc_id);
CREATE TABLE IF NOT EXISTS doc_access_info (doc_id TEXT PRIMARY KEY, nguon TEXT, ghi_chu TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS doc_fields (
    doc_id TEXT NOT NULL, truong TEXT NOT NULL, gia_tri TEXT, nguon TEXT NOT NULL, do_tin_cay TEXT, trang INTEGER,
    phien_ban_bo_trich TEXT, nguoi_duyet TEXT, ghi_chu TEXT, cap_nhat TEXT, PRIMARY KEY (doc_id, truong)
);
CREATE INDEX IF NOT EXISTS ix_doc_fields_value ON doc_fields (truong, gia_tri, doc_id);
CREATE TABLE IF NOT EXISTS doc_relations (
    doc_id TEXT NOT NULL, loai TEXT NOT NULL, so_hieu_dich TEXT, so_hieu_dich_norm TEXT, doc_id_dich TEXT,
    trang INTEGER, nguon TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_doc_relations_doc ON doc_relations (doc_id);
CREATE INDEX IF NOT EXISTS ix_doc_relations_target ON doc_relations (doc_id_dich);
CREATE INDEX IF NOT EXISTS ix_doc_relations_code ON doc_relations (so_hieu_dich_norm);
CREATE TABLE IF NOT EXISTS doc_tasks (
    doc_id TEXT NOT NULL, phong TEXT, nguoi TEXT, noi_dung TEXT, han TEXT, loai_han TEXT, trang INTEGER, nguon TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_doc_tasks_doc ON doc_tasks (doc_id);
CREATE INDEX IF NOT EXISTS ix_doc_tasks_phong ON doc_tasks (phong, han);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, head, body, tokenize = 'unicode61 remove_diacritics 2'
);
"""

def _fts_quote(text: str) -> str:
    return '"' + fold(text).replace('"', '""') + '"'


class SQLiteStore:
    """Kho văn bản + nhánh từ khóa. Mỗi lời gọi mở kết nối riêng nên dùng được từ nhiều luồng."""

    name = "keyword"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._so_hieu_cache: Optional[List[Tuple[str, str]]] = None
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ ghi
    def upsert_document(self, doc: DocRecord, parents: Sequence[ParentSection], chunks: Sequence[Chunk]) -> None:
        with self._write_lock, self._conn() as c:
            self._delete(c, doc.doc_id)
            c.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc.doc_id, doc.source_file, doc.so_hieu, normalize_code(doc.so_hieu) if doc.so_hieu else "",
                 doc.ngay_ban_hanh or None, doc.nam, doc.loai_vb, doc.don_vi, doc.ho_so, doc.lang, doc.pair_id,
                 doc.muc_mat, fold(" ".join([doc.so_hieu, doc.trich_yeu, doc.title, doc.source_file])),
                 json.dumps(doc.to_dict(), ensure_ascii=False), doc.chunker_version, datetime.now().isoformat(timespec="seconds")),
            )
            c.executemany(
                "INSERT INTO parents VALUES (?,?,?,?,?,?)",
                [(p.parent_id, p.doc_id, p.breadcrumb, p.text, p.page_from, p.page_to) for p in parents],
            )
            c.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(ch.chunk_id, ch.doc_id, ch.parent_id, ch.seq, ch.page_from, ch.page_to, ch.breadcrumb, ch.context,
                  ch.text, ch.search_text, json.dumps(ch.codes, ensure_ascii=False), int(ch.is_table), ch.tokens) for ch in chunks],
            )
            c.executemany(
                "INSERT INTO chunk_codes VALUES (?,?,?)",
                [(code, ch.chunk_id, ch.doc_id) for ch in chunks for code in ch.codes],
            )
            c.executemany(
                "INSERT INTO chunks_fts (chunk_id, head, body) VALUES (?,?,?)",
                [(ch.chunk_id, fold(ch.context), fold(ch.search_text[len(ch.context):])) for ch in chunks],
            )
        self._so_hieu_cache = None

    def delete_document(self, doc_id: str) -> None:
        with self._write_lock, self._conn() as c:
            self._delete(c, doc_id)
        self._so_hieu_cache = None

    @staticmethod
    def _delete(c: sqlite3.Connection, doc_id: str) -> None:
        c.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)", (doc_id,))
        for table in ("chunk_codes", "chunks", "parents", "documents"):
            c.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))

    # SearchBranch: nội dung đã được ghi trong upsert_document
    def index(self, doc: DocRecord, chunks: Sequence[Chunk]) -> None:
        return None

    def delete(self, doc_id: str) -> None:
        return None

    # ------------------------------------------------------------------ đọc
    def indexed_version(self, doc_id: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT chunker_version FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row["chunker_version"] if row else None

    def get_document(self, doc_id: str) -> Optional[DocRecord]:
        with self._conn() as c:
            row = c.execute("SELECT meta_json FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return DocRecord(**json.loads(row["meta_json"])) if row else None

    def get_documents(self, doc_ids: Sequence[str]) -> Dict[str, DocRecord]:
        if not doc_ids:
            return {}
        marks = ",".join("?" * len(doc_ids))
        with self._conn() as c:
            rows = c.execute(f"SELECT doc_id, meta_json FROM documents WHERE doc_id IN ({marks})", list(doc_ids)).fetchall()
        return {r["doc_id"]: DocRecord(**json.loads(r["meta_json"])) for r in rows}

    def find_documents_by_code(self, code: str, limit: int = 5) -> List[DocRecord]:
        norm = normalize_code(code)
        with self._conn() as c:
            rows = c.execute("SELECT meta_json FROM documents WHERE so_hieu_norm = ? LIMIT ?", (norm, limit)).fetchall()
        if not rows:
            close = self._fuzzy_so_hieu(norm, limit)
            if close:
                with self._conn() as c:
                    marks = ",".join("?" * len(close))
                    rows = c.execute(f"SELECT meta_json FROM documents WHERE doc_id IN ({marks})", close).fetchall()
        return [DocRecord(**json.loads(r["meta_json"])) for r in rows]

    def list_documents(self, filters: Filters, keyword: str = "", limit: int = 200, offset: int = 0) -> Tuple[List[DocRecord], int]:
        where, args = self._filter_sql(filters, alias="")
        if keyword:
            for term in content_terms(keyword):
                where.append("meta_fold LIKE ?")
                args.append(f"%{fold(term)}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM documents {clause}", args).fetchone()[0]
            rows = c.execute(
                f"SELECT meta_json FROM documents {clause} ORDER BY ngay_ban_hanh IS NULL, ngay_ban_hanh, source_file LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return [DocRecord(**json.loads(r["meta_json"])) for r in rows], total

    def get_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, Chunk]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", list(chunk_ids)).fetchall()
        return {r["chunk_id"]: self._row_to_chunk(r) for r in rows}

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY seq", (doc_id,)).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_parents(self, parent_ids: Sequence[str]) -> Dict[str, ParentSection]:
        if not parent_ids:
            return {}
        marks = ",".join("?" * len(parent_ids))
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM parents WHERE parent_id IN ({marks})", list(parent_ids)).fetchall()
        return {r["parent_id"]: ParentSection(**dict(r)) for r in rows}

    def stats(self) -> Dict[str, int]:
        with self._conn() as c:
            return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("documents", "parents", "chunks")}

    @staticmethod
    def _row_to_chunk(r: sqlite3.Row) -> Chunk:
        d = dict(r)
        d["codes"] = json.loads(d.pop("codes_json") or "[]")
        d["is_table"] = bool(d["is_table"])
        return Chunk(**d)

    # ------------------------------------------------------------------ tìm kiếm
    def search(self, query: ParsedQuery, filters: Filters, k: int) -> Ranked:
        where, args = self._filter_sql(filters, alias="d.")
        clause = (" AND " + " AND ".join(where)) if where else ""
        ordered: List[str] = []

        with self._conn() as c:
            # (1) khớp chính xác số hiệu / mã
            if query.codes:
                marks = ",".join("?" * len(query.codes))
                rows = c.execute(
                    f"SELECT DISTINCT cc.chunk_id FROM chunk_codes cc JOIN documents d ON d.doc_id = cc.doc_id "
                    f"JOIN chunks ch ON ch.chunk_id = cc.chunk_id WHERE cc.code IN ({marks}){clause} ORDER BY ch.seq LIMIT ?",
                    [*query.codes, *args, k],
                ).fetchall()
                ordered.extend(r["chunk_id"] for r in rows)
                # (3) gần đúng khi gõ thiếu/sai số hiệu
                if not rows:
                    for code in query.codes:
                        for doc_id in self._fuzzy_so_hieu(code, 3):
                            rows = c.execute(
                                f"SELECT ch.chunk_id FROM chunks ch JOIN documents d ON d.doc_id = ch.doc_id "
                                f"WHERE ch.doc_id = ?{clause} ORDER BY ch.seq LIMIT 3",
                                [doc_id, *args],
                            ).fetchall()
                            ordered.extend(r["chunk_id"] for r in rows)

            # (2) toàn văn BM25, đã mở rộng viết tắt; ngữ cảnh (head) trọng số 3, nội dung trọng số 1
            match = self._match_expression(query)
            if match:
                rows = c.execute(
                    f"SELECT f.chunk_id FROM chunks_fts f JOIN chunks ch ON ch.chunk_id = f.chunk_id "
                    f"JOIN documents d ON d.doc_id = ch.doc_id WHERE chunks_fts MATCH ?{clause} "
                    f"ORDER BY bm25(chunks_fts, 0.0, 3.0, 1.0) LIMIT ?",
                    [match, *args, k],
                ).fetchall()
                ordered.extend(r["chunk_id"] for r in rows)

        seen: Dict[str, int] = {}
        for cid in ordered:
            if cid not in seen:
                seen[cid] = len(seen) + 1
        return list(seen.items())[:k]

    @staticmethod
    def _match_expression(query: ParsedQuery) -> str:
        parts = [_fts_quote(t) for t in query.terms]
        for exp in query.expansions:
            parts.extend(_fts_quote(x) for x in [exp.matched, *exp.alternatives])
        parts.extend(_fts_quote(code) for code in query.codes)
        return " OR ".join(dict.fromkeys(p for p in parts if p.strip('"')))

    def _fuzzy_so_hieu(self, code: str, limit: int) -> List[str]:
        if self._so_hieu_cache is None:
            with self._conn() as c:
                self._so_hieu_cache = [(r[0], r[1]) for r in c.execute("SELECT doc_id, so_hieu_norm FROM documents WHERE so_hieu_norm != ''")]
        norms = [n for _, n in self._so_hieu_cache]
        close = difflib.get_close_matches(normalize_code(code), norms, n=limit, cutoff=0.8)
        return [doc_id for doc_id, n in self._so_hieu_cache if n in close][:limit]

    @staticmethod
    def _filter_sql(f: Filters, alias: str) -> Tuple[List[str], list]:
        where: List[str] = [f"{alias}muc_mat <= ?"]
        args: list = [f.max_muc_mat]
        for col, values in (("doc_id", f.doc_ids), ("loai_vb", f.loai_vb), ("don_vi", f.don_vi)):
            if values:
                where.append(f"{alias}{col} IN ({','.join('?' * len(values))})")
                args.extend(values)
        if f.year_from:
            where.append(f"{alias}nam >= ?")
            args.append(f.year_from)
        if f.year_to:
            where.append(f"{alias}nam <= ?")
            args.append(f.year_to)
        if f.lang:
            where.append(f"{alias}lang = ?")
            args.append(f.lang)
        if f.huong_vb:
            where.append(f"{alias}doc_id IN (SELECT doc_id FROM doc_fields WHERE truong = 'huong_vb' "
                         f"AND gia_tri IN ({','.join('?' * len(f.huong_vb))}))")
            args.extend(f.huong_vb)
        if f.access is not None:
            # nhãn phòng nằm ở bảng riêng (đổi quyền không phải nạp lại văn bản); văn bản chưa gán nhãn không khớp
            parts, phong, ids = [], sorted(f.access.phong), sorted(f.access.doc_ids)
            if phong:
                parts.append(f"{alias}doc_id IN (SELECT doc_id FROM doc_access WHERE phong IN ({','.join('?' * len(phong))}))")
                args.extend(phong)
            if ids:
                parts.append(f"{alias}doc_id IN ({','.join('?' * len(ids))})")
                args.extend(ids)
            where.append("(" + " OR ".join(parts) + ")" if parts else "0")
        return where, args

    # ------------------------------------------------------------------ nhãn phân quyền
    def set_access_labels(self, doc_id: str, phong: Sequence[str], nguon: str = "", ghi_chu: str = "") -> None:
        with self._write_lock, self._conn() as c:
            c.execute("DELETE FROM doc_access WHERE doc_id = ?", (doc_id,))
            c.executemany("INSERT INTO doc_access VALUES (?, ?)", [(doc_id, p) for p in dict.fromkeys(phong)])
            c.execute("INSERT OR REPLACE INTO doc_access_info VALUES (?, ?, ?, datetime('now'))", (doc_id, nguon, ghi_chu))

    def access_labels(self, doc_ids: Sequence[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {d: [] for d in doc_ids}
        ids = list(doc_ids)
        with self._conn() as c:
            for i in range(0, len(ids), 500):
                part = ids[i:i + 500]
                for r in c.execute(f"SELECT doc_id, phong FROM doc_access WHERE doc_id IN ({','.join('?' * len(part))})", part):
                    out[r[0]].append(r[1])
        return out

    # ------------------------------------------------------------------ metadata bổ sung (src.indexing.enrich)
    @staticmethod
    def _parts(ids: Sequence[str], size: int = 500) -> Iterator[List[str]]:
        ids = list(ids)
        for i in range(0, len(ids), size):
            yield ids[i:i + size]

    def set_doc_fields(self, doc_id: str, fields: Sequence[DocField]) -> List[str]:
        changed: List[str] = []
        with self._write_lock, self._conn() as c:
            for f in fields:
                old = c.execute("SELECT gia_tri, nguon, trang, phien_ban_bo_trich, ghi_chu FROM doc_fields "
                                "WHERE doc_id = ? AND truong = ?", (doc_id, f.truong)).fetchone()
                if old is not None and nguon_rank(old["nguon"]) > nguon_rank(f.nguon):
                    continue                    # không để luật/AI ghi đè giá trị người đã duyệt
                if old is not None and tuple(old) == (f.gia_tri, f.nguon, f.trang, f.phien_ban_bo_trich, f.ghi_chu):
                    continue
                c.execute("INSERT OR REPLACE INTO doc_fields VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))",
                          (doc_id, f.truong, f.gia_tri, f.nguon, f.do_tin_cay, f.trang, f.phien_ban_bo_trich,
                           f.nguoi_duyet, f.ghi_chu))
                changed.append(f.truong)
        return changed

    def doc_fields(self, doc_ids: Sequence[str], truong: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, DocField]]:
        out: Dict[str, Dict[str, DocField]] = {d: {} for d in doc_ids}
        extra, extra_args = "", []
        if truong:
            extra = f" AND truong IN ({','.join('?' * len(truong))})"
            extra_args = list(truong)
        with self._conn() as c:
            for part in self._parts(doc_ids):
                rows = c.execute(f"SELECT * FROM doc_fields WHERE doc_id IN ({','.join('?' * len(part))}){extra}",
                                 [*part, *extra_args]).fetchall()
                for r in rows:
                    out[r["doc_id"]][r["truong"]] = DocField(
                        r["truong"], r["gia_tri"] or "", r["nguon"], r["trang"], r["do_tin_cay"] or "",
                        r["phien_ban_bo_trich"] or "", r["ghi_chu"] or "", r["nguoi_duyet"] or "")
        return out

    def set_doc_relations(self, doc_id: str, relations: Sequence[DocRelation], nguon: str) -> None:
        with self._write_lock, self._conn() as c:
            c.execute("DELETE FROM doc_relations WHERE doc_id = ? AND nguon = ?", (doc_id, nguon))
            c.executemany("INSERT INTO doc_relations VALUES (?,?,?,?,?,?,?)",
                          [(doc_id, r.loai, r.so_hieu_dich, normalize_code(r.so_hieu_dich) if r.so_hieu_dich else "",
                            r.doc_id_dich or "", r.trang, nguon) for r in relations])

    @staticmethod
    def _row_to_relation(r: sqlite3.Row) -> DocRelation:
        return DocRelation(r["loai"], r["so_hieu_dich"] or "", r["doc_id_dich"] or "", r["trang"], r["nguon"], r["doc_id"])

    def doc_relations(self, doc_ids: Sequence[str]) -> List[DocRelation]:
        out: List[DocRelation] = []
        with self._conn() as c:
            for part in self._parts(doc_ids):
                out += [self._row_to_relation(r) for r in c.execute(
                    f"SELECT * FROM doc_relations WHERE doc_id IN ({','.join('?' * len(part))}) ORDER BY doc_id, rowid", part)]
        return out

    def relations_to(self, doc_id: str, so_hieu: str = "") -> List[DocRelation]:
        norm = normalize_code(so_hieu) if so_hieu else ""
        with self._conn() as c:
            rows = c.execute("SELECT * FROM doc_relations WHERE (doc_id_dich = ? OR (? != '' AND so_hieu_dich_norm = ?)) "
                             "AND doc_id != ? ORDER BY rowid", (doc_id, norm, norm, doc_id)).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def set_doc_tasks(self, doc_id: str, tasks: Sequence[DocTask], nguon: str) -> None:
        with self._write_lock, self._conn() as c:
            c.execute("DELETE FROM doc_tasks WHERE doc_id = ? AND nguon = ?", (doc_id, nguon))
            c.executemany("INSERT INTO doc_tasks VALUES (?,?,?,?,?,?,?,?)",
                          [(doc_id, t.phong, t.nguoi, t.noi_dung, t.han, t.loai_han, t.trang, nguon) for t in tasks])

    def doc_tasks(self, doc_ids: Sequence[str]) -> List[DocTask]:
        out: List[DocTask] = []
        with self._conn() as c:
            for part in self._parts(doc_ids):
                for r in c.execute(f"SELECT * FROM doc_tasks WHERE doc_id IN ({','.join('?' * len(part))}) "
                                   "ORDER BY doc_id, rowid", part):
                    out.append(DocTask(r["phong"] or "", r["nguoi"] or "", r["noi_dung"] or "", r["han"] or "",
                                       r["loai_han"] or "", r["trang"], r["nguon"], r["doc_id"]))
        return out
