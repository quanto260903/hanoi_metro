"""
Hanoi Metro AI Agent - Chạy thử tổng duyệt (dress rehearsal) trước khi chạy toàn kho.

Chọn mẫu đại diện theo thư mục (tỉ lệ theo số trang), chạy đủ chuỗi
    trích text trên máy → gửi API (VLM) → ghép Markdown → nạp chỉ mục → kiểm tra tìm kiếm / hỏi đáp,
đo thời gian, lỗi, chi phí, dung lượng từng bước, kiểm tra chất lượng đầu ra và an toàn (tài liệu mật),
rồi ngoại suy cho toàn kho. Kết quả: data/rehearsal/report.md (+ samples.csv).

Chạy: python -m src.ingestion.rehearsal [--files 60] [--api-pages 350] [--seed 2026] [--no-llm]
"""

import argparse
import csv
import json
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

from src.config import settings
from src.ingestion.cloud_vlm import CloudVLMParser
from src.ingestion.corpus import Corpus, CorpusStore, FilePlan, load_plan

ROOT = Path(os.getenv("CORPUS_ROOT", r"D:\40GB"))
RESULTS = Path(os.getenv("CORPUS_RESULTS_DIR", r"D:\40GB\_phan_tich_chi_phi_VLM\ket_qua"))
OUT = settings.project_root / "data" / "rehearsal"

# ký tự đặc trưng của font TCVN3/VNI cũ hiển thị sai (vd. "Céng hßa x· héi") – dấu hiệu lớp chữ hỏng
_LEGACY = set("µ¸¶·¹¨»¾¼½Æ©ÇÊÈÉË®ÌÐÎÏÑªÒÕÓÔÖ×ÝØÜÞßãáâä«åèæçé¬êíëìîïóñòôõ­öùø÷úûþüýÿ")
_VI = re.compile(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)


def folder_key(path: str) -> str:
    parts = path.replace("/", "\\").split("\\")
    return "\\".join(parts[:2]) if len(parts) > 2 else parts[0]


def pick_sample(manifest: List[dict], n_files: int, api_cap: int, seed: int, exclude: set) -> List[dict]:
    """Tỉ lệ theo số trang mỗi thư mục, mỗi thư mục lớn ít nhất 1 file, + 1 file dài chủ yếu text gốc."""
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in manifest:
        if r["md5"] not in exclude and int(r["pages"] or 0) > 0:
            groups[folder_key(r["path"])].append(r)
    total_pages = sum(int(r["pages"]) for g in groups.values() for r in g)
    quota = {k: max(1 if len(g) >= 20 else 0, round(n_files * sum(int(r["pages"]) for r in g) / total_pages)) for k, g in groups.items()}

    chosen: List[dict] = []
    api_used = 0

    def api_pages(r):
        return int(r["n_vlm"] or 0) + int(r["n_standard_ocr"] or 0)

    long_text = [r for g in groups.values() for r in g if 150 <= int(r["pages"]) <= 400 and api_pages(r) <= 25]
    if long_text:
        r = rng.choice(long_text)
        chosen.append(r)
        api_used += api_pages(r)
    for key in sorted(quota, key=lambda k: -quota[k]):
        cands = groups[key][:]
        rng.shuffle(cands)
        got = 0
        for r in cands:
            if got >= quota[key]:
                break
            if r in chosen or int(r["pages"]) > 120:
                continue
            if api_used + api_pages(r) > api_cap:
                continue
            chosen.append(r)
            api_used += api_pages(r)
            got += 1
    return chosen


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


class Peak:
    def __init__(self):
        self.proc = psutil.Process()
        self.peak = 0

    def tick(self) -> None:
        self.peak = max(self.peak, self.proc.memory_info().rss)


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Chạy thử tổng duyệt trước khi chạy toàn kho")
    ap.add_argument("--files", type=int, default=60)
    ap.add_argument("--api-pages", type=int, default=350)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-llm", action="store_true", help="Bỏ bước hỏi đáp qua LLM")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    peak = Peak()
    log: List[str] = []
    issues: List[str] = []

    def note(line: str) -> None:
        print(line, flush=True)
        log.append(line)

    # ------------------------------------------------------------------ chọn mẫu
    manifest = list(csv.DictReader(open(RESULTS / "ingest_manifest.csv", encoding="utf-8-sig")))
    store = CorpusStore(settings.project_root / "data" / "corpus.db")
    already = {m for m, st in store.file_status().items() if st == "assembled"}
    sample = pick_sample(manifest, args.files, args.api_pages, args.seed, already)
    md5s = {r["md5"] for r in sample}
    with open(OUT / "samples.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)
    all_plans = load_plan(RESULTS)
    plans: Dict[str, FilePlan] = {m: all_plans[m] for m in md5s}
    routes = Counter(r for p in plans.values() for r in p.routes.values())
    note(f"Mẫu: {len(plans)} file, {sum(p.total_pages for p in plans.values())} trang, phân luồng {dict(routes)}, "
         f"{len({folder_key(r['path']) for r in sample})} thư mục")

    parser = CloudVLMParser()
    corpus = Corpus(ROOT, store, parser, settings.project_root / "data" / "corpus_md", os.getenv("CORPUS_OCR_VIA", "vlm"))

    # ------------------------------------------------------------------ an toàn: tài liệu mật
    bl = parser.blocklist
    blocked_in_manifest = [r["path"] for r in manifest if bl.is_blocked(r["md5"])]
    with sqlite3.connect(settings.project_root / "data" / "vlm_ledger.db") as c:
        sent = {r[0] for r in c.execute("SELECT DISTINCT md5 FROM vlm_pages")}
    leaked = [m for m in sent if bl.is_blocked(m)]
    note(f"Danh sách chặn: {len(bl.blocked)} md5, cấu hình hợp lệ={getattr(bl, 'configured', True)}; "
         f"có trong danh sách nạp: {len(blocked_in_manifest)}; đã từng gửi API: {len(leaked)}")
    if blocked_in_manifest or leaked or not getattr(bl, "configured", True):
        issues.append("AN TOÀN: tài liệu mật lọt vào danh sách nạp hoặc đã gửi API – DỪNG, kiểm tra ngay")

    # ------------------------------------------------------------------ 1. trích text trên máy
    t0 = time.perf_counter()
    local_stats = corpus.run_local(plans, args.workers)
    t_local = time.perf_counter() - t0
    peak.tick()
    local_pages = local_stats["pages"]
    note(f"[1] Trích trên máy: {local_stats} trong {t_local:.1f}s")

    direct_lens: List[int] = []
    legacy_pages: List[str] = []
    empty_direct: List[str] = []
    no_diacritics: List[str] = []
    for plan in plans.values():
        local = store.local_pages(plan.md5)
        for p, route in plan.routes.items():
            if route != "DIRECT_TEXT" or p not in local:
                continue
            text = local[p] or ""
            direct_lens.append(len(text))
            ref = f"{Path(plan.path).name} tr.{p + 1}"
            letters = [ch for ch in text if ch.isalpha()]
            if len(text.strip()) < 30:
                empty_direct.append(ref)
            elif letters and sum(ch in _LEGACY for ch in letters) / len(letters) > 0.03:
                legacy_pages.append(ref)
            elif len(letters) > 200 and not _VI.search(text) and re.search(r"\b(cua|nhung|duoc|cong ty)\b", text.lower()):
                no_diacritics.append(ref)

    # ------------------------------------------------------------------ 2. gửi API
    pending = corpus.pending_api_pages(plans)
    before = parser.ledger.stats()
    t0 = time.perf_counter()
    api_stats = corpus.run_vlm_sync(plans, args.workers) if pending else {"pages": 0, "errors": 0, "stopped_budget": 0}
    t_api = time.perf_counter() - t0
    after = parser.ledger.stats()
    peak.tick()
    api_cost = after["cost_usd"] - before["cost_usd"]
    note(f"[2] Gửi API: {len(pending)} trang chờ → {api_stats} trong {t_api:.1f}s, chi {api_cost:.4f} USD")
    if api_stats.get("errors"):
        issues.append(f"API: {api_stats['errors']} trang lỗi (xem log)")
    if api_stats.get("stopped_budget"):
        issues.append("API: dừng vì chạm trần ngân sách VLM_BUDGET_USD")

    rows = []
    with sqlite3.connect(settings.project_root / "data" / "vlm_ledger.db") as c:
        marks = ",".join("?" * len(md5s))
        rows = c.execute(f"SELECT md5, page, markdown, in_tokens, out_tokens, cost_usd, raw FROM vlm_pages "
                         f"WHERE model=? AND md5 IN ({marks})", [parser.model, *md5s]).fetchall()
    path_of = {p.md5: Path(p.path).name for p in plans.values()}
    cap_hits = [f"{path_of[m]} tr.{pg}" for m, pg, _, _, out, _, _ in rows if out >= parser.max_tokens]
    unreadable = [f"{path_of[m]} tr.{pg}" for m, pg, md, *_ in rows if len(re.sub(r"\[[^\]]*\]|\s", "", md or "")) < 15]
    loop_cut = [f"{path_of[m]} tr.{pg}" for m, pg, md, _, _, _, raw in rows if raw and md and len(raw) > 1.5 * len(md) + 500]
    outs = [r[4] for r in rows]

    # ------------------------------------------------------------------ 3. ghép Markdown
    t0 = time.perf_counter()
    asm = corpus.assemble(plans)
    t_asm = time.perf_counter() - t0
    note(f"[3] Ghép Markdown: {asm} trong {t_asm:.1f}s")
    if asm.get("incomplete"):
        issues.append(f"Ghép: {asm['incomplete']} file thiếu trang (chưa có kết quả API)")

    # ------------------------------------------------------------------ 4. nạp chỉ mục
    from src.runtime import build_runtime
    rt = build_runtime()
    md_paths = [Path(p) for m, p in store_out_paths(store, md5s) if p]
    t0 = time.perf_counter()
    report = rt.indexer.index_paths(md_paths)
    t_idx = time.perf_counter() - t0
    peak.tick()
    note(f"[4] Nạp chỉ mục: {report.summary()} trong {t_idx:.1f}s")
    if report.failed:
        issues.append(f"Chỉ mục: {len(report.failed)} file lỗi: {list(report.failed.items())[:3]}")

    docs = rt.store.get_documents(list(md5s))
    profiles = Counter(d.ho_so for d in docs.values())
    admin_like = [d for d in docs.values() if re.search(r"CV|QĐ|TB|TTr|BC|GM|KH", Path(d.source_file).stem)]
    so_hieu_ok = sum(1 for d in admin_like if d.so_hieu)
    date_ok = sum(1 for d in admin_like if d.ngay_ban_hanh)
    chunk_tokens: List[int] = []
    for d in docs.values():
        chunk_tokens += [c.tokens for c in rt.store.get_document_chunks(d.doc_id)]
    chunk_tokens.sort()
    tok_p95 = chunk_tokens[int(len(chunk_tokens) * 0.95)] if chunk_tokens else 0

    # ------------------------------------------------------------------ 5. tìm kiếm tự kiểm
    code_hits, code_total, title_hits, title_total = 0, 0, 0, 0
    misses: List[str] = []
    lat: List[float] = []
    for d in docs.values():
        if d.so_hieu:
            code_total += 1
            res = rt.engine.search(d.so_hieu)
            lat.append(res.latency_ms)
            if res.evidence and res.evidence[0].doc.doc_id == d.doc_id:
                code_hits += 1
            else:
                misses.append(f"số hiệu {d.so_hieu} ({d.source_file})")
        subject = d.trich_yeu or d.title
        words = subject.split()
        if len(words) >= 5:
            title_total += 1
            res = rt.engine.search(" ".join(words[:14]))
            lat.append(res.latency_ms)
            if d.doc_id in {ev.doc.doc_id for ev in res.evidence[:5]}:
                title_hits += 1
            else:
                misses.append(f"trích yếu '{' '.join(words[:8])}…' ({d.source_file})")
    note(f"[5] Tìm theo số hiệu: {code_hits}/{code_total} đúng hạng 1; theo trích yếu: {title_hits}/{title_total} trong top 5")

    # ------------------------------------------------------------------ 6. hỏi đáp qua LLM
    qa: List[Tuple[str, str, str]] = []
    if not args.no_llm:
        picks = [d for d in docs.values() if d.so_hieu][:2] + [d for d in docs.values() if not d.so_hieu][:1]
        for d in picks:
            ref = d.so_hieu or d.doc_id
            try:
                t0 = time.perf_counter()
                ans = rt.assistant.summarize(ref, style="ngan")
                ok = "Nguồn" in ans.text and len(ans.text) > 200
                qa.append((f"Tóm tắt {ref}", "OK" if ok else "THIẾU NỘI DUNG/NGUỒN", f"{time.perf_counter() - t0:.1f}s"))
            except Exception as e:
                qa.append((f"Tóm tắt {ref}", f"LỖI {type(e).__name__}", ""))
                issues.append(f"Hỏi đáp: lỗi khi tóm tắt {ref}: {type(e).__name__}")
        note(f"[6] Hỏi đáp: {qa}")

    # ------------------------------------------------------------------ 7. chạy lại không tốn thêm (tiếp tục được sau sự cố)
    rerun_local = corpus.run_local(plans, args.workers)["pages"]
    rerun_api = len(corpus.pending_api_pages(plans))
    rerun_idx = rt.indexer.index_paths(md_paths)
    note(f"[7] Chạy lại: trích thêm {rerun_local} trang, còn {rerun_api} trang chờ API, nạp lại {len(rerun_idx.indexed)} văn bản")
    if rerun_local or rerun_api or rerun_idx.indexed:
        issues.append("Chạy lại vẫn xử lý thêm – cơ chế tiếp tục sau sự cố không đúng")

    # ------------------------------------------------------------------ ngoại suy
    full_routes = Counter()
    for r in manifest:
        full_routes["DIRECT_TEXT"] += int(r["n_direct_text"] or 0)
        full_routes["STANDARD_OCR"] += int(r["n_standard_ocr"] or 0)
        full_routes["VLM"] += int(r["n_vlm"] or 0)
    full_api = full_routes["VLM"] + full_routes["STANDARD_OCR"]
    s_local = t_local / local_pages if local_pages else 0
    s_api = t_api / api_stats["pages"] if api_stats.get("pages") else 0
    usd_page = api_cost / api_stats["pages"] if api_stats.get("pages") else 0
    sample_pages = sum(p.total_pages for p in plans.values())
    chunks_per_page = report.chunks / max(1, sum(plans[m].total_pages for m in md5s if m in docs))
    s_chunk = t_idx / report.chunks if report.chunks else 0
    full_pages = sum(int(r["pages"]) for r in manifest)
    full_chunks = chunks_per_page * full_pages
    md_bytes = sum(Path(p).stat().st_size for p in md_paths if Path(p).exists())
    qdrant_bytes_per_chunk = 1024 * 4 * 2.2 + 600     # vector float32 + HNSW + payload (ước tính)

    # ------------------------------------------------------------------ báo cáo
    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "–"

    lines = [
        "# Chạy thử tổng duyệt trước khi chạy toàn kho",
        "",
        f"- Thời điểm: {time.strftime('%d/%m/%Y %H:%M')}; mô hình VLM/LLM: {parser.model}; luồng song song: {args.workers}",
        f"- Mẫu: {len(plans)} file / {sample_pages} trang từ {len({folder_key(r['path']) for r in sample})} thư mục (tỉ lệ theo số trang, seed {args.seed}); "
        f"phân luồng: {dict(routes)}",
        "",
        "## Kết luận",
        "",
        ("**KHÔNG có sự cố chặn** – có thể chạy toàn kho." if not issues else "**CÓ VẤN ĐỀ cần xử lý trước khi chạy toàn kho:**"),
        *[f"- {i}" for i in issues],
        "",
        "## Kết quả từng bước",
        "",
        "| Bước | Kết quả | Thời gian |",
        "|---|---|---:|",
        f"| 1. Trích text trên máy | {local_pages} trang, lỗi {local_stats['errors']} file | {t_local:.0f}s ({s_local:.2f}s/trang) |",
        f"| 2. Gửi API | {api_stats.get('pages', 0)} trang, lỗi {api_stats.get('errors', 0)}, chi {api_cost:.4f} USD | {t_api:.0f}s ({s_api:.2f}s/trang, {args.workers} luồng) |",
        f"| 3. Ghép Markdown | {asm} | {t_asm:.0f}s |",
        f"| 4. Nạp chỉ mục | {len(report.indexed)} văn bản, {report.chunks} chunk, lỗi {len(report.failed)} | {t_idx:.0f}s ({s_chunk:.2f}s/chunk) |",
        f"| 5. Tìm theo số hiệu (hạng 1) | {pct(code_hits, code_total)} | TB {statistics.mean(lat) if lat else 0:.0f} ms |",
        f"| 5. Tìm theo trích yếu (top 5) | {pct(title_hits, title_total)} | |",
        f"| 6. Tóm tắt qua LLM | {', '.join(f'{q}: {r}' for q, r, _ in qa) or 'bỏ qua'} | |",
        f"| 7. Chạy lại (tiếp tục sau sự cố) | trích thêm {rerun_local}, API còn {rerun_api}, nạp lại {len(rerun_idx.indexed)} | |",
        "",
        "## Chất lượng đầu ra",
        "",
        f"- Trang text gốc: {len(direct_lens)} trang, độ dài trung vị {statistics.median(direct_lens) if direct_lens else 0:.0f} ký tự; "
        f"gần như rỗng: {len(empty_direct)}; lớp chữ font cũ (TCVN3/VNI) hỏng: {len(legacy_pages)}; tiếng Việt mất dấu: {len(no_diacritics)}",
        f"- Trang API: {len(rows)} trang; token ra trung vị {statistics.median(outs) if outs else 0:.0f}, tối đa {max(outs) if outs else 0}; "
        f"chạm trần token: {len(cap_hits)}; tự ngắt do lặp: {len(loop_cut)}; không đọc được (ảnh mờ/trống): {len(unreadable)}",
        f"- Chunk: trung vị {statistics.median(chunk_tokens) if chunk_tokens else 0:.0f} token, p95 {tok_p95}, tối đa {chunk_tokens[-1] if chunk_tokens else 0}",
        f"- Hồ sơ chunking: {dict(profiles)}; văn bản hành chính có số hiệu: {pct(so_hieu_ok, len(admin_like))}, có ngày ban hành: {pct(date_ok, len(admin_like))}",
        "",
        "### Trang cần xem lại",
        *[f"- Text gốc gần như rỗng: {x}" for x in empty_direct[:10]],
        *[f"- Lớp chữ font cũ hỏng (nên chuyển VLM): {x}" for x in legacy_pages[:10]],
        *[f"- Mất dấu tiếng Việt: {x}" for x in no_diacritics[:10]],
        *[f"- API chạm trần token: {x}" for x in cap_hits[:10]],
        *[f"- API tự ngắt do lặp: {x}" for x in loop_cut[:10]],
        *[f"- Không đọc được: {x}" for x in unreadable[:15]],
        *[f"- Tìm không ra: {x}" for x in misses[:15]],
        "",
        "## Ngoại suy cho toàn kho",
        "",
        f"Toàn kho: {len(manifest):,} file, {full_pages:,} trang (text gốc {full_routes['DIRECT_TEXT']:,}, OCR {full_routes['STANDARD_OCR']:,}, VLM {full_routes['VLM']:,}).",
        "",
        "| Hạng mục | Ước tính |",
        "|---|---|",
        f"| Trích text trên máy | ~{s_local * full_routes['DIRECT_TEXT'] / 3600:.1f} giờ |",
        f"| Gửi API trực tiếp ({full_api:,} trang, {args.workers} luồng) | ~{s_api * full_api / 3600:.1f} giờ, ~{usd_page * full_api:.0f} USD |",
        f"| Gửi API qua Batch (−50%) | ~{usd_page * full_api * settings.vlm.batch_discount:.0f} USD, thời gian trả kết quả không cam kết (≤ 24 giờ/lô) |",
        f"| Nạp chỉ mục (~{full_chunks:,.0f} chunk) | ~{s_chunk * full_chunks / 3600:.1f} giờ |",
        f"| Dung lượng Markdown | ~{mb(md_bytes / max(1, sample_pages) * full_pages)} |",
        f"| Dung lượng Qdrant | ~{mb(qdrant_bytes_per_chunk * full_chunks)} |",
        f"| RAM tiến trình nạp (đỉnh trong lượt thử) | {mb(peak.peak)} |",
        f"| Ổ đĩa còn trống ({settings.project_root.anchor}) | {mb(psutil.disk_usage(str(settings.project_root.anchor)).free)} |",
        "",
        "## Nhật ký",
        "",
        *[f"    {x}" for x in log],
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nBáo cáo: {OUT / 'report.md'}")
    return 1 if issues else 0


def store_out_paths(store: CorpusStore, md5s: set) -> List[Tuple[str, str]]:
    with store._conn() as c:
        return [(r[0], r[1]) for r in c.execute("SELECT md5, out_path FROM files WHERE status='assembled'") if r[0] in md5s]


if __name__ == "__main__":
    sys.exit(main())
