"""
Hanoi Metro AI Agent - VLM Pilot Run (Mục 5 Báo cáo dự toán: đo token thực tế trên trang mẫu Luồng 3).
Chọn ngẫu nhiên có phân tầng N trang VLM từ kết quả page_router (chỉ file trong ingest_manifest.csv: đã loại trùng,
đã loại tài liệu có dấu chỉ độ mật), gửi VLM API qua CloudVLMParser (kiểm tra danh sách chặn, hạn mức, ghi sổ), rồi xuất:
  pilot_pages.csv   từng trang: lý do, token vào/ra, chi phí, số ký tự, cờ bất thường
  pilot_report.md   token TB theo lý do, ước tính có trọng số cho 3 kịch bản, trang bất thường
  review.html       ảnh trang gốc cạnh bản bóc tách để người kiểm tra đối chiếu (mẫu --review trang)

  python -m src.ingestion.pilot --root "D:\\40GB" --results "D:\\40GB\\_phan_tich_chi_phi_VLM\\ket_qua" --n 200
"""

import argparse
import base64
import csv
import html
import logging
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import fitz

from src.config import settings
from src.ingestion.cloud_vlm import BudgetExceeded, CloudVLMParser

logger = logging.getLogger("hanoi_metro.pilot")

SCENARIO_PAGES = {"Kịch bản 1": 52661, "Kịch bản 2": 64000, "Kịch bản 3": 111011}


@dataclass
class Sample:
    path: str
    md5: str
    page: int            # 0-based như pages_routed.csv
    reason: str
    reasons: str
    result: Dict = field(default_factory=dict)


def load_population(results_dir: Path) -> Dict[str, List[Sample]]:
    allowed = {r["md5"] for r in csv.DictReader(open(results_dir / "ingest_manifest.csv", encoding="utf-8-sig"))}
    by_reason: Dict[str, List[Sample]] = defaultdict(list)
    for r in csv.DictReader(open(results_dir / "pages_routed.csv", encoding="utf-8-sig")):
        if r["route"] == "VLM" and r["md5"] in allowed:
            by_reason[r["reasons"].split("|")[0]].append(Sample(r["path"], r["md5"], int(r["page"]), r["reasons"].split("|")[0], r["reasons"]))
    return by_reason


def allocate(sizes: Dict[str, int], n: int, minimum: int) -> Dict[str, int]:
    """Mỗi nhóm ít nhất `minimum` trang (nếu có đủ), phần còn lại chia theo tỷ lệ kích thước nhóm."""
    alloc = {k: min(v, minimum) for k, v in sizes.items()}
    rest = n - sum(alloc.values())
    remaining = {k: v - alloc[k] for k, v in sizes.items() if v > alloc[k]}
    total = sum(remaining.values()) or 1
    for k, v in sorted(remaining.items(), key=lambda kv: -kv[1]):
        add = min(v, round(rest * v / total))
        alloc[k] += add
    while sum(alloc.values()) > n:
        k = max(alloc, key=alloc.get)
        alloc[k] -= 1
    for k in sorted(remaining, key=lambda key: -remaining[key]):      # bù phần hụt do làm tròn
        if sum(alloc.values()) >= n:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
    return alloc


def run_page(parser: CloudVLMParser, root: Path, s: Sample) -> Dict:
    pdf = fitz.open(root / s.path)
    try:
        res = parser.parse_page(pdf[s.page], s.page + 1, s.md5)
    finally:
        pdf.close()
    text = res.markdown
    return {
        "in_tokens": res.in_tokens, "out_tokens": res.out_tokens, "cost_usd": res.cost_usd, "cached": res.cached,
        "chars": len(text), "has_table": ("|" in text or "<table" in text.lower()),
        "markdown": text,
    }


def render_preview(root: Path, s: Sample, dpi: int = 90) -> str:
    pdf = fitz.open(root / s.path)
    try:
        return base64.b64encode(pdf[s.page].get_pixmap(dpi=dpi, alpha=False).tobytes("jpeg", jpg_quality=70)).decode()
    finally:
        pdf.close()


def write_outputs(out: Path, samples: List[Sample], pop_sizes: Dict[str, int], root: Path, review_n: int, seed: int) -> None:
    done = [s for s in samples if s.result and "error" not in s.result]
    with open(out / "pilot_pages.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "md5", "page_1based", "reason", "reasons", "in_tokens", "out_tokens", "cost_usd", "chars", "has_table", "error"])
        for s in samples:
            r = s.result
            w.writerow([s.path, s.md5, s.page + 1, s.reason, s.reasons, r.get("in_tokens"), r.get("out_tokens"),
                        f"{r.get('cost_usd', 0):.6f}", r.get("chars"), r.get("has_table"), r.get("error", "")])

    cfg = settings.vlm
    lines = [f"# Chạy thử VLM trên trang thật – {cfg.model_name}", "",
             f"- Trang mẫu: {len(samples)} (xong {len(done)}, lỗi {len(samples) - len(done)}); phân tầng theo lý do phân luồng.",
             f"- Đơn giá dùng để tính: {cfg.price_in_per_m} USD vào / {cfg.price_out_per_m} USD ra mỗi 1M token (giá tham khảo, cần đối chiếu trang giá chính thức).",
             f"- Tổng chi lượt chạy này: {sum(s.result.get('cost_usd', 0) for s in done):.4f} USD.", "",
             "| Lý do phân luồng | Trang trong kho | Mẫu | Token vào TB | Token ra TB | Ra tối đa | USD/trang TB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    per_reason_cost: Dict[str, float] = {}
    for reason, size in sorted(pop_sizes.items(), key=lambda kv: -kv[1]):
        rs = [s.result for s in done if s.reason == reason]
        if not rs:
            continue
        cost = statistics.mean(r["cost_usd"] for r in rs)
        per_reason_cost[reason] = cost
        lines.append(f"| {reason} | {size:,} | {len(rs)} | {statistics.mean(r['in_tokens'] for r in rs):,.0f} | "
                     f"{statistics.mean(r['out_tokens'] for r in rs):,.0f} | {max(r['out_tokens'] for r in rs):,} | {cost:.5f} |")
    covered = sum(pop_sizes[k] for k in per_reason_cost)
    weighted = sum(per_reason_cost[k] * pop_sizes[k] for k in per_reason_cost) / covered if covered else 0
    lines += ["", f"**Chi phí TB có trọng số theo kho: {weighted:.5f} USD/trang**", "",
              "| Kịch bản | Số trang | Gọi thường | Batch API (−50%) |", "|---|---:|---:|---:|"]
    for name, pages in SCENARIO_PAGES.items():
        lines.append(f"| {name} | {pages:,} | {pages * weighted:,.2f} USD | {pages * weighted / 2:,.2f} USD |")
    outs = [s.result["out_tokens"] for s in done]
    if outs:
        p95 = sorted(outs)[int(len(outs) * 0.95) - 1]
        lines += ["", f"- Token ra: trung vị {statistics.median(outs):,.0f}, p95 {p95:,}, tối đa {max(outs):,}."]
    unreadable = [s for s in done if s.result["chars"] < 50]
    odd = [s for s in done if s.result["out_tokens"] >= cfg.max_tokens * 0.9]
    errors = [s for s in samples if "error" in s.result]
    if unreadable:
        lines += ["", f"## Trang gần như không có chữ đọc được ({len(unreadable)}): xem ảnh, scan lại nếu bản giấy có nội dung", ""]
        lines += [f"- {s.path} – trang {s.page + 1} ({s.reason}): `{s.result['markdown'][:60]}`" for s in unreadable]
    if odd or errors:
        lines += ["", "## Trang chạm trần token hoặc lỗi (cần xem)", ""]
        lines += [f"- {s.path} – trang {s.page + 1} ({s.reason}): {s.result['chars']} ký tự, {s.result['out_tokens']} token ra" for s in odd]
        lines += [f"- {s.path} – trang {s.page + 1}: LỖI {s.result['error']}" for s in errors]
    (out / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rng = random.Random(seed + 1)
    by_reason = defaultdict(list)
    for s in done:
        by_reason[s.reason].append(s)
    picks: List[Sample] = []
    while len(picks) < min(review_n, len(done)):          # xoay vòng theo lý do để mẫu rà soát đa dạng
        for reason in list(by_reason):
            if by_reason[reason] and len(picks) < review_n:
                picks.append(by_reason[reason].pop(rng.randrange(len(by_reason[reason]))))
    parts = ["<!doctype html><meta charset='utf-8'><title>Rà soát bóc tách VLM</title><style>",
             "body{font-family:system-ui,sans-serif;margin:16px} .p{display:grid;grid-template-columns:1fr 1fr;gap:16px;"
             "border-bottom:2px solid #888;padding:16px 0} img{width:100%;border:1px solid #ccc} pre{white-space:pre-wrap;"
             "font-size:13px;background:#f6f6f6;padding:8px;max-height:1100px;overflow:auto} h3{grid-column:1/3;margin:0}"
             "label{margin-right:12px}</style>",
             f"<h1>Rà soát bóc tách VLM ({settings.vlm.model_name}) – {len(picks)} trang</h1>",
             "<p>So từng trang: ảnh gốc (trái) và bản bóc tách (phải). Đánh dấu kết quả; ghi lỗi cụ thể (sai số, sai tên, thiếu dòng, bảng sai cấu trúc).</p>"]
    for i, s in enumerate(picks, start=1):
        parts.append(f"<div class='p'><h3>{i}. {html.escape(s.path)} – trang {s.page + 1} <small>({html.escape(s.reasons)})</small></h3>"
                     f"<img src='data:image/jpeg;base64,{render_preview(root, s)}'><div><pre>{html.escape(s.result['markdown'])}</pre>"
                     f"<label><input type=radio name=q{i}> Đúng</label><label><input type=radio name=q{i}> Lỗi nhỏ</label>"
                     f"<label><input type=radio name=q{i}> Lỗi nghiêm trọng</label><br><textarea rows=2 style='width:100%' placeholder='Ghi chú lỗi'></textarea></div></div>")
    (out / "review.html").write_text("\n".join(parts), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Chạy thử VLM API trên trang thật để đo token và chất lượng")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-per-reason", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--review", type=int, default=30)
    ap.add_argument("--out", type=Path, default=settings.project_root / "data" / "pilot")
    args = ap.parse_args(argv)

    if not settings.vlm.is_cloud:
        print("VLM_PROVIDER không phải API đám mây; chạy thử này đo chi phí API.")
        return 1
    population = load_population(args.results)
    sizes = {k: len(v) for k, v in population.items()}
    alloc = allocate(sizes, args.n, args.min_per_reason)
    rng = random.Random(args.seed)
    samples = [s for reason, k in alloc.items() for s in rng.sample(population[reason], k)]
    print(f"Tổng thể {sum(sizes.values()):,} trang VLM; lấy mẫu {len(samples)}: {dict(Counter(s.reason for s in samples))}")

    args.out.mkdir(parents=True, exist_ok=True)
    parser = CloudVLMParser()
    stop = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_page, parser, args.root, s): s for s in samples}
        for i, fut in enumerate(as_completed(futures), start=1):
            s = futures[fut]
            try:
                s.result = fut.result()
            except BudgetExceeded as e:
                s.result = {"error": f"hạn mức: {e}"}
                if not stop:
                    print(f"DỪNG: {e}")
                    stop = True
                    for f in futures:
                        f.cancel()
            except Exception as e:  # VLMUnavailable, lỗi đọc PDF...: ghi lại, chạy tiếp trang khác
                s.result = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
            if i % 20 == 0:
                print(f"  {i}/{len(samples)} trang")
    write_outputs(args.out, samples, sizes, args.root, args.review, args.seed)
    print((args.out / "pilot_report.md").read_text(encoding="utf-8"))
    print(f"Rà soát: {args.out / 'review.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
