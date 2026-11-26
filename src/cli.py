"""
Hanoi Metro AI Agent - Command Line Interface.

  python -m src.cli index [--force] [--file F.md]       Nạp data/processed_md vào kho văn bản + Qdrant
  python -m src.cli status                              Thống kê chỉ mục
  python -m src.cli search "câu hỏi"                    Xem đoạn trích tìm được (không gọi LLM)
  python -m src.cli ask "câu hỏi"                       Hỏi đáp có trích dẫn
  python -m src.cli summarize "916/CV-VT&QLPT" [--style ngan|gach_dau_dong|chi_tiet]
  python -m src.cli report "chủ đề" [--type bao_cao|to_trinh|cong_van|tong_hop] [--doc SỐ_HIỆU ...] [--out f.md]
  python -m src.cli list [--loai CV QĐ] [--from 2024] [--to 2024] [--keyword "..."]
  python -m src.cli eval data/eval/questions.jsonl      Đánh giá thu nhỏ C1 / C2 / C4
  python -m src.cli mine-abbrev                         Khai thác ứng viên viết tắt để duyệt
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.config import settings
from src.runtime import build_runtime
from src.store.base import Filters


def _filters(args) -> Filters:
    return Filters(
        loai_vb=getattr(args, "loai", None) or None,
        year_from=getattr(args, "year_from", None),
        year_to=getattr(args, "year_to", None),
    )


def _print_answer(answer) -> None:
    for note in answer.notes:
        print(f"> {note}")
    print(answer.text)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m src.cli", description="Hanoi Metro AI Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index")
    s.add_argument("--force", action="store_true")
    s.add_argument("--file", type=Path, action="append")
    s.add_argument("--dir", type=Path, help="Thư mục Markdown cần nạp (mặc định PROCESSED_MD_DIR; toàn kho: data/corpus_md)")
    sub.add_parser("status")

    for name in ("search", "ask"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("--loai", nargs="*")
        s.add_argument("--from", dest="year_from", type=int)
        s.add_argument("--to", dest="year_to", type=int)
        s.add_argument("-k", type=int, default=None)

    s = sub.add_parser("summarize")
    s.add_argument("document")
    s.add_argument("--style", default="chi_tiet", choices=["ngan", "gach_dau_dong", "chi_tiet"])
    s.add_argument("--focus", default="")

    s = sub.add_parser("report")
    s.add_argument("topic")
    s.add_argument("--type", default="bao_cao", choices=["bao_cao", "to_trinh", "cong_van", "tong_hop"])
    s.add_argument("--doc", action="append")
    s.add_argument("--recipient", default="")
    s.add_argument("--unit", default="")
    s.add_argument("--instructions", default="")
    s.add_argument("--out", type=Path)

    s = sub.add_parser("list")
    s.add_argument("--loai", nargs="*")
    s.add_argument("--from", dest="year_from", type=int)
    s.add_argument("--to", dest="year_to", type=int)
    s.add_argument("--keyword", default="")
    s.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("eval")
    s.add_argument("questions", type=Path)

    sub.add_parser("mine-abbrev")
    args = p.parse_args(argv)
    rt = build_runtime()

    if args.cmd == "index":
        paths = args.file or sorted((args.dir or settings.ingestion.processed_md_dir).glob("*.md"))
        report = rt.indexer.index_paths(paths, force=args.force)
        print(report.summary())
        for f, err in report.failed.items():
            print(f"  LỖI {f}: {err}")
        return 1 if report.failed else 0

    if args.cmd == "status":
        print(json.dumps(rt.store.stats(), ensure_ascii=False))
        docs, _ = rt.store.list_documents(Filters(), limit=100000)
        print("Hồ sơ chunking:", dict(Counter(d.ho_so for d in docs)))
        print("Cần rà soát (độ tin cậy medium/low):")
        for d in docs:
            if d.ho_so_confidence != "high":
                print(f"  {d.ho_so} {d.ho_so_confidence:6} {d.source_file} — {'; '.join(d.ho_so_reasons)}")
        return 0

    if args.cmd == "search":
        res = rt.engine.search(args.query, _filters(args), k=args.k)
        print(f"Phân tích: mã={res.query.codes} viết tắt={[(e.matched, e.alternatives) for e in res.query.expansions]} "
              f"liệt kê={res.query.listing} ({res.latency_ms:.0f} ms){' LỖI NHÁNH: ' + ','.join(res.degraded) if res.degraded else ''}")
        for ev in res.evidence:
            print(f"\n[{ev.rank}] {ev.citation}  (RRF {ev.score:.4f}, nhánh {ev.branches})\n    {ev.chunk.context}")
            print("    " + ev.chunk.text[:300].replace("\n", " ") + ("…" if len(ev.chunk.text) > 300 else ""))
        if res.listing:
            docs, total = res.listing
            print(f"\nDanh sách đầy đủ: {total} văn bản")
        return 0

    if args.cmd == "ask":
        _print_answer(rt.assistant.ask(args.query, _filters(args)))
        return 0

    if args.cmd == "summarize":
        _print_answer(rt.assistant.summarize(args.document, args.style, args.focus))
        return 0

    if args.cmd == "report":
        answer = rt.assistant.draft_report(args.topic, args.type, args.doc, args.recipient, args.unit, args.instructions)
        _print_answer(answer)
        if args.out:
            args.out.write_text(answer.text, encoding="utf-8")
            print(f"\nĐã lưu dự thảo: {args.out}")
        return 0

    if args.cmd == "list":
        from src.agent.assistant import format_listing
        docs, total = rt.store.list_documents(_filters(args), keyword=args.keyword, limit=args.limit)
        print(format_listing(docs, total))
        return 0

    if args.cmd == "eval":
        from src.eval.bakeoff import format_report, load_questions, run_bakeoff
        print(format_report(run_bakeoff(rt.engine, load_questions(args.questions))))
        return 0

    if args.cmd == "mine-abbrev":
        from src.text.abbrev import AbbreviationDictionary
        known = {e.abbr.upper() for e in AbbreviationDictionary.load(settings.retrieval.synonym_dict_path, approved_only=False).entries}
        found: Counter = Counter()
        for md in settings.ingestion.processed_md_dir.glob("*.md"):
            found.update(AbbreviationDictionary.mine_candidates(md.read_text(encoding="utf-8")))
        for (abbr, full), n in found.most_common():
            if abbr.upper() not in known:
                print(json.dumps({"abbr": abbr, "expansions": [full], "domain": "", "approved": False, "count": n}, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
