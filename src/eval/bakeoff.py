"""
Hanoi Metro AI Agent - Retrieval Bake-off (Mục 5.2 đề xuất).
Chạy cùng bộ câu hỏi có đáp án trên các cấu hình C1 (vector), C2 (từ khóa), C4 (vector + từ khóa, RRF);
đo Recall@5, Recall@10, MRR, độ trễ p95, tách theo loại câu hỏi (ma_so, viet_tat, tu_nhien, liet_ke).
Bộ câu hỏi dùng lại mỗi khi đổi chunking, mô hình, tham số.

Định dạng JSONL, mỗi dòng:
  {"id": "q1", "type": "ma_so", "question": "...", "answers": [{"source_file": "...", "page": 1}]}
  (đáp án nhận "doc_id" hoặc "source_file"; "page" tùy chọn)
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.retrieval.engine import Evidence, SearchEngine
from src.text.normalize import fold

CONFIGS: Dict[str, Optional[List[str]]] = {
    "C1-vector": ["vector"],
    "C2-keyword": ["keyword"],
    "C4-hybrid": None,
}


@dataclass
class Question:
    id: str
    type: str
    question: str
    answers: List[Dict]


@dataclass
class Score:
    hits5: List[float] = field(default_factory=list)
    hits10: List[float] = field(default_factory=list)
    rr: List[float] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)


def load_questions(path: Path) -> List[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            out.append(Question(**json.loads(line)))
    return out


def matches(ev: Evidence, answer: Dict) -> bool:
    if "doc_id" in answer and ev.doc.doc_id != answer["doc_id"]:
        return False
    if "source_file" in answer and fold(answer["source_file"]) not in fold(ev.doc.source_file):
        return False
    page = answer.get("page")
    return page is None or ev.chunk.page_from <= int(page) <= ev.chunk.page_to


def first_hit(evidence: Sequence[Evidence], answers: List[Dict]) -> Optional[int]:
    for i, ev in enumerate(evidence, start=1):
        if any(matches(ev, a) for a in answers):
            return i
    return None


def run_bakeoff(engine: SearchEngine, questions: List[Question], configs: Dict[str, Optional[List[str]]] = CONFIGS) -> Dict[str, Dict[str, Score]]:
    results: Dict[str, Dict[str, Score]] = {}
    for name, branches in configs.items():
        per_type: Dict[str, Score] = {}
        for q in questions:
            res = engine.search(q.question, k=10, branches=branches)
            pos = first_hit(res.evidence, q.answers)
            for key in (q.type, "TẤT CẢ"):
                s = per_type.setdefault(key, Score())
                s.hits5.append(1.0 if pos and pos <= 5 else 0.0)
                s.hits10.append(1.0 if pos and pos <= 10 else 0.0)
                s.rr.append(1.0 / pos if pos else 0.0)
                s.latency.append(res.latency_ms)
        results[name] = per_type
    return results


def _p95(values: List[float]) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=20)[-1]


def format_report(results: Dict[str, Dict[str, Score]]) -> str:
    lines = ["| Cấu hình | Loại câu hỏi | Số câu | Recall@5 | Recall@10 | MRR | p95 (ms) |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, per_type in results.items():
        for qtype, s in sorted(per_type.items(), key=lambda kv: (kv[0] == "TẤT CẢ", kv[0])):
            n = len(s.rr)
            lines.append(
                f"| {name} | {qtype} | {n} | {sum(s.hits5) / n:.0%} | {sum(s.hits10) / n:.0%} | {sum(s.rr) / n:.3f} | {_p95(s.latency):.0f} |"
            )
    return "\n".join(lines)
