from pathlib import Path
import fitz
from tabulate import tabulate
from src.ingestion.classifier import PageClassifier

pdf_dir = Path("data/raw_pdfs")
classifier = PageClassifier()
rows = []

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    doc = fitz.open(pdf_path)
    for idx, page in enumerate(doc, start=1):
        res = classifier.classify_page(page, idx)
        m = res.metrics
        rows.append([
            pdf_path.name[:35],
            f"{idx}/{len(doc)}",
            res.page_type.value,
            m.get("vlm_score", 0),
            f"{m.get('n_rows', 0)}x{m.get('n_cols', 0)}",
            m.get("has_merged", False),
            m.get("table_intersections", 0),
            m.get("laplacian_variance", 0),
            m.get("skew_angle", 0),
            m.get("has_corner_shadow", False),
            "; ".join(res.reasons),
        ])

headers = [
    "File", "Trang", "Loại", "Điểm VLM", "HàngxCột",
    "Gộp ô", "Điểm giao", "Độ nét", "Nghiêng", "Bóng góc", "Lý do"
]
print(tabulate(rows, headers=headers, tablefmt="grid"))