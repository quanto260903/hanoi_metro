"""
Hanoi Metro AI Agent - Gán nhãn phòng ban cho văn bản (phục vụ phân quyền).

Quy trình: máy GỢI Ý → người có thẩm quyền DUYỆT trong Excel → nạp nhãn đã duyệt. Nhãn không bao giờ được áp
thẳng từ gợi ý; cột phong_duyet do người duyệt điền mới có hiệu lực.

  python -m src.access.labels goi-y   [--out data/access/nhan_van_ban.csv]
  python -m src.access.labels ap-dung --file data/access/nhan_van_ban.csv
  python -m src.access.labels thong-ke
  python -m src.access.labels ung-vien --phong VTQLPT [--out data/access/thi_diem_VTQLPT.csv]
  python -m src.access.labels du-thao-chia [--out data/access/du_thao_chia.csv]

Gợi ý theo luật cố định (không gọi API):
  1. Mã đơn vị cuối số hiệu văn bản do Công ty ĐSHN ban hành: 31/TB-ATCL → ATCL
  2. Tên phòng ở phần đầu văn bản ("PHÒNG KẾ HOẠCH - DỰ ÁN") → KHDA
  3. Thư mục khai báo trong phong_ban.yaml (thu_muc)
Văn bản không khớp luật nào để trống → người duyệt điền.
Gợi ý theo nội dung (AI, python -m src.indexing.enrich chay --ai) được GHÉP vào cột phong_goi_y, lý do ghi
"AI (cần kiểm tra)"; AI không bao giờ ghi cột phong_duyet hay bảng nhãn phân quyền.
"""

import argparse
import csv
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.access.policy import CHUNG, AccessPolicy, load_policy
from src.config import settings
from src.indexing.models import DocRecord
from src.store.base import Filters
from src.text.normalize import fold

logger = logging.getLogger("hanoi_metro.access.labels")

COLUMNS = ["doc_id", "so_hieu", "ngay", "trich_yeu", "source_file", "thu_muc",
           "phong_goi_y", "ly_do", "phong_hien_tai", "phong_duyet", "ghi_chu"]
_DSHN = ("duong sat ha noi", "dshn")


def _norm(text: str) -> str:
    """Bỏ dấu, chữ thường, gộp gạch nối/khoảng trắng: "Phòng Kế hoạch – Dự án" ~ "PHÒNG KẾ HOẠCH - DỰ ÁN"."""
    return re.sub(r"[\s\-–—]+", " ", fold(text)).strip()


def _issued_by_company(doc: DocRecord) -> bool:
    text = fold(f"{doc.co_quan} {doc.don_vi} {doc.so_hieu}")
    return any(k in text for k in _DSHN)


def suggest(doc: DocRecord, policy: AccessPolicy) -> Tuple[List[str], List[str]]:
    """→ (mã phòng gợi ý, lý do). Chỉ luật cố định; rỗng khi không chắc."""
    found: Dict[str, str] = {}
    if "/" in doc.so_hieu and _issued_by_company(doc):
        units = [u.strip().upper() for u in doc.so_hieu.split("/", 1)[1].split("-")[1:]]
        for dept in policy.departments.values():
            hit = next((u for u in units if u in dept.aliases), None)
            if hit:
                found.setdefault(dept.code, f"mã đơn vị '{hit}' trong số hiệu {doc.so_hieu}")
    head = _norm(f"{doc.don_vi} {doc.co_quan}")
    for dept in policy.departments.values():
        name = _norm(dept.name)
        if name.startswith("phong ") and name in head:
            found.setdefault(dept.code, f"đơn vị ban hành: {doc.don_vi or doc.co_quan}")
    path = fold(doc.source_path.replace("\\", "/"))
    for dept in policy.departments.values():
        for pattern in dept.folders:
            if fold(pattern) in path:
                found.setdefault(dept.code, f"thư mục '{pattern}'")
    return list(found), list(found.values())


AI_FIELDS = ("phong_chu_tri", "phong_phoi_hop")
AI_TAG = "AI (cần kiểm tra)"


def _ai_hint(chu_tri: List[Tuple[str, str]], phoi_hop: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    codes = list(dict.fromkeys([c for c, _ in chu_tri] + [c for c, _ in phoi_hop]))
    why = [f"{AI_TAG}: {c} chủ trì – {r}" for c, r in chu_tri] + [f"{AI_TAG}: {c} phối hợp – {r}" for c, r in phoi_hop]
    return codes, why


def _ai_from_fields(fields: Dict[str, object]) -> Tuple[List[str], List[str]]:
    """Gợi ý AI đã lưu ở doc_fields (nguon=ai) → (mã, lý do). Chỉ để hiển thị cho người duyệt."""
    def pairs(name: str) -> List[Tuple[str, str]]:
        f = fields.get(name)
        if f is None or f.nguon != "ai" or not f.gia_tri:
            return []
        reasons = dict(x.split(": ", 1) for x in f.ghi_chu.split(" | ") if ": " in x)
        return [(c, reasons.get(c, "")) for c in f.gia_tri.split(";") if c]
    return _ai_hint(pairs("phong_chu_tri"), pairs("phong_phoi_hop"))


def _merge_row(row: Dict[str, str], codes: List[str], why: List[str]) -> None:
    current = [c for c in row.get("phong_goi_y", "").split(";") if c]
    row["phong_goi_y"] = ";".join(dict.fromkeys(current + codes))
    kept = [x for x in row.get("ly_do", "").split(" | ") if x and not x.startswith(AI_TAG)]
    row["ly_do"] = " | ".join(kept + why)


def merge_ai_suggestions(path: Path, results: Dict[str, object]) -> Optional[int]:
    """
    Ghi gợi ý AI vào file duyệt đã có (giữ nguyên cột phong_duyet, ghi_chu người duyệt đã điền).
    Trả về số dòng được cập nhật; None nếu chưa có file.
    """
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or COLUMNS
        rows = list(reader)
    changed = 0
    for row in rows:
        res = results.get((row.get("doc_id") or "").strip().lower())
        if res is None:
            continue
        codes, why = _ai_hint(res.chu_tri, res.phoi_hop)
        _merge_row(row, codes, why)
        changed += 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    return changed


def _all_documents(store) -> List[DocRecord]:
    docs, _ = store.list_documents(Filters(), limit=1_000_000)
    return docs


def cmd_suggest(rt, policy: AccessPolicy, out: Path) -> Dict[str, int]:
    docs = _all_documents(rt.store)
    current = rt.store.access_labels([d.doc_id for d in docs])
    ai_fields = rt.store.doc_fields([d.doc_id for d in docs], AI_FIELDS) if hasattr(rt.store, "doc_fields") else {}
    stats = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:       # utf-8-sig: Excel mở đúng tiếng Việt
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for d in sorted(docs, key=lambda x: (x.source_path, x.source_file)):
            phong, why = suggest(d, policy)
            ai_codes, ai_why = _ai_from_fields(ai_fields.get(d.doc_id, {}))
            stats["co_goi_y" if phong or ai_codes else "chua_co_goi_y"] += 1
            stats["co_goi_y_ai"] += bool(ai_codes)
            stats["da_co_nhan"] += bool(current.get(d.doc_id))
            folder = d.source_path.replace("\\", "/").rsplit("/", 1)[0]
            w.writerow({
                "doc_id": d.doc_id, "so_hieu": d.so_hieu, "ngay": d.ngay_ban_hanh, "trich_yeu": (d.trich_yeu or d.title)[:200],
                "source_file": d.source_file, "thu_muc": folder,
                "phong_goi_y": ";".join(dict.fromkeys(phong + ai_codes)), "ly_do": " | ".join(why + ai_why),
                "phong_hien_tai": ";".join(current.get(d.doc_id, [])),
                "phong_duyet": ";".join(current.get(d.doc_id, [])), "ghi_chu": "",
            })
    stats["tong"] = len(docs)
    return dict(stats)


def parse_labels(value: str, policy: AccessPolicy) -> Optional[List[str]]:
    """'KHDA; tckt' → ['KHDA', 'TCKT']; '-' → [] (gỡ nhãn); '' → None (không đổi). Mã lạ → ValueError."""
    value = (value or "").strip()
    if not value:
        return None
    if value == "-":
        return []
    codes = [c.strip().upper() for c in value.replace(",", ";").split(";") if c.strip()]
    bad = [c for c in codes if c not in policy.departments and c != CHUNG]
    if bad:
        raise ValueError(f"mã phòng không có trong phong_ban.yaml: {', '.join(bad)}")
    return list(dict.fromkeys(codes))


def cmd_apply(rt, policy: AccessPolicy, path: Path) -> Dict[str, int]:
    """Nạp cột phong_duyet. Kiểm tra TOÀN BỘ file trước; có lỗi thì không áp dòng nào."""
    rows: List[Tuple[str, List[str], str]] = []
    errors: List[str] = []
    known = {d.doc_id for d in _all_documents(rt.store)}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            doc_id = (row.get("doc_id") or "").strip().lower()
            try:
                labels = parse_labels(row.get("phong_duyet", ""), policy)
            except ValueError as e:
                errors.append(f"dòng {i}: {e}")
                continue
            if labels is None:
                continue
            if doc_id not in known:
                errors.append(f"dòng {i}: doc_id {doc_id or '(trống)'} không có trong chỉ mục")
                continue
            rows.append((doc_id, labels, (row.get("ghi_chu") or "").strip()))
    if errors:
        for e in errors[:50]:
            print("LỖI", e)
        raise SystemExit(f"Không áp dụng: {len(errors)} lỗi trong {path}. Sửa rồi chạy lại.")
    vector = rt.vector
    if vector is not None and hasattr(vector, "ensure_access_index"):
        vector.ensure_access_index()
    current = rt.store.access_labels([d for d, _, _ in rows])
    changed = 0
    for doc_id, labels, note in rows:
        if sorted(current.get(doc_id, [])) == sorted(labels):
            continue
        # Qdrant trước, kho sau: nếu đứt giữa chừng, lần chạy lại vẫn thấy khác biệt và làm lại
        if vector is not None:
            vector.set_access_payload(doc_id, labels)
        rt.store.set_access_labels(doc_id, labels, nguon="duyet", ghi_chu=note)
        changed += 1
    return {"dong_hop_le": len(rows), "da_cap_nhat": changed}


def candidate_score(doc: DocRecord, code: str, policy: AccessPolicy, first_page: str,
                    ai_fields: Dict[str, object]) -> Tuple[int, List[str]]:
    """
    Điểm "có thể thuộc phòng `code`" để XẾP danh sách cho người duyệt (không gán nhãn):
    luật cố định +3, AI chủ trì +3 / phối hợp +1, từ khóa ở trích yếu/tên file +2, ở trang đầu +1 (tối đa +3).
    """
    dept = policy.departments[code]
    score, why = 0, []
    codes, _ = suggest(doc, policy)
    if code in codes:
        score += 3
        why.append("luật (mã đơn vị / tên phòng / thư mục)")
    ct, ph = ai_fields.get("phong_chu_tri"), ai_fields.get("phong_phoi_hop")
    if ct is not None and code in ct.gia_tri.split(";"):
        score += 3
        why.append("AI: chủ trì")
    elif ph is not None and code in ph.gia_tri.split(";"):
        score += 1
        why.append("AI: phối hợp")
    head = f" {_norm(f'{doc.so_hieu} {doc.trich_yeu} {doc.title} {doc.source_file}')} "
    body = f" {_norm(first_page[:4000])} "
    in_head = [k for k in dept.keywords if f" {_norm(k)} " in head or (len(_norm(k)) >= 4 and _norm(k) in head)]
    in_body = [k for k in dept.keywords if k not in in_head and f" {_norm(k)} " in body]
    if in_head:
        score += 2
        why.append("từ khóa ở trích yếu/tên file: " + ", ".join(in_head[:4]))
    if in_body:
        score += min(3, len(in_body))
        why.append("từ khóa ở trang đầu: " + ", ".join(in_body[:4]))
    return score, why


def cmd_candidates(rt, policy: AccessPolicy, code: str, out: Path, min_score: int = 2) -> Dict[str, int]:
    """File ứng viên cùng định dạng file duyệt (chạy được ap-dung): người duyệt điền phong_duyet, để trống = bỏ qua."""
    from src.indexing.enrich import document_pages
    docs = _all_documents(rt.store)
    ids = [d.doc_id for d in docs]
    current = rt.store.access_labels(ids)
    ai = rt.store.doc_fields(ids, AI_FIELDS) if hasattr(rt.store, "doc_fields") else {}
    rows = []
    for d in docs:
        pages = document_pages(rt.store, d.doc_id)
        first = pages[min(pages)] if pages else ""
        score, why = candidate_score(d, code, policy, first, ai.get(d.doc_id, {}))
        if score >= min_score:
            rows.append((score, d, why))
    rows.sort(key=lambda x: (-x[0], x[1].source_file))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS + ["diem", "muc_do"])
        w.writeheader()
        for score, d, why in rows:
            w.writerow({
                "doc_id": d.doc_id, "so_hieu": d.so_hieu, "ngay": d.ngay_ban_hanh,
                "trich_yeu": (d.trich_yeu or d.title)[:200], "source_file": d.source_file,
                "thu_muc": d.source_path.replace("\\", "/").rsplit("/", 1)[0], "phong_goi_y": code,
                "ly_do": " | ".join(why), "phong_hien_tai": ";".join(current.get(d.doc_id, [])),
                "phong_duyet": "", "ghi_chu": "", "diem": score, "muc_do": "cao" if score >= 4 else "trung_binh",
            })
    return {"ung_vien": len(rows), "muc_cao": sum(s >= 4 for s, _, _ in rows), "tong_van_ban": len(docs)}


PERSONAL_OWNER = "TCNS"
_RX_ID_WORD = re.compile(r"(c[aă]n c[uư][oớ]c|ch[uứ]ng minh nh[aâ]n d[aâ]n|\bcccd\b|\bcmnd\b)", re.I)
_RX_ID_NUM = re.compile(r"(?<!\d)0\d{11}(?!\d)")


def has_personal_ids(text: str) -> bool:
    """Danh sách có số CCCD (12 số) kèm chữ "căn cước"/"chứng minh nhân dân": dữ liệu cá nhân → chỉ TCNS xem."""
    return bool(_RX_ID_WORD.search(text or "")) and len(set(_RX_ID_NUM.findall(text or ""))) >= 1


def doc_code_hits(doc: DocRecord, policy: AccessPolicy, first_page: str = "") -> List[str]:
    """Phòng có tiền tố mã tài liệu (ma_tai_lieu) khớp tên file / số hiệu / đầu trang 1."""
    names = [doc.source_file.upper(), (doc.so_hieu or "").upper()]
    head = (first_page or "")[:1500].upper()
    out = []
    for dept in policy.departments.values():
        for pat in dept.doc_codes:
            hit = any(n.startswith(pat) or (pat.startswith("-") and pat in n) for n in names) \
                or (len(pat) >= 5 and pat in head)
            if hit:
                out.append(dept.code)
                break
    return out


def draft_split_codes(doc: DocRecord, policy: AccessPolicy, ai_fields: Dict[str, object],
                      text: str = "") -> Tuple[List[str], str]:
    """
    Đề xuất nhãn xem cho MỘT văn bản (nhiều phòng được): luật cố định (số hiệu, mã tài liệu) ∪ AI "phong_can_xem"
    (gồm chủ trì, phối hợp, có thể có CHUNG). Văn bản chứa số CCCD → CHỈ TCNS. → (mã, nguồn).
    Chỉ là bản nháp: có hiệu lực khi quản trị chạy ap-dung trên file. `text`: nội dung các trang (trang 1 trước).
    """
    if has_personal_ids(text):
        return [PERSONAL_OWNER], "luật: có số CCCD (thông tin cá nhân) → chỉ TCNS"
    rule_codes, _ = suggest(doc, policy)
    rule_codes = list(dict.fromkeys(rule_codes + doc_code_hits(doc, policy, text)))
    f = ai_fields.get("phong_can_xem")
    ai_codes = [c for c in f.gia_tri.split(";") if c] if f is not None and f.nguon == "ai" and f.gia_tri else []
    known = set(policy.departments) | {CHUNG}
    codes = [c for c in dict.fromkeys(rule_codes + ai_codes) if c in known]
    src = "+".join(x for x, used in (("luật", rule_codes), (f"AI {f.phien_ban_bo_trich}" if f is not None else "AI",
                                                             ai_codes)) if used)
    return codes, src


def cmd_draft_split(rt, policy: AccessPolicy, out: Path) -> Dict[str, int]:
    """
    File nháp chia văn bản cho các phòng, cùng định dạng file duyệt: cột phong_duyet ĐIỀN SẴN theo luật + AI.
    Quản trị xem/sửa rồi chạy ap-dung (khi đó nhãn mới có hiệu lực, nguon=duyet). Văn bản chưa có đề xuất để trống
    (ap-dung bỏ qua, giữ nhãn hiện tại).
    """
    docs = _all_documents(rt.store)
    ids = [d.doc_id for d in docs]
    current = rt.store.access_labels(ids)
    ai = rt.store.doc_fields(ids, ("phong_can_xem",)) if hasattr(rt.store, "doc_fields") else {}
    from src.indexing.enrich import document_pages
    counts: Counter = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for d in sorted(docs, key=lambda x: x.source_file):
            pages = document_pages(rt.store, d.doc_id)
            text = "\n".join(pages[p] for p in sorted(pages))
            codes, src = draft_split_codes(d, policy, ai.get(d.doc_id, {}), text)
            counts["co_de_xuat" if codes else "chua_co_de_xuat"] += 1
            for c in codes:
                counts[c] += 1
            w.writerow({
                "doc_id": d.doc_id, "so_hieu": d.so_hieu, "ngay": d.ngay_ban_hanh,
                "trich_yeu": (d.trich_yeu or d.title)[:200], "source_file": d.source_file,
                "thu_muc": d.source_path.replace("\\", "/").rsplit("/", 1)[0],
                "phong_goi_y": ";".join(codes), "ly_do": src,
                "phong_hien_tai": ";".join(current.get(d.doc_id, [])),
                "phong_duyet": ";".join(codes), "ghi_chu": f"chia theo {src}" if codes else "",
            })
    return dict(counts)


def cmd_stats(rt) -> Dict[str, int]:
    docs = _all_documents(rt.store)
    labels = rt.store.access_labels([d.doc_id for d in docs])
    c = Counter()
    for d in docs:
        ls = labels.get(d.doc_id, [])
        if not ls:
            c["(chưa gán nhãn – chỉ vai trò xem toàn bộ thấy)"] += 1
        for p in ls:
            c[p] += 1
    return dict(c.most_common())


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Gán nhãn phòng ban cho văn bản (phân quyền)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("goi-y")
    s.add_argument("--out", type=Path, default=settings.project_root / "data/access/nhan_van_ban.csv")
    s = sub.add_parser("ap-dung")
    s.add_argument("--file", type=Path, required=True)
    sub.add_parser("thong-ke")
    s = sub.add_parser("du-thao-chia", help="File nháp chia mọi văn bản cho các phòng (luật + AI), điền sẵn phong_duyet")
    s.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("ung-vien", help="Lập danh sách văn bản ứng viên của một phòng để duyệt (không gán nhãn)")
    s.add_argument("--phong", required=True)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--diem-toi-thieu", type=int, default=2)
    args = ap.parse_args(argv)

    from src.runtime import build_runtime
    rt = build_runtime()
    cfg_dir = Path(settings.agent.access_config_dir)
    policy = load_policy(cfg_dir if cfg_dir.is_absolute() else settings.project_root / cfg_dir, enabled=False)
    if policy.problems:
        print("Cấu hình phân quyền có lỗi:", *policy.problems, sep="\n  ")
        return 2
    if args.cmd == "goi-y":
        print("Gợi ý:", cmd_suggest(rt, policy, args.out))
        print(f"Đã ghi {args.out}. Mở bằng Excel, điền/sửa cột phong_duyet (mã phòng, cách nhau bằng ;  '-' = gỡ nhãn), "
              "rồi chạy: python -m src.access.labels ap-dung --file <file>")
    elif args.cmd == "du-thao-chia":
        out = args.out or settings.project_root / "data/access/du_thao_chia.csv"
        print("Đề xuất:", cmd_draft_split(rt, policy, out))
        print(f"Đã ghi {out}. Xem/sửa cột phong_duyet (nhiều phòng cách nhau bằng ;), rồi chạy: "
              f"python -m src.access.labels ap-dung --file {out}")
    elif args.cmd == "ung-vien":
        code = args.phong.upper()
        if code not in policy.departments:
            print(f"Mã phòng '{code}' không có. Các mã: {', '.join(policy.departments)}")
            return 2
        out = args.out or settings.project_root / f"data/access/thi_diem_{code}.csv"
        print("Ứng viên:", cmd_candidates(rt, policy, code, out, args.diem_toi_thieu))
        print(f"Đã ghi {out}. Người có thẩm quyền điền cột phong_duyet ({code}, hoặc {code};CHUNG, hoặc '-' nếu không thuộc), "
              f"dòng để trống được bỏ qua; rồi chạy: python -m src.access.labels ap-dung --file {out}")
    elif args.cmd == "ap-dung":
        print("Áp dụng:", cmd_apply(rt, policy, args.file))
    else:
        for k, v in cmd_stats(rt).items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
