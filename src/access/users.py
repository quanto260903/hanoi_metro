"""
Hanoi Metro AI Agent - Quản trị khai báo phòng ban từ dòng lệnh (người chạy lệnh có quyền như quản trị hệ thống).

  python -m src.access.users cho-duyet [--phong KHDA]
  python -m src.access.users duyet  <email> [--vai-tro nhan_vien]
  python -m src.access.users tu-choi <email> [--ly-do "..."]
  python -m src.access.users danh-sach            (người dùng đã duyệt, theo đơn vị)
  python -m src.access.users them <email> --phong VP --vai-tro quan_tri --ho-ten "..."   (thêm trực tiếp, vd tài
                                                   khoản quản trị đầu tiên để duyệt được trong khung chat)

Duyệt xong có hiệu lực ngay trên gateway đang chạy (gateway tự nạp lại nguoi_dung.csv).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.access.policy import load_policy
from src.access.registration import _EMAIL, CHO_DUYET, Registrations
from src.config import settings


def _paths():
    cfg = settings.agent
    cfg_dir = Path(cfg.access_config_dir)
    cfg_dir = cfg_dir if cfg_dir.is_absolute() else settings.project_root / cfg_dir
    reg = Path(cfg.access_registrations)
    return cfg_dir, reg if reg.is_absolute() else settings.project_root / reg


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Duyệt khai báo phòng ban")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cho-duyet")
    s.add_argument("--phong")
    s = sub.add_parser("duyet")
    s.add_argument("email")
    s.add_argument("--vai-tro", default="nhan_vien")
    s = sub.add_parser("tu-choi")
    s.add_argument("email")
    s.add_argument("--ly-do", default="")
    sub.add_parser("danh-sach")
    s = sub.add_parser("them")
    s.add_argument("email")
    s.add_argument("--phong", required=True)
    s.add_argument("--vai-tro", required=True)
    s.add_argument("--ho-ten", default="")
    args = ap.parse_args(argv)

    cfg_dir, reg_path = _paths()
    policy = load_policy(cfg_dir, enabled=False)
    if policy.problems:
        print("Cấu hình phân quyền có lỗi:", *policy.problems, sep="\n  ")
        return 2
    regs = Registrations(reg_path, cfg_dir / "nguoi_dung.csv")

    if args.cmd == "cho-duyet":
        items = regs.pending(args.phong.upper() if args.phong else None)
        for r in items:
            print(f"  {r['email']:<35} {r['phong']:<8} {r['ho_ten']:<30} {r['ngay_khai']}")
        print(f"{len(items)} khai báo đang chờ duyệt.")
    elif args.cmd in ("duyet", "tu-choi"):
        req = regs.get(args.email)
        if not req or req["trang_thai"] != CHO_DUYET:
            print(f"Không có khai báo đang chờ duyệt của {args.email}.")
            return 1
        if args.cmd == "duyet":
            if args.vai_tro not in policy.roles:
                print(f"Vai trò '{args.vai_tro}' không có. Các vai trò: {', '.join(policy.roles)}")
                return 1
            regs.decide(args.email, True, "quan_tri_cli", args.vai_tro)
            print(f"Đã duyệt {args.email} vào {req['phong']}, vai trò {args.vai_tro}.")
        else:
            regs.decide(args.email, False, "quan_tri_cli", note=args.ly_do)
            print(f"Đã từ chối khai báo của {args.email}.")
    elif args.cmd == "them":
        phong = args.phong.upper()
        if phong not in policy.departments or args.vai_tro not in policy.roles:
            print(f"Mã đơn vị hoặc vai trò không hợp lệ. Đơn vị: {', '.join(policy.departments)}; vai trò: {', '.join(policy.roles)}")
            return 1
        if not _EMAIL.match(args.email.strip()):
            print("Email không hợp lệ.")
            return 1
        regs.add_user(args.email, args.ho_ten, phong, args.vai_tro)
        print(f"Đã ghi {args.email.strip().lower()} vào {phong}, vai trò {args.vai_tro}.")
    else:
        by_dept = Counter(u["phong"] for u in policy.users.values())
        for code, dept in policy.departments.items():
            print(f"  {code:<8} {by_dept.get(code, 0):>4}  {dept.name}")
        print(f"Tổng: {len(policy.users)} người dùng đã duyệt; {len(regs.pending())} khai báo chờ duyệt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
