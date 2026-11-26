"""
Hanoi Metro AI Agent - Khai báo phòng ban và duyệt khai báo.

Luồng: người dùng đăng ký LibreChat → lần đầu dùng trợ lý gõ  /phongban <MÃ> <Họ tên>  → khai báo "chờ duyệt"
(data/access/dang_ky.csv) → trưởng phòng của phòng đó hoặc quản trị duyệt (/duyet trong chat, hoặc
python -m src.access.users) → hệ thống ghi dòng vào config/access/nguoi_dung.csv → quyền có hiệu lực ngay.

- Email lấy từ LibreChat (đã đăng nhập), người dùng không tự gõ email được; chỉ chọn mã phòng trong danh mục cố định.
- Khai báo KHÔNG tự cấp quyền (trừ khi bật tu_dong_duyet để thí điểm, và khi đó chỉ cấp vai trò nhan_vien).
- Trưởng phòng chỉ duyệt người khai báo vào phòng mình và chỉ cấp nhan_vien; vai trò cao hơn do quản trị cấp.
"""

import csv
import io
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.access.policy import AccessPolicy

CHO_DUYET, DA_DUYET, TU_CHOI = "cho_duyet", "da_duyet", "tu_choi"
REQUEST_COLUMNS = ["email", "ho_ten", "phong", "trang_thai", "ngay_khai", "nguoi_xu_ly", "ngay_xu_ly", "ghi_chu"]
USER_COLUMNS = ["email", "ho_ten", "phong", "vai_tro", "xem_them"]
_EMAIL = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")
_LOCK = threading.Lock()          # một tiến trình gateway; CLI chạy riêng nên ghi file theo kiểu thay thế nguyên tử


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _clean(text: str, limit: int = 80) -> str:
    """Bỏ ký tự có thể phá CSV / công thức Excel (=, +, -, @ đầu ô)."""
    text = " ".join((text or "").replace(",", " ").replace(";", " ").split())[:limit]
    return text.lstrip("=+-@\t")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8-sig", newline="")
    os.replace(tmp, path)


class Registrations:
    """Sổ khai báo (một dòng mỗi email, lần khai báo sau thay lần trước) + ghi danh sách người dùng đã duyệt."""

    def __init__(self, requests_path: Path, users_path: Path):
        self.requests_path, self.users_path = Path(requests_path), Path(users_path)

    # ------------------------------------------------------------------ sổ khai báo
    def _read(self) -> Dict[str, dict]:
        if not self.requests_path.exists():
            return {}
        with self.requests_path.open(encoding="utf-8-sig", newline="") as fh:
            return {r["email"].strip().lower(): r for r in csv.DictReader(fh) if (r.get("email") or "").strip()}

    def _save(self, rows: Dict[str, dict]) -> None:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=REQUEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
        _write_atomic(self.requests_path, buf.getvalue())

    def get(self, email: str) -> Optional[dict]:
        return self._read().get((email or "").strip().lower())

    def pending(self, phong: Optional[str] = None) -> List[dict]:
        return [r for r in self._read().values() if r["trang_thai"] == CHO_DUYET and (phong is None or r["phong"] == phong)]

    def declare(self, email: str, phong: str, ho_ten: str) -> dict:
        email = email.strip().lower()
        with _LOCK:
            rows = self._read()
            rows[email] = {"email": email, "ho_ten": _clean(ho_ten), "phong": phong, "trang_thai": CHO_DUYET,
                           "ngay_khai": _now(), "nguoi_xu_ly": "", "ngay_xu_ly": "", "ghi_chu": ""}
            self._save(rows)
            return rows[email]

    def decide(self, email: str, approve: bool, by: str, vai_tro: str = "nhan_vien", note: str = "") -> dict:
        email = email.strip().lower()
        with _LOCK:
            rows = self._read()
            row = rows[email]
            if approve:
                self._upsert_user(email, row["ho_ten"], row["phong"], vai_tro)
            row.update(trang_thai=DA_DUYET if approve else TU_CHOI, nguoi_xu_ly=by, ngay_xu_ly=_now(), ghi_chu=_clean(note, 200))
            self._save(rows)
            return row

    # ------------------------------------------------------------------ danh sách người dùng
    def add_user(self, email: str, ho_ten: str, phong: str, vai_tro: str) -> None:
        with _LOCK:
            self._upsert_user(email.strip().lower(), _clean(ho_ten), phong, vai_tro)

    def _upsert_user(self, email: str, ho_ten: str, phong: str, vai_tro: str) -> None:
        """Thêm/sửa đúng một dòng trong nguoi_dung.csv, giữ nguyên các dòng chú thích và người dùng khác."""
        lines = self.users_path.read_text(encoding="utf-8-sig").splitlines() if self.users_path.exists() else []
        if not lines:
            lines = [",".join(USER_COLUMNS)]
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow([email, ho_ten, phong, vai_tro, ""])
        new_line, kept, replaced = buf.getvalue(), [], False
        for line in lines:
            first = line.split(",", 1)[0].strip().lower()
            if first == email:
                if not replaced:
                    kept.append(new_line)
                    replaced = True
                continue
            kept.append(line)
        if not replaced:
            kept.append(new_line)
        _write_atomic(self.users_path, "\n".join(kept) + "\n")


# ====================================================================== lệnh trong khung chat
COMMANDS = ("phongban", "choduyet", "duyet", "tuchoi")


def department_menu(policy: AccessPolicy) -> str:
    rows = "\n".join(f"| `{d.code}` | {d.name} |" for d in policy.departments.values())
    return f"| Mã | Đơn vị |\n|---|---|\n{rows}"


def onboarding(policy: AccessPolicy) -> str:
    return ("**Chào mừng bạn đến với Trợ lý tài liệu Hanoi Metro.** Để dùng kho tài liệu, hãy khai báo đơn vị công tác "
            "bằng cách gõ:\n\n`/phongban <MÃ> <Họ và tên>`  – ví dụ: `/phongban KHDA Nguyễn Văn A`\n\n"
            f"{department_menu(policy)}\n\n"
            + ("Khai báo được áp dụng ngay với vai trò nhân viên." if policy.auto_approve else
               "Khai báo sẽ được **trưởng phòng hoặc quản trị hệ thống duyệt**; sau khi duyệt bạn dùng được tài liệu của đơn vị mình."))


def status_text(policy: AccessPolicy, req: Optional[dict], email: str) -> str:
    row = policy.users.get(email)
    if row:
        dept = policy.departments.get(row["phong"])
        return f"Tài khoản của bạn thuộc **{dept.name if dept else row['phong']}**, vai trò `{row['vai_tro']}`."
    if req and req["trang_thai"] == CHO_DUYET:
        dept = policy.departments.get(req["phong"])
        return (f"Khai báo **{dept.name if dept else req['phong']}** (ngày {req['ngay_khai']}) đang **chờ duyệt**. "
                "Liên hệ trưởng phòng hoặc quản trị hệ thống để được duyệt.")
    if req and req["trang_thai"] == TU_CHOI:
        return ("Khai báo trước của bạn **không được duyệt**" + (f" ({req['ghi_chu']})" if req.get("ghi_chu") else "")
                + ". Bạn có thể khai báo lại bằng `/phongban <MÃ> <Họ và tên>`.")
    return onboarding(policy)


@dataclass
class CommandResult:
    text: str
    outcome: str              # dùng cho nhật ký truy cập


def handle_command(name: str, rest: str, email: str, policy: AccessPolicy, regs: Registrations) -> CommandResult:
    """Xử lý /phongban, /choduyet, /duyet, /tuchoi. Hoàn toàn tra bảng, không gọi LLM."""
    email = (email or "").strip().lower()
    if not _EMAIL.match(email):
        return CommandResult("Không xác định được tài khoản đăng nhập (LibreChat chưa gửi email người dùng). "
                             "Liên hệ quản trị hệ thống.", "khong_ro_nguoi_dung")
    args = rest.split()
    if name == "phongban":
        if not args:
            return CommandResult(status_text(policy, regs.get(email), email), "xem_khai_bao")
        code = args[0].upper()
        if code not in policy.departments:
            return CommandResult(f"Mã `{_clean(args[0], 20)}` không có trong danh mục. Chọn một mã dưới đây:\n\n"
                                 f"{department_menu(policy)}", "khai_bao_sai")
        if not policy.email_allowed(email):
            return CommandResult("Email này không thuộc tên miền được phép khai báo. Hãy đăng ký LibreChat bằng email "
                                 "công ty.", "khai_bao_sai_mien")
        current = policy.users.get(email)
        if current and current["phong"] == code:
            return CommandResult(status_text(policy, None, email), "xem_khai_bao")
        ho_ten = " ".join(args[1:])
        if not ho_ten.strip():
            return CommandResult("Vui lòng ghi kèm họ và tên, ví dụ: `/phongban KHDA Nguyễn Văn A`", "khai_bao_sai")
        regs.declare(email, code, ho_ten)
        dept = policy.departments[code]
        if policy.auto_approve and not current:
            regs.decide(email, True, "tu_dong")
            return CommandResult(f"Đã ghi nhận bạn thuộc **{dept.name}** (vai trò nhân viên). Bạn có thể bắt đầu đặt câu hỏi.",
                                 "khai_bao_tu_dong_duyet")
        extra = (" Quyền hiện tại của bạn được giữ nguyên cho tới khi khai báo mới được duyệt." if current else "")
        return CommandResult(f"Đã ghi nhận khai báo **{dept.name}**, đang **chờ duyệt** bởi trưởng phòng hoặc quản trị "
                             f"hệ thống.{extra}", "khai_bao")

    right = policy.approval_right(email)
    if right is None:
        return CommandResult("Bạn không có quyền duyệt khai báo.", "khong_co_quyen_duyet")
    own = policy.users[email]["phong"]
    if name == "choduyet":
        items = regs.pending(None if right == "tat_ca" else own)
        if not items:
            return CommandResult("Không có khai báo nào đang chờ duyệt.", "xem_cho_duyet")
        lines = "\n".join(f"| {r['email']} | {r['ho_ten']} | {r['phong']} | {r['ngay_khai']} |" for r in items)
        return CommandResult("| Email | Họ tên | Đơn vị | Ngày khai |\n|---|---|---|---|\n" + lines +
                             "\n\nDuyệt: `/duyet <email>` · Từ chối: `/tuchoi <email> <lý do>`", "xem_cho_duyet")

    if not args or not _EMAIL.match(args[0]):
        return CommandResult(f"Cú pháp: `/{name} <email>`" + (" [vai trò]" if name == "duyet" else " <lý do>"), "duyet_sai")
    target = args[0].lower()
    req = regs.get(target)
    if not req or req["trang_thai"] != CHO_DUYET:
        return CommandResult(f"Không có khai báo đang chờ duyệt của {target}.", "duyet_sai")
    if right == "phong_minh" and req["phong"] != own:
        return CommandResult(f"{target} khai báo đơn vị `{req['phong']}`; bạn chỉ duyệt được khai báo của đơn vị mình.",
                             "khong_co_quyen_duyet")
    if name == "tuchoi":
        regs.decide(target, False, email, note=" ".join(args[1:]))
        return CommandResult(f"Đã từ chối khai báo của {target}.", "tu_choi_khai_bao")
    vai_tro = args[1] if len(args) > 1 else "nhan_vien"
    if vai_tro not in policy.roles:
        return CommandResult(f"Vai trò `{_clean(vai_tro, 30)}` không có. Các vai trò: {', '.join(policy.roles)}", "duyet_sai")
    if right == "phong_minh" and vai_tro != "nhan_vien":
        return CommandResult("Trưởng phòng chỉ cấp được vai trò nhan_vien; vai trò khác do quản trị hệ thống cấp.",
                             "khong_co_quyen_duyet")
    regs.decide(target, True, email, vai_tro)
    dept = policy.departments[req["phong"]]
    return CommandResult(f"Đã duyệt {target} ({req['ho_ten']}) vào **{dept.name}**, vai trò `{vai_tro}`. "
                         "Có hiệu lực ở câu hỏi tiếp theo của người dùng.", "duyet_khai_bao")
