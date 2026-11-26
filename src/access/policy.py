"""
Hanoi Metro AI Agent - Phân quyền theo phòng ban.

Quyết định "ai được xem gì, dùng chức năng nào" là TRA BẢNG, không dùng AI:
  người dùng (email) → phòng, vai trò          : config/access/nguoi_dung.csv (do Phòng TCNS / quản trị cập nhật)
  vai trò → phạm vi xem, chức năng             : config/access/vai_tro.yaml
  văn bản → phòng (nhãn đã duyệt)              : bảng doc_access trong kho (nạp bằng src.access.labels)
Mô hình ngôn ngữ chỉ nhận những đoạn đã qua bộ lọc này, nên không thể mở rộng quyền dù bị lừa hay "bịa".

Nguyên tắc mặc định chặn: văn bản chưa gán nhãn chỉ vai trò xem toàn bộ thấy được; email lạ không thấy gì
(hoặc chỉ tài liệu chung, tùy cấu hình).
"""

import csv
import hashlib
import hmac
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import yaml

from src.store.base import AccessScope

logger = logging.getLogger("hanoi_metro.access")

CHUNG = "CHUNG"                                   # nhãn tài liệu chung, mọi người dùng đã đăng ký đều xem được
FEATURES = ("hoi_dap", "tom_tat", "soan_thao", "liet_ke")


@dataclass(frozen=True)
class Department:
    code: str
    name: str
    aliases: FrozenSet[str] = frozenset()        # mã đơn vị xuất hiện trong số hiệu: "VT&QLPT", "VTQLPT"...
    folders: tuple = ()                           # mẫu đường dẫn thư mục của phòng (gợi ý nhãn)
    mo_ta: str = ""                               # chức năng nhiệm vụ tóm tắt (dùng cho bước gợi ý nhãn)
    keywords: tuple = ()                          # từ khóa lập danh sách văn bản ứng viên (không tự gán nhãn)
    doc_codes: tuple = ()                         # tiền tố mã tài liệu kỹ thuật (ma_tai_lieu) – luật cố định khi chia


@dataclass(frozen=True)
class Principal:
    """Người đang hỏi, sau khi tra bảng. scope=None: xem toàn bộ kho."""
    email: str
    ho_ten: str = ""
    phong: str = ""
    vai_tro: str = ""
    scope: Optional[AccessScope] = None
    features: FrozenSet[str] = frozenset()
    registered: bool = True

    @property
    def user_key(self) -> str:
        return user_key(self.email)


def user_key(email: str) -> str:
    """Định danh ngắn, không lộ email (dùng trong liên kết mở file và nhật ký)."""
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


@dataclass
class AccessPolicy:
    enabled: bool = False
    departments: Dict[str, Department] = field(default_factory=dict)
    roles: Dict[str, dict] = field(default_factory=dict)
    users: Dict[str, dict] = field(default_factory=dict)          # email (chữ thường) → hàng trong nguoi_dung.csv
    grants: Dict[str, FrozenSet[str]] = field(default_factory=dict)  # email → doc_id được cấp riêng
    unknown_user: str = "tu_choi"                                  # tu_choi | chi_tai_lieu_chung
    auto_approve: bool = False                                     # khai báo /phongban được duyệt ngay (thí điểm)
    email_domains: FrozenSet[str] = frozenset()                    # tên miền được khai báo; rỗng = không giới hạn
    problems: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ tra quyền
    def resolve(self, email: str) -> Principal:
        email = (email or "").strip().lower()
        if not self.enabled:
            return Principal(email, scope=None, features=frozenset(FEATURES))
        row = self.users.get(email)
        if row is None:
            if self.unknown_user == "chi_tai_lieu_chung" and email:
                return Principal(email, scope=AccessScope(frozenset({CHUNG})), features=frozenset({"hoi_dap", "liet_ke"}),
                                 registered=False)
            return Principal(email, scope=AccessScope(frozenset()), features=frozenset(), registered=False)
        role = self.roles.get(row["vai_tro"]) or {}
        features = frozenset(f for f in role.get("chuc_nang", []) if f in FEATURES)
        grants = self.grants.get(email, frozenset())
        view = role.get("xem", [])
        if view == "tat_ca":
            return Principal(email, row.get("ho_ten", ""), row["phong"], row["vai_tro"], None, features)
        phong = set()
        for item in view or []:
            if item == "phong_minh":
                phong.add(row["phong"])
            elif item in self.departments or item == CHUNG:
                phong.add(item)
        phong.update(p for p in row.get("xem_them", []) if p in self.departments or p == CHUNG)
        return Principal(email, row.get("ho_ten", ""), row["phong"], row["vai_tro"], AccessScope(frozenset(phong), grants), features)

    def by_key(self, key: str) -> Optional[Principal]:
        """Tra người dùng từ định danh trong liên kết mở file (so khớp hằng thời gian)."""
        for email in self.users:
            if hmac.compare_digest(user_key(email), key):
                return self.resolve(email)
        return None

    def approval_right(self, email: str) -> Optional[str]:
        """Quyền duyệt khai báo của người dùng: 'tat_ca' | 'phong_minh' | None (không tùy bật/tắt phân quyền)."""
        row = self.users.get((email or "").strip().lower())
        right = (self.roles.get(row["vai_tro"]) or {}).get("duyet") if row else None
        return right if right in ("tat_ca", "phong_minh") else None

    def email_allowed(self, email: str) -> bool:
        domain = (email or "").strip().lower().rpartition("@")[2]
        return bool(domain) and (not self.email_domains or domain in self.email_domains)

    @staticmethod
    def can_view(principal: Principal, doc_id: str, labels: List[str]) -> bool:
        scope = principal.scope
        return scope is None or doc_id in scope.doc_ids or bool(scope.phong & set(labels))


# ---------------------------------------------------------------------- nạp cấu hình
def _split(value: str) -> List[str]:
    return [x.strip().upper() for x in (value or "").replace(",", ";").split(";") if x.strip()]


def load_policy(config_dir: Path, enabled: bool) -> AccessPolicy:
    """
    Đọc cấu hình phân quyền. Lỗi cấu hình khi đã BẬT phân quyền → khóa hết (không ai xem được), ghi lý do,
    thay vì chạy tiếp với bảng thiếu (fail-closed).
    """
    config_dir = Path(config_dir)
    policy = AccessPolicy(enabled=enabled)
    try:
        dept_cfg = yaml.safe_load((config_dir / "phong_ban.yaml").read_text(encoding="utf-8")) or {}
        for code, d in (dept_cfg.get("phong_ban") or {}).items():
            code = str(code).upper()
            policy.departments[code] = Department(
                code, d.get("ten", code), frozenset(str(a).upper() for a in d.get("ma_trong_so_hieu", []) or []),
                tuple(d.get("thu_muc", []) or []), (d.get("chuc_nang") or "").strip(),
                tuple(str(k) for k in d.get("tu_khoa", []) or []),
                tuple(str(k).upper() for k in d.get("ma_tai_lieu", []) or []))
        role_cfg = yaml.safe_load((config_dir / "vai_tro.yaml").read_text(encoding="utf-8")) or {}
        policy.roles = role_cfg.get("vai_tro") or {}
        policy.unknown_user = role_cfg.get("nguoi_dung_chua_dang_ky", "tu_choi")
        reg = role_cfg.get("khai_bao_phong_ban") or {}
        policy.auto_approve = reg.get("tu_dong_duyet") is True
        policy.email_domains = frozenset(str(d).strip().lower().lstrip("@") for d in reg.get("ten_mien_email") or [] if d)
        users_file = config_dir / "nguoi_dung.csv"
        if users_file.exists():
            with users_file.open(encoding="utf-8-sig", newline="") as fh:
                for i, row in enumerate(csv.DictReader(fh), start=2):
                    email = (row.get("email") or "").strip().lower()
                    if not email or email.startswith("#"):
                        continue
                    phong, vai_tro = (row.get("phong") or "").strip().upper(), (row.get("vai_tro") or "").strip()
                    if phong not in policy.departments:
                        policy.problems.append(f"nguoi_dung.csv dòng {i}: phòng '{phong}' không có trong phong_ban.yaml")
                        continue
                    if vai_tro not in policy.roles:
                        policy.problems.append(f"nguoi_dung.csv dòng {i}: vai trò '{vai_tro}' không có trong vai_tro.yaml")
                        continue
                    policy.users[email] = {"ho_ten": (row.get("ho_ten") or "").strip(), "phong": phong, "vai_tro": vai_tro,
                                           "xem_them": _split(row.get("xem_them", ""))}
        grants_file = config_dir / "cap_rieng.csv"
        if grants_file.exists():
            tmp: Dict[str, set] = {}
            with grants_file.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    email, doc_id = (row.get("email") or "").strip().lower(), (row.get("doc_id") or "").strip().lower()
                    if email and len(doc_id) == 32:
                        tmp.setdefault(email, set()).add(doc_id)
            policy.grants = {k: frozenset(v) for k, v in tmp.items()}
    except (OSError, yaml.YAMLError, AttributeError, TypeError) as e:
        policy.problems.append(f"Không đọc được cấu hình phân quyền ở {config_dir}: {type(e).__name__}: {e}")
        if enabled:
            policy.users, policy.roles, policy.grants = {}, {}, {}
    for p in policy.problems:
        logger.error(f"Phân quyền: {p}")
    if enabled:
        logger.info(f"Phân quyền BẬT: {len(policy.departments)} phòng ban, {len(policy.users)} người dùng, "
                    f"{len(policy.roles)} vai trò; email chưa đăng ký: {policy.unknown_user}")
    return policy


class PolicySource:
    """
    Nạp lại cấu hình khi file thay đổi (kiểm tra thời điểm sửa file ở mỗi yêu cầu, rất rẻ): duyệt khai báo hoặc sửa
    nguoi_dung.csv có hiệu lực ngay ở câu hỏi tiếp theo, không cần khởi động lại gateway.
    """
    FILES = ("phong_ban.yaml", "vai_tro.yaml", "nguoi_dung.csv", "cap_rieng.csv")

    def __init__(self, config_dir: Path, enabled: bool):
        self.config_dir, self.enabled = Path(config_dir), enabled
        self._lock = threading.Lock()
        self._stamp: tuple = ()
        self._policy: Optional[AccessPolicy] = None

    def _current_stamp(self) -> tuple:
        out = []
        for name in self.FILES:
            try:
                st = (self.config_dir / name).stat()
                out.append((st.st_mtime_ns, st.st_size))
            except OSError:
                out.append(None)
        return tuple(out)

    def get(self) -> AccessPolicy:
        stamp = self._current_stamp()
        with self._lock:
            if self._policy is None or stamp != self._stamp:
                self._policy, self._stamp = load_policy(self.config_dir, self.enabled), stamp
            return self._policy
