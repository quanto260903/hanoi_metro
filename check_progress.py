"""Theo dõi tiến độ trích text trên máy (corpus local) – chỉ đọc corpus.db, không ảnh hưởng tiến trình đang chạy.

Chạy trong một cửa sổ PowerShell khác:  .venv\\Scripts\\python.exe check_progress.py
Dừng bằng Ctrl+C. Nếu số trang không tăng sau vài lần in liên tiếp thì tiến trình có thể đang treo.
"""

import sqlite3
import sys
import time

TOTAL = 53_471 + 7_717      # trang text gốc + trang trắng trong toàn kho (theo ingest_manifest)
EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def count() -> int:
    con = sqlite3.connect("file:data/corpus.db?mode=ro", uri=True, timeout=30)
    try:
        return con.execute("select count(*) from local_pages").fetchone()[0]
    finally:
        con.close()


prev, stalled = count(), 0
print(f"Đã xong {prev:,}/{TOTAL:,} trang. Cập nhật mỗi {EVERY}s, Ctrl+C để thoát.")
while True:
    time.sleep(EVERY)
    now = count()
    rate = (now - prev) / EVERY
    stalled = stalled + 1 if now == prev else 0
    eta = f"còn ~{(TOTAL - now) / rate / 3600:.1f} giờ" if rate > 0 else "KHÔNG TĂNG"
    warn = f"  <-- đứng yên {stalled} lần liên tiếp, kiểm tra cửa sổ đang chạy" if stalled >= 3 else ""
    print(f"{time.strftime('%H:%M:%S')}  {now:,}/{TOTAL:,} ({now / TOTAL:.1%})  +{now - prev} trang  {rate:.1f} trang/s  {eta}{warn}")
    prev = now
