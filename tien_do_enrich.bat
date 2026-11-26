@echo off
REM Xem tien do buoc AI bo sung metadata (khong goi API, chi doc so ghi chi phi).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('data/enrich_ai_ledger.sqlite'); n,last=c.execute(\"select count(*),max(created_at) from enrich_ai where version='ai-v8'\").fetchone(); s=c.execute('select sum(cost_usd) from enrich_ai').fetchone()[0]; print(f'Ban ai-v8: da xong {n} / toi da 2915 van ban; lan cuoi: {last}; tong da chi {s:.3f} USD (han muc: dong ENRICH_AI_BUDGET_USD trong .env)')"
echo.
echo --- 5 dong cuoi nhat ky ---
powershell -NoProfile -Command "Get-Content data\enrich_ai_full.log -Tail 5 -Encoding UTF8"
pause
