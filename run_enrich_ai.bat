@echo off
REM Chay buoc AI bo sung metadata cho toan kho (co the chay lai: van ban da xong se duoc bo qua).
REM Nhat ky: data\enrich_ai_full.log
title Hanoi Metro Enrich AI
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -m src.indexing.enrich chay --ai --chi-ai >> data\enrich_ai_full.log 2>&1
echo Xong (ma thoat %ERRORLEVEL%). Xem data\enrich_ai_full.log
