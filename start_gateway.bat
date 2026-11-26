@echo off
REM Khoi dong dich vu hoi dap 1 luot "Hanoi Metro" cho LibreChat (http://127.0.0.1:8766/v1).
REM Can: Docker Desktop dang chay (Qdrant hanoi_metro_qdrant), Ollama dang chay (embedding bge-m3).
REM Thuong chay qua start_all.bat.
title Hanoi Metro Gateway
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m src.gateway.server
echo.
echo Dich vu hoi dap da dung. Xem thong bao loi phia tren (neu co).
pause
