@echo off
REM Khoi dong toan bo he thong Tro ly tai lieu Hanoi Metro:
REM   1. Docker Desktop (LibreChat, MongoDB, Meilisearch, Qdrant...)
REM   2. Ollama (mo hinh nhung bge-m3 cho tim kiem)
REM   3. Dich vu hoi dap 1 luot (gateway 127.0.0.1:8766), cua so thu nho "Hanoi Metro Gateway"
REM   4. Mo LibreChat tren trinh duyet
REM Chay lai nhieu lan khong sao: buoc nao da chay thi bo qua.
REM Luu y: file .bat phai giu xuong dong CRLF va chi dung chu khong dau.
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0.."

echo [1/4] Docker Desktop...
docker info >nul 2>&1
if not errorlevel 1 goto docker_ok
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
set /a TRIES=0
:wait_docker
timeout /t 5 /nobreak >nul
set /a TRIES+=1
docker info >nul 2>&1
if not errorlevel 1 goto docker_ok
if %TRIES% GEQ 60 goto docker_fail
goto wait_docker
:docker_fail
echo     Docker Desktop khong khoi dong duoc sau 5 phut. Mo Docker Desktop bang tay roi chay lai file nay.
pause
exit /b 1
:docker_ok
pushd "%ROOT%"
docker compose up -d >nul 2>&1
popd
docker start hanoi_metro_qdrant >nul 2>&1
echo     OK

echo [2/4] Ollama (bge-m3)...
curl -s -o nul --max-time 3 http://127.0.0.1:11434/api/version
if not errorlevel 1 goto ollama_ok
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" goto ollama_app
start "Ollama" /min ollama serve
goto ollama_wait
:ollama_app
start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
:ollama_wait
timeout /t 8 /nobreak >nul
:ollama_ok
echo     OK

echo [3/4] Dich vu hoi dap Hanoi Metro...
curl -s -o nul --max-time 3 http://127.0.0.1:8766/v1/models
if not errorlevel 1 goto gateway_ok
start "Hanoi Metro Gateway" /min cmd /k "%~dp0start_gateway.bat"
set /a TRIES=0
:wait_gateway
timeout /t 3 /nobreak >nul
set /a TRIES+=1
curl -s -o nul --max-time 3 http://127.0.0.1:8766/v1/models
if not errorlevel 1 goto gateway_ok
if %TRIES% GEQ 40 goto gateway_fail
goto wait_gateway
:gateway_fail
echo     Gateway chua len sau 2 phut. Xem loi trong cua so "Hanoi Metro Gateway".
pause
exit /b 1
:gateway_ok
echo     OK

echo [4/4] Mo LibreChat...
set /a TRIES=0
:wait_librechat
curl -s -o nul --max-time 3 http://127.0.0.1:3080
if not errorlevel 1 goto librechat_ok
set /a TRIES+=1
if %TRIES% GEQ 40 goto librechat_ok
timeout /t 3 /nobreak >nul
goto wait_librechat
:librechat_ok
start "" http://localhost:3080
echo.
echo He thong da san sang. Trong LibreChat chon nhom "Tro ly tai lieu Hanoi Metro".
echo Muon tat dich vu hoi dap: dong cua so "Hanoi Metro Gateway".
timeout /t 10 >nul
endlocal
