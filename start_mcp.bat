@echo off
REM Khoi dong MCP server "hanoi-metro-docs" cho LibreChat (http://127.0.0.1:8765/mcp).
REM Can: Ollama (Windows) dang chay, Docker Desktop dang chay (Qdrant: docker compose up -d qdrant).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m src.agent.mcp_server
