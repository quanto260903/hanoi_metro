import asyncio

from src.agent.llm import LLMError
from src.agent import mcp_server

EXPECTED_TOOLS = {"search_documents", "ask_documents", "summarize_document", "draft_report", "list_documents", "read_document"}


import pytest


@pytest.fixture(autouse=True)
def _access_off(monkeypatch):
    """Kiểm thử không phụ thuộc .env thật: mặc định phân quyền tắt; bài kiểm thử khóa tự bật lại."""
    monkeypatch.setattr(mcp_server.settings.agent, "access_control", False)


def test_tools_locked_when_access_control_on(monkeypatch):
    monkeypatch.setattr(mcp_server.settings.agent, "access_control", True)
    text = _text(asyncio.run(mcp_server.server.call_tool("list_documents", {})))
    assert text == mcp_server.MCP_LOCKED


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


def test_server_exposes_document_tools():
    tools = asyncio.run(mcp_server.server.list_tools())
    assert EXPECTED_TOOLS <= {t.name for t in tools}


def test_list_documents_tool_reads_store(indexed_store, monkeypatch):
    monkeypatch.setitem(mcp_server.runtime.__dict__, "store", indexed_store)
    result = asyncio.run(mcp_server.server.call_tool("list_documents", {"loai_vb": ["CV"]}))
    text = _text(result)
    assert "916/CV-VT&QLPT" in text and "Tổng cộng 1 văn bản" in text


class _BrokenAssistant:
    def ask(self, question, filters):
        raise LLMError("llm_unavailable", "http://10.0.0.5:11434 connection refused – secret stack")


def test_llm_failure_returns_safe_message(monkeypatch):
    monkeypatch.setitem(mcp_server.runtime.__dict__, "assistant", _BrokenAssistant())
    text = _text(asyncio.run(mcp_server.server.call_tool("ask_documents", {"question": "x"})))
    assert "không phản hồi" in text
    assert "10.0.0.5" not in text and "secret" not in text
