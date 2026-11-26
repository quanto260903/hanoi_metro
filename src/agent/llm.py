"""
Hanoi Metro AI Agent - Chat Model Clients.
- API tương thích OpenAI: Gemini (generativelanguage.googleapis.com/v1beta/openai), vLLM, OpenAI...
  LLM_REASONING_EFFORT=none tắt chế độ thinking của Gemini 2.5 Flash (không phát sinh token suy luận tính giá đầu ra).
- Ollama nội bộ: gọi API gốc /api/chat để đặt num_ctx (endpoint /v1 của Ollama bỏ qua num_ctx và dùng context mặc định
  2–4k token, làm cắt cụt âm thầm phần tài liệu đưa vào).
Cả hai đều có `stream()` để dịch vụ hỏi đáp trả chữ dần cho LibreChat.
"""

import json
import logging
import os
from typing import Iterator, Optional, Protocol, Tuple
from urllib.parse import urlparse

import httpx

from src.config import settings

logger = logging.getLogger("hanoi_metro.llm")


class LLMError(RuntimeError):
    """Lỗi gọi LLM; `code` ổn định để tầng giao tiếp ánh xạ thành thông báo cho người dùng."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ChatModel(Protocol):
    external: bool   # True: dữ liệu gửi ra dịch vụ bên ngoài máy chủ nội bộ

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str: ...

    def stream(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Iterator[str]: ...


def token_limit_kwargs(base_url: str, limit: int) -> dict:
    """OpenAI đời mới (gpt-5, gpt-6...) từ chối `max_tokens`, bắt buộc `max_completion_tokens`; Gemini/vLLM/Ollama dùng `max_tokens`."""
    host = (urlparse(base_url).hostname or "").lower()
    return {"max_completion_tokens": limit} if host.endswith("openai.com") else {"max_tokens": limit}


def is_external_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return not (host in ("localhost", "127.0.0.1", "::1", "host.docker.internal") or host.startswith(("10.", "192.168.", "172.")))


class OllamaChatModel:
    external = False

    def __init__(self, base_url: str, model: str, num_ctx: int, timeout: float, temperature: float, max_tokens: int):
        self.url = base_url.rstrip("/").removesuffix("/v1") + "/api/chat"
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _payload(self, system: str, user: str, max_tokens: Optional[int], temperature: Optional[float], stream: bool) -> dict:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        return "".join(self.stream(system, user, max_tokens, temperature)).strip()

    def stream(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Iterator[str]:
        try:
            with httpx.stream("POST", self.url, json=self._payload(system, user, max_tokens, temperature, True), timeout=self.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = (data.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if data.get("done") and data.get("done_reason") == "length":
                        logger.warning("LLM dừng vì hết num_predict; câu trả lời có thể bị cụt.")
        except httpx.TimeoutException as e:
            raise LLMError("llm_timeout", f"LLM '{self.model}' quá thời gian {self.timeout}s") from e
        except httpx.HTTPError as e:
            raise LLMError("llm_unavailable", f"Không gọi được LLM '{self.model}' tại {self.url}: {e}") from e


class OpenAIChatModel:
    def __init__(self, temperature: float, max_tokens: int, reasoning_effort: str = ""):
        self.client = settings.get_llm_client()
        self.model = settings.llm.model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        # mô hình suy luận (reasoning_effort khác none) chỉ nhận temperature mặc định → không gửi tham số này
        self.reasoning = reasoning_effort not in ("", "none")
        self.external = is_external_url(settings.llm.base_url)
        self.last_usage: Optional[Tuple[int, int]] = None      # (token vào, token ra kể cả suy luận) của lần gọi gần nhất

    def _create(self, system: str, user: str, max_tokens: Optional[int], temperature: Optional[float], stream: bool):
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **({} if self.reasoning else {"temperature": self.temperature if temperature is None else temperature}),
                stream=stream,
                **token_limit_kwargs(settings.llm.base_url, max_tokens or self.max_tokens),
                **self.extra,
            )
        except Exception as e:
            if getattr(e, "status_code", None) == 429:
                code = "llm_rate_limited"   # hết hạn mức gọi API (theo phút hoặc theo ngày của gói đang dùng)
            elif "timeout" in type(e).__name__.lower():
                code = "llm_timeout"
            else:
                code = "llm_unavailable"
            raise LLMError(code, f"Không gọi được LLM '{self.model}': {type(e).__name__}") from e

    def complete(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        resp = self._create(system, user, max_tokens, temperature, stream=False)
        usage = getattr(resp, "usage", None)
        self.last_usage = (int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)) if usage else None
        return (resp.choices[0].message.content or "").strip()

    def stream(self, system: str, user: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Iterator[str]:
        try:
            for event in self._create(system, user, max_tokens, temperature, stream=True):
                if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except LLMError:
            raise
        except Exception as e:
            raise LLMError("llm_unavailable", f"LLM '{self.model}' ngắt kết nối giữa chừng: {type(e).__name__}") from e


def create_chat_model() -> ChatModel:
    cfg = settings.llm
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider == "ollama" or (provider == "auto" and "11434" in cfg.base_url):
        return OllamaChatModel(
            base_url=cfg.base_url,
            model=cfg.model_name,
            num_ctx=int(os.getenv("LLM_NUM_CTX", "16384")),
            timeout=float(cfg.timeout_seconds),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    return OpenAIChatModel(cfg.temperature, cfg.max_tokens, os.getenv("LLM_REASONING_EFFORT", ""))
