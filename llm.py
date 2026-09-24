"""Thin client for the local DeepSeek V4 Flash (vLLM, OpenAI-compatible) on the Spark.

No API key: the endpoint has no auth. LLM_BASE_URL / LLM_MODEL follow the
convention used across ~/code, so this moves to the LiteLLM gateway by env alone.
"""
import os
from collections.abc import Iterator

from openai import OpenAI

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash-dspark")

_client = OpenAI(base_url=LLM_BASE_URL, api_key=os.environ.get("LLM_API_KEY", "not-needed"), timeout=300)


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 2048) -> str:
    resp = _client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content or ""


def stream(messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096) -> Iterator[str]:
    resp = _client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
    )
    for event in resp:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content
