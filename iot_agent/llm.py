"""LLM client with native tool-calling and lenient JSON parsing."""
from __future__ import annotations

import ast
import json
import os
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, api_key: str, api_url: str, model: str, embed_url: str, embed_model: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}

    @classmethod
    def from_config(cls) -> "LLMClient":
        api_key = os.getenv("GLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        api_url = os.getenv("GLM_API_URL") or os.getenv("OPENAI_BASE_URL") or ""
        model = os.getenv("GLM_MODEL") or os.getenv("OPENAI_MODEL") or ""
        embed_url = os.getenv("EMBED_URL") or "https://open.bigmodel.cn/api/paas/v4/embeddings"
        embed_model = os.getenv("EMBED_MODEL") or "embedding-3"
        if not (api_key and api_url and model):
            try:
                from config import API_KEY, API_URL, MODEL
            except ImportError as exc:
                raise LLMError("missing LLM config; set GLM_API_KEY/GLM_API_URL/GLM_MODEL or create config.py") from exc
            api_key = api_key or API_KEY
            api_url = api_url or API_URL
            model = model or MODEL
        return cls(api_key, api_url, model, embed_url, embed_model)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            raise LLMError(f"LLM HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json()["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("invalid LLM response shape") from exc

    def embed(self, text: str, timeout: int = 60) -> list[float]:
        response = requests.post(
            self.embed_url,
            headers=self.headers,
            json={"model": self.embed_model, "input": text},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise LLMError(f"embedding HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json()["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("invalid embedding response shape") from exc


def strip_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip()


def normalize_text(text: str) -> str:
    replacements = {
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',
        "\u2018": "'", "\u2019": "'",
        "\uff5b": "{", "\uff5d": "}", "\uff08": "(", "\uff09": ")",
        "\uff3b": "[", "\uff3d": "]", "\uff0c": ",", "\uff1a": ":",
        "\uff1b": ";", "\u3000": " ", "\ufeff": "", "\u200b": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def parse_json_lenient(text: str) -> dict[str, Any]:
    normalized = normalize_text(strip_json(text))
    candidates = [normalized]
    start, end = normalized.find("{"), normalized.rfind("}")
    if start != -1 and end > start:
        candidates.append(normalized[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError):
            continue
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, dict):
                return value
        except (SyntaxError, ValueError):
            continue
    raise ValueError("could not parse model output as a JSON object")
