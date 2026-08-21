"""Provider-independent answer generation adapters."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .config import DEFAULT_CONFIG
from .prompts import SYSTEM_PROMPTS, build_user_prompt


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, contexts: list[dict[str, Any]], language: str) -> str:
        """Generate an answer grounded in contexts."""


class ContextPreviewGenerator(AnswerGenerator):
    """Deterministic, API-free preview for application and demo checks."""

    def generate(self, query: str, contexts: list[dict[str, Any]], language: str) -> str:
        if language == "ko":
            return (
                "LLM API를 호출하지 않은 근거 미리보기입니다. 아래 Sources의 검색 결과를 "
                "확인한 뒤 `--generate --provider openai`로 근거 기반 답변을 생성할 수 있습니다."
            )
        return (
            "This is an API-free context preview. Review the retrieved Sources below, "
            "or use `--generate --provider openai` for a grounded answer."
        )


class OpenAIAnswerGenerator(AnswerGenerator):
    """Optional OpenAI Responses API adapter; credentials are never stored in code."""

    def __init__(self, model: str = DEFAULT_CONFIG.openai_model) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI generator.")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install the optional `openai` package to use generation.") from error
        self.model = model
        self.client = OpenAI()

    def generate(self, query: str, contexts: list[dict[str, Any]], language: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPTS[language],
            input=build_user_prompt(query, contexts, language),
        )
        return response.output_text.strip()


def build_generator(provider: str) -> AnswerGenerator:
    if provider == "preview":
        return ContextPreviewGenerator()
    if provider == "openai":
        return OpenAIAnswerGenerator()
    raise ValueError(f"Unsupported generator provider: {provider}")

