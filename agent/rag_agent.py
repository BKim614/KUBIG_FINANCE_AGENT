"""End-to-end orchestration for retrieval and optional generation."""

from __future__ import annotations

import re
from typing import Any

from .generator import AnswerGenerator, ContextPreviewGenerator
from .retriever import FinanceRetriever


HANGUL_RE = re.compile(r"[가-힣]")


def detect_language(query: str) -> str:
    """Return Korean when Hangul is present; otherwise return English."""
    return "ko" if HANGUL_RE.search(query) else "en"


class FinanceRAGAgent:
    def __init__(
        self,
        retriever: FinanceRetriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever or FinanceRetriever()
        self.generator = generator or ContextPreviewGenerator()

    def ask(self, query: str, *, generate: bool = True) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        language = detect_language(query)
        sources = self.retriever.retrieve(query)
        answer = self.generator.generate(query, sources, language) if generate else None
        return {
            "query": query,
            "language": language,
            "answer": answer,
            "sources": sources,
        }

