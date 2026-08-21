"""Grounded bilingual prompt templates."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPTS = {
    "ko": """당신은 외국인의 한국 금융생활을 돕는 정보 안내 AI입니다.
제공된 검색 근거만 우선 사용해 한국어로 답하세요.
근거에 없는 사실을 만들지 말고, 숫자·기간·서류·기관·상품명은 근거와 정확히 일치시켜야 합니다.
근거가 부족하거나 서로 충돌하면 그 한계를 명확히 말하세요.
개별 상황에 대한 확정적인 금융·법률 판단 대신 해당 금융기관 또는 공식 기관 확인이 필요할 수 있음을 안내하세요.
사실 주장 뒤에는 실제 근거 번호만 [1], [2] 형식으로 인용하세요. 제공되지 않은 번호를 인용하지 마세요.""",
    "en": """You are an information assistant for foreigners navigating financial services in Korea.
Answer in English and prioritize only the retrieved context supplied below.
Do not invent facts. Numbers, time periods, documents, institutions, and product names must match the context exactly.
If the evidence is insufficient or conflicting, say so clearly.
Avoid presenting individualized financial or legal guidance as definitive; advise checking with the relevant bank or official institution when appropriate.
Cite factual claims only with the real source numbers [1], [2], etc. Do not cite a source number that was not provided.""",
}


def format_contexts(contexts: list[dict[str, Any]]) -> str:
    blocks = []
    for index, context in enumerate(contexts, 1):
        metadata = context.get("metadata", {})
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        page = page_start if page_start == page_end else f"{page_start}-{page_end}"
        blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"chunk_id: {context.get('chunk_id') or 'N/A'}",
                    f"document: {context.get('document_id') or 'N/A'}",
                    f"title: {metadata.get('title') or 'N/A'}",
                    f"page: {page if page_start is not None else 'N/A'}",
                    f"url: {metadata.get('source_url') or 'N/A'}",
                    "text:",
                    context.get("text", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, contexts: list[dict[str, Any]], language: str) -> str:
    context_text = format_contexts(contexts)
    question_label = "질문" if language == "ko" else "Question"
    return f"{context_text}\n\n{question_label}:\n{query}"

