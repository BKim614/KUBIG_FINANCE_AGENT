"""Command-line demo for the KUBIG Finance Agent."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .generator import build_generator
from .rag_agent import FinanceRAGAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KUBIG bilingual Finance RAG Agent")
    parser.add_argument("query", nargs="?", help="Financial question; prompts interactively if omitted")
    parser.add_argument("--retrieve-only", action="store_true", help="Retrieve and rerank without an LLM call")
    parser.add_argument("--generate", action="store_true", help="Generate an answer after retrieval")
    parser.add_argument("--provider", choices=("preview", "openai"), default="preview")
    parser.add_argument("--json", action="store_true", help="Print the complete result as JSON")
    return parser


def _source_label(source: dict[str, Any]) -> str:
    metadata = source.get("metadata", {})
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    page = ""
    if page_start is not None:
        page = f", p.{page_start}" if page_start == page_end else f", pp.{page_start}-{page_end}"
    return f"{source['chunk_id']} ({metadata.get('title') or source.get('document_id') or 'unknown'}{page})"


def main() -> None:
    args = build_parser().parse_args()
    if args.retrieve_only and args.generate:
        raise SystemExit("Choose either --retrieve-only or --generate, not both.")
    query = args.query
    if not query:
        print("=" * 32)
        print("KUBIG Finance Agent")
        print("=" * 32)
        query = input("Ask a financial question:\n> ").strip()

    generate = args.generate
    # Retrieval-only must never require an SDK or API key, even if a provider
    # flag was accidentally supplied.
    generator = build_generator(args.provider if generate else "preview")
    print("Searching...", flush=True)
    result = FinanceRAGAgent(generator=generator).ask(query, generate=generate)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\nDetected language: {result['language']}")
    print(f"Retrieved {len(result['sources'])} sources.")
    if result["answer"] is not None:
        print("\nAnswer:")
        print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"[{source['rank']}] {_source_label(source)}")
        print(f"    dense={source['retrieval_score']:.4f}, reranker={source['reranker_score']:.4f}")


if __name__ == "__main__":
    main()
