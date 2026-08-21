"""Runtime configuration for the validated Finance RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AgentConfig:
    chunks_path: Path = ROOT_DIR / "Retriever_dataset-20260816T123809Z-1-001" / "chunks" / "chunk_400_60" / "chunks.jsonl"
    embedding_cache_path: Path = ROOT_DIR / "retrieval_eval" / "cache" / "dense_emb_400_60.npy"
    dense_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    candidate_k: int = 20
    final_k: int = 5
    expected_chunk_count: int = 563
    expected_chunks_sha256: str = "085d8ddf536741e227467bf00f5d67fa5d88b4ab396f82e91834ca6714aa1b3b"
    expected_cache_sha256: str = "4320086ce98c11a882d45f67c3cb80a82d4f7c2e89de349f7a358ece977b11c6"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")


DEFAULT_CONFIG = AgentConfig()

