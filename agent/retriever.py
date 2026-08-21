"""Validated BGE-M3 Dense Top-20 → reranker Top-5 retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import AgentConfig, DEFAULT_CONFIG


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FinanceRetriever:
    """Application wrapper matching the finalized evaluation implementation."""

    def __init__(self, config: AgentConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.chunks = self._load_and_validate_chunks()
        self.chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self._dense_model = None
        self._reranker_model = None
        self._embeddings = None

    def _load_and_validate_chunks(self) -> list[dict[str, Any]]:
        path = self.config.chunks_path
        if not path.is_file():
            raise FileNotFoundError(f"Final chunks file not found: {path}")
        if _sha256(path) != self.config.expected_chunks_sha256:
            raise ValueError("Final chunks hash mismatch; refusing to use an unvalidated corpus/order.")
        with path.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        if len(chunks) != self.config.expected_chunk_count:
            raise ValueError(f"Expected {self.config.expected_chunk_count} chunks, found {len(chunks)}.")
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Duplicate chunk_id found in Final corpus.")
        return chunks

    def _ensure_models(self) -> None:
        if self._dense_model is not None:
            return
        import numpy as np
        from sentence_transformers import CrossEncoder, SentenceTransformer

        cache_path = self.config.embedding_cache_path
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Validated embedding cache not found: {cache_path}. "
                "This application does not silently rebuild evaluation assets."
            )
        if _sha256(cache_path) != self.config.expected_cache_sha256:
            raise ValueError("Embedding cache hash mismatch; refusing unsafe cache reuse.")
        embeddings = np.load(cache_path)
        if embeddings.shape != (self.config.expected_chunk_count, 1024):
            raise ValueError(f"Expected embedding shape (563, 1024), found {embeddings.shape}.")

        self._dense_model = SentenceTransformer(self.config.dense_model)
        self._reranker_model = CrossEncoder(self.config.reranker_model, max_length=512)
        self._embeddings = embeddings

    def retrieve(
        self,
        query: str,
        *,
        candidate_k: int | None = None,
        final_k: int | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        candidate_k = candidate_k or self.config.candidate_k
        final_k = final_k or self.config.final_k
        if candidate_k != self.config.candidate_k or final_k != self.config.final_k:
            raise ValueError("Validated production depths are fixed at candidate_k=20 and final_k=5.")

        self._ensure_models()
        import numpy as np

        query_embedding = self._dense_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = self._embeddings @ query_embedding
        candidate_indices = np.argsort(-dense_scores)[:candidate_k]
        candidates = [
            (self.chunks[index]["chunk_id"], float(dense_scores[index]))
            for index in candidate_indices
        ]

        pairs = [(query, self.chunk_by_id[chunk_id]["text"]) for chunk_id, _ in candidates]
        reranker_scores = self._reranker_model.predict(pairs)
        reranked = sorted(
            (
                (chunk_id, retrieval_score, float(reranker_score))
                for (chunk_id, retrieval_score), reranker_score in zip(
                    candidates, reranker_scores, strict=True
                )
            ),
            key=lambda item: item[2],
            reverse=True,
        )[:final_k]

        return [
            self._format_result(rank, chunk_id, retrieval_score, reranker_score)
            for rank, (chunk_id, retrieval_score, reranker_score) in enumerate(reranked, 1)
        ]

    def _format_result(
        self,
        rank: int,
        chunk_id: str,
        retrieval_score: float,
        reranker_score: float,
    ) -> dict[str, Any]:
        chunk = self.chunk_by_id[chunk_id]
        metadata_keys = (
            "title", "organization", "language", "category", "source_type",
            "source_url", "page_start", "page_end", "token_count",
        )
        metadata = {key: chunk[key] for key in metadata_keys if key in chunk}
        return {
            "rank": rank,
            "chunk_id": chunk_id,
            "document_id": chunk.get("doc_id"),
            "text": chunk["text"],
            "retrieval_score": retrieval_score,
            "reranker_score": reranker_score,
            "metadata": metadata,
        }

