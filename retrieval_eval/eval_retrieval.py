"""
6.3 Retrieval 평가: BM25 / Dense Retrieval / Hybrid Retrieval / Hybrid + Reranker 비교.

사용법:
    python eval_retrieval.py --chunk-size 300_50 --lang ko --split test

전제:
- 데이터 위치는 이 스크립트 기준 상위 폴더(과제/)의
  Retriever_dataset-20260816T123809Z-1-001/{documents,chunks}/ 및
  rag_evaluation_dataset.jsonl 을 그대로 사용한다.
- 평가 데이터셋에는 정답 chunk_id가 없으므로, evidence(source_file/source_url + quote)를
  chunk 텍스트와 매칭해 gold chunk_id 집합을 자동 생성한다 (quote 문자열 포함 매칭,
  실패 시 문장 단위로 재시도). 매칭에 완전히 실패한 문항은 정답이 없는 것으로 보고
  Retrieval 지표 계산에서 제외하고 개수를 리포트한다.
- Dense/Hybrid/Reranker는 BGE-M3(BAAI/bge-m3) 및 bge-reranker-v2-m3를 사용한다
  (프로젝트 문서 5장에서 최종 채택한 모델).
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DOCUMENTS_PATH = ROOT_DIR / "Retriever_dataset-20260816T123809Z-1-001" / "documents" / "documents.jsonl"
CHUNKS_DIR = ROOT_DIR / "Retriever_dataset-20260816T123809Z-1-001" / "chunks"
EVAL_DATASET_PATH = ROOT_DIR / "rag_evaluation_dataset.jsonl"
CACHE_DIR = SCRIPT_DIR / "cache"

HANGUL_RE = re.compile(r"[가-힣]+")
LATIN_NUM_RE = re.compile(r"[A-Za-z0-9]+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\n+")


# ---------------------------------------------------------------------------
# 데이터 로딩 & Gold 라벨 생성
# ---------------------------------------------------------------------------

def read_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_text(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_doc_index(documents):
    """(source_type, source_file 또는 source_url) -> doc_id"""
    index = {}
    for d in documents:
        stype = d.get("source_type")
        key = d.get("source_file") if stype == "pdf" else d.get("source_url")
        key = (key or "").strip()
        if key:
            index[(stype, key)] = d["id"]
    return index


def build_gold_labels(questions, doc_index, chunks):
    chunks_by_doc = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c["doc_id"]].append(c)
    norm_cache = {}
    for c in chunks:
        norm_cache[c["chunk_id"]] = normalize_text(c["text"])

    gold = {}
    unmatched_evidence = []

    for q in questions:
        gold_chunks = set()
        for ev in q.get("evidence", []):
            stype = ev.get("source_type")
            key = ev.get("source_file") if stype == "pdf" else ev.get("source_url")
            key = (key or "").strip()
            doc_id = doc_index.get((stype, key))
            if doc_id is None:
                unmatched_evidence.append((q["id"], "doc_not_found", key))
                continue

            quote = normalize_text(ev.get("quote", ""))
            if not quote:
                continue

            candidates = chunks_by_doc.get(doc_id, [])
            matched = False
            for c in candidates:
                if quote in norm_cache[c["chunk_id"]]:
                    gold_chunks.add(c["chunk_id"])
                    matched = True

            if not matched:
                pieces = [p.strip() for p in SENT_SPLIT_RE.split(quote) if len(p.strip()) >= 8]
                for p in pieces:
                    for c in candidates:
                        if p in norm_cache[c["chunk_id"]]:
                            gold_chunks.add(c["chunk_id"])
                            matched = True

            if not matched:
                unmatched_evidence.append((q["id"], "quote_not_found", doc_id))

        gold[q["id"]] = gold_chunks

    return gold, unmatched_evidence


# ---------------------------------------------------------------------------
# Tokenizer (한글: 음절 bigram 근사 / 영숫자: 단어 단위)
# ---------------------------------------------------------------------------

def tokenize(text: str):
    text = text.lower()
    tokens = []
    for m in HANGUL_RE.finditer(text):
        word = m.group()
        tokens.append(word)
        if len(word) > 2:
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    for m in LATIN_NUM_RE.finditer(text):
        tokens.append(m.group())
    return tokens


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class BM25Retriever:
    name = "BM25"

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked]


class DenseRetriever:
    name = "Dense Retrieval (BGE-M3)"

    def __init__(self, chunks, model_name="BAAI/bge-m3", batch_size=32, cache_path=None):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.model = SentenceTransformer(model_name)

        if cache_path is not None and Path(cache_path).exists():
            self.embeddings = np.load(cache_path)
        else:
            texts = [c["text"] for c in chunks]
            self.embeddings = self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
            )
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.embeddings)

    def search(self, query, top_k=10):
        import numpy as np

        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb
        ranked = np.argsort(-scores)[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked]


def rrf_fuse(rank_lists, k=60, top_k=10):
    scores = defaultdict(float)
    for items in rank_lists:
        for rank, (chunk_id, _) in enumerate(items, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


class HybridRetriever:
    name = "Hybrid Retrieval (RRF)"

    def __init__(self, bm25_retriever, dense_retriever, k=60, candidate_k=50):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.k = k
        self.candidate_k = candidate_k

    def search(self, query, top_k=10):
        bm25_ranked = self.bm25.search(query, top_k=self.candidate_k)
        dense_ranked = self.dense.search(query, top_k=self.candidate_k)
        return rrf_fuse([bm25_ranked, dense_ranked], k=self.k, top_k=top_k)


class RerankRetriever:
    name = "Hybrid + Reranker (bge-reranker-v2-m3)"

    def __init__(self, hybrid_retriever, chunk_by_id, model_name="BAAI/bge-reranker-v2-m3", candidate_k=20):
        from sentence_transformers import CrossEncoder

        self.hybrid = hybrid_retriever
        self.chunk_by_id = chunk_by_id
        self.reranker = CrossEncoder(model_name, max_length=512)
        self.candidate_k = candidate_k

    def search(self, query, top_k=10):
        candidates = self.hybrid.search(query, top_k=self.candidate_k)
        if not candidates:
            return []
        pairs = [(query, self.chunk_by_id[cid]["text"]) for cid, _ in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(cid, float(s)) for cid, s in ranked]


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids, gold, k):
    top = set(retrieved_ids[:k])
    return len(top & gold) / len(gold)


def mrr_at_k(retrieved_ids, gold, k):
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids, gold, k):
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(retriever, questions, gold, lang, k):
    recalls, mrrs, ndcgs = [], [], []
    for q in questions:
        g = gold.get(q["id"], set())
        if not g:
            continue
        query_text = q["question"][lang]
        results = retriever.search(query_text, top_k=k)
        retrieved_ids = [cid for cid, _ in results]
        recalls.append(recall_at_k(retrieved_ids, g, k))
        mrrs.append(mrr_at_k(retrieved_ids, g, k))
        ndcgs.append(ndcg_at_k(retrieved_ids, g, k))
    n = len(recalls)
    return {
        "method": retriever.name,
        "n_questions": n,
        f"Recall@{k}": sum(recalls) / n if n else None,
        f"MRR@{k}": sum(mrrs) / n if n else None,
        f"nDCG@{k}": sum(ndcgs) / n if n else None,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrieval 비교 평가 (BM25 / Dense / Hybrid / Hybrid+Reranker)")
    parser.add_argument("--chunk-size", choices=["300_50", "400_60", "500_80"], default="300_50")
    parser.add_argument("--lang", choices=["ko", "en"], default="ko")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="test")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true", help="Dense 임베딩 캐시를 쓰지 않고 새로 계산")
    args = parser.parse_args()

    documents = read_jsonl(DOCUMENTS_PATH)
    doc_index = build_doc_index(documents)

    chunks_path = CHUNKS_DIR / f"chunk_{args.chunk_size}" / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    all_questions = read_jsonl(EVAL_DATASET_PATH)
    if args.split == "both":
        questions = all_questions
    else:
        questions = [q for q in all_questions if q["split"] == args.split]

    print(f"[data] questions={len(questions)} (split={args.split}), chunks={len(chunks)} (chunk_size={args.chunk_size})")

    gold, unmatched = build_gold_labels(questions, doc_index, chunks)
    n_with_gold = sum(1 for g in gold.values() if g)
    print(f"[gold] resolved for {n_with_gold}/{len(questions)} questions, {len(unmatched)} evidence items unmatched")
    if unmatched:
        unmatched_path = SCRIPT_DIR / f"unmatched_evidence_{args.chunk_size}.json"
        with open(unmatched_path, "w", encoding="utf-8") as f:
            json.dump(unmatched, f, ensure_ascii=False, indent=2)
        print(f"[gold] unmatched evidence detail -> {unmatched_path}")

    print("[build] BM25 index...")
    bm25 = BM25Retriever(chunks)

    print("[build] Dense index (BGE-M3, first run downloads the model)...")
    dense_cache = None if args.no_cache else CACHE_DIR / f"dense_emb_{args.chunk_size}.npy"
    dense = DenseRetriever(chunks, cache_path=dense_cache)

    hybrid = HybridRetriever(bm25, dense)

    print("[build] Reranker (bge-reranker-v2-m3, first run downloads the model)...")
    reranked = RerankRetriever(hybrid, chunk_by_id)

    results = []
    for retriever in [bm25, dense, hybrid, reranked]:
        print(f"[eval] {retriever.name} ...")
        res = evaluate_retriever(retriever, questions, gold, args.lang, k=args.k)
        results.append(res)

    out_path = SCRIPT_DIR / f"results_{args.chunk_size}_{args.lang}_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    header = f"| Method | Recall@{args.k} | MRR@{args.k} | nDCG@{args.k} | N |"
    sep = "|---|---|---|---|---|"
    print("\n" + header)
    print(sep)
    for r in results:
        rec = r[f"Recall@{args.k}"]
        mrr = r[f"MRR@{args.k}"]
        ndcg = r[f"nDCG@{args.k}"]
        print(f"| {r['method']} | {rec:.3f} | {mrr:.3f} | {ndcg:.3f} | {r['n_questions']} |")

    print(f"\n[done] results saved -> {out_path}")


if __name__ == "__main__":
    main()
