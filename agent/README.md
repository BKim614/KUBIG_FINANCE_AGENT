# KUBIG Finance RAG Agent

This folder connects the finalized Korean/English retrieval pipeline to an optional grounded answer generator. It is application code and does not modify or import the evaluation runner.

## Architecture

```text
Korean or English question
→ BAAI/bge-m3 Dense Top-20
→ BAAI/bge-reranker-v2-m3
→ Top-5 evidence chunks
→ optional LLM answer in the detected query language
→ actual source metadata
```

English queries are embedded directly against the unchanged Korean-majority corpus. Translation, Nori, sparse retrieval, and fusion are not in the final application path.

## Validated fixed configuration

- Corpus: `Retriever_dataset-20260816T123809Z-1-001/chunks/chunk_400_60/chunks.jsonl`
- Corpus size: exactly 563 chunks
- Dense model: `BAAI/bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`, max length 512
- Candidate/final depth: 20/5; the application rejects different values
- Cache: `retrieval_eval/cache/dense_emb_400_60.npy`, shape `(563, 1024)`

The retriever verifies the SHA-256 of both the chunks file and cache before reuse. A mismatch fails closed instead of silently rebuilding or using embeddings in an uncertain chunk order.

## Requirements

Retrieval uses the environment already used by evaluation:

```bash
pip install numpy sentence-transformers
```

OpenAI generation is optional:

```bash
pip install openai
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-mini"  # optional override
```

Keys are read only from environment variables and must not be committed. The adapter uses the OpenAI Responses API (`instructions`, `input`, and `output_text`).

## CLI

API-free retrieval:

```bash
python -m agent.cli --retrieve-only \
  "What documents do I need to open a bank account in Korea?"

python -m agent.cli --retrieve-only \
  "외국인이 한국에서 통장을 만들려면 어떤 서류가 필요한가요?"
```

Interactive mode is also retrieval-only by default:

```bash
python -m agent.cli --retrieve-only
```

Generate a grounded answer with OpenAI:

```bash
python -m agent.cli --generate --provider openai \
  "What documents do I need to open a bank account in Korea?"
```

Deterministic context-preview answer without an API:

```bash
python -m agent.cli --generate --provider preview \
  "보이스피싱 전화를 믿고 돈을 보냈다면 지금 무엇을 해야 하나요?"
```

Use `--json` for the full structured result, including text, scores, and metadata.

## Python API

```python
from agent import FinanceRAGAgent

agent = FinanceRAGAgent()
result = agent.ask("What documents do I need to open a bank account?", generate=False)
```

The result contains `query`, detected `language` (`ko` or `en`), optional `answer`, and five `sources`.

## Source and citation mapping

Each source is returned with its real `chunk_id`, `document_id`, chunk text, Dense score, reranker score, and only metadata present in the corpus (`title`, page, URL, organization, and related fields). Prompt sources are numbered `[Source 1]` through `[Source 5]`; the model is instructed to cite only those numbers as `[1]`–`[5]`.

## Grounding rules

- Prefer only retrieved context and state when evidence is insufficient.
- Preserve context numbers, conditions, documents, institutions, and product names.
- Answer in Korean when Hangul is present in the query; otherwise answer in English.
- Avoid definitive individualized financial or legal judgments.
- Never invent source metadata or citation numbers.

## Known limitations

- Language detection is intentionally deterministic: any Hangul routes the answer to Korean; all other input routes to English.
- Retrieval loads two local models and can be slow on first use. Reuse one `FinanceRAGAgent` instance for multiple questions.
- Retrieval quality does not guarantee that every Top-5 chunk fully answers the question.
- Citation correctness is prompt-enforced; production deployments should additionally validate generated citation markers.
- The optional API generator requires network access, the `openai` package, a valid key, and account billing. Retrieve-only never does.
- This is informational guidance, not a substitute for a bank, regulator, lawyer, or financial adviser.

