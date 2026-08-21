# Finance RAG Agent Implementation Summary

## Implemented files

- `config.py`: fixed paths, models, retrieval depths, corpus/cache provenance hashes
- `retriever.py`: validated cache loading, BGE-M3 Dense Top-20, reranker Top-5, metadata-rich results
- `prompts.py`: bilingual grounding and source templates
- `generator.py`: provider-independent interface, API-free preview, optional OpenAI Responses adapter
- `rag_agent.py`: language detection and retrieval/generation orchestration
- `cli.py`: interactive or one-shot retrieve-only/generation demo
- `examples/sample_queries.json`: four English and two Korean smoke-test questions
- `README.md`: setup, use, output, grounding, and limitations

## Connection to validated retrieval

The application reproduces the finalized architecture rather than importing the experiment runner. It uses the same normalized BGE-M3 query embedding, matrix-product Dense scoring, Top-20 depth, `bge-reranker-v2-m3` query–chunk pairs, max length 512, descending reranker sort, and Top-5 depth.

The source corpus is the unchanged Final 400/60 `chunks.jsonl` with 563 chunks. The existing `(563, 1024)` embedding cache is reused only after exact corpus and cache SHA-256 validation, which also protects chunk-order provenance. The application never writes evaluation assets or silently recomputes the cache.

## Language and generation

Hangul presence deterministically selects Korean; otherwise English is selected. Both languages use the same multilingual Dense→Reranker path without translation.

`AnswerGenerator` is provider independent. `ContextPreviewGenerator` works without credentials. `OpenAIAnswerGenerator` is optional, lazily imports the SDK, reads `OPENAI_API_KEY`, and uses `OPENAI_MODEL` (default `gpt-5-mini`). No key is stored in code. No live LLM API call was made during implementation because no credential is required for retrieval validation.

The bilingual prompt restricts answers to supplied evidence, requires exact financial numbers/conditions, asks for an explicit insufficiency statement, avoids definitive legal/financial claims, and maps citations to the five actual retrieved sources.

## Application sanity result

Static imports, JSON parsing, language routing, fixed corpus count, cache/corpus hashes, CLI help, and six actual retrieve-only questions passed. Every question returned exactly five sources with real chunk/document/title/page metadata.

Representative outputs:

- English account-opening query: Top sources included `WEB003_0001` and `PDF004_0001` (`How Foreigners Can Open a Bank Account in Korea`, p.1).
- English remittance query: rank 1 was `PDF001_0189` (`Financial Guide Book for Foreigners in Korea`, p.109).
- English card query: rank 1 was `PDF001_0099`, p.62.
- English voice-phishing query: rank 1 was `PDF008_0019` (`사기 예방 백과사전`, p.15).
- Korean account-opening query: rank 1 was `PDF001_0026`, p.21.
- Korean voice-phishing query: rank 1 was `PDF008_0019`, p.15.

This was an application smoke test, not a retrieval benchmark or architecture retuning exercise.

## Currently usable

- Korean and English retrieval with validated Top-20→Top-5 settings
- Interactive and one-shot CLI
- Structured JSON output
- API-free retrieve-only and deterministic preview
- Optional OpenAI grounded generation when the SDK/key/network are available

## Still needed for production

- Provision and approve an LLM provider/model and operational credentials
- Measure end-to-end latency and resource requirements in the deployment environment
- Add generated-citation validation and safety/PII handling appropriate to the service
- Add conversation state, observability, timeout/retry policies, and user-facing disclaimers
- Run downstream answer-quality/hallucination evaluation separately from the already-frozen retrieval evaluation

