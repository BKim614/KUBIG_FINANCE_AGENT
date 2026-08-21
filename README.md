# KUBIG_FINANCE_AGENT

이 repository는 KUBIG 26기 「다국어 금융 상황 이해 AI Agent」 프로젝트의 기존 pipeline([kangminhyeok02/KUBIG_FINANCE](https://github.com/kangminhyeok02/KUBIG_FINANCE), [youhan200203/KUBIG_FINANCE](https://github.com/youhan200203/KUBIG_FINANCE))을 기반으로, **데이터/Retrieval 파이프라인 검증 및 보정 작업**을 진행한 로컬 작업본입니다.

원본 pipeline(PDF 수집, RAG 설계, retriever/reranker 구성 등)은 팀원들이 만든 것이며, 이 repository는 그 위에서 corpus 추출 품질, chunking, gold label mapping을 감사(audit)하고 발견된 문제를 보정한 결과물입니다. 프로젝트 개요·문제 정의·데이터셋 구성 등 원본 설명은 [`다국어_금융_AI_Agent_프로젝트_소개.md`](./다국어_금융_AI_Agent_프로젝트_소개.md)를 참고하세요.

## Data / Retrieval Validation Update

기존 `youhan200203/KUBIG_FINANCE` 최신본을 기준으로 corpus 추출 품질과 gold label mapping을 전수 검증하고, 발견된 문제를 보정했습니다.

### 1. PDF Extraction Validation
- 기존 `KUBIG_FINANCE`(kangminhyeok02) 버전에서 PDF001의 일부 표(주택청약저축 개요, 체크/신용카드 비교, 은행 외국인 데스크 목록)에서 표 셀 값이 통째로 소실되는 문제를 원본 PDF와 직접 대조하여 확인.
- `youhan200203` 최신 PDF 재추출본에서 위 누락 값들이 실제로 복구되었는지 재검증 완료(p.56, p.62, p.162~168 회귀 테스트 PASS).

### 2. Chunking
- Tokenizer: `BAAI/bge-m3`
- Final chunk setting: **400 / 60** (chunk_size / overlap)
- Final chunk 수: **563개**

### 3. PDF002 Correction
- 최신 pipeline에서 PDF002가 원본 43페이지 전체를 corpus에 포함하고 있었음을 발견(원래 corpus 설계는 보이스피싱 관련 페이지만 발췌하는 것).
- 원래 corpus 설계에 맞게 **17~19페이지만** 포함하도록 page filtering을 복원.

### 4. Gold Label Validation
- 전체 evidence(167건)에 대해 gold chunk mapping을 재검증.
- Chunk 경계에 걸쳐 있는 evidence는 인접 chunk를 포함하는 multi-gold로 보정.
- 최종 상태: `PARTIAL / INVALID / CORPUS_ERROR / UNMATCHED = 0`

### 5. Final Korean Test (held-out, 400/60)
- Held-out test questions: **40**

| Metric | Value |
|---|---:|
| Dense Recall@20 | 0.9333 |
| Hybrid Recall@20 | 0.9125 |
| Hybrid + Reranker Recall@5 | 0.7958 |
| Hit@5 | 0.825 |
| MRR@5 | 0.6342 |
| nDCG@5 | 0.6334 |

자세한 검증 과정과 근거는 [`retrieval_eval/final_data_validation.md`](./retrieval_eval/final_data_validation.md)를 참고하세요.
