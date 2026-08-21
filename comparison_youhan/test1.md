# Retrieval 및 Reranker 평가

- 평가 split: `test`
- 질문 언어: `ko`
- 후보 확보: BM25 / Dense / Hybrid의 Recall@20 중심 비교
- 최종 정렬: Hybrid와 Hybrid+Reranker의 MRR@5·nDCG@5 비교
- Recall은 evidence별 대체 gold chunk 중 하나 이상을 회수한 비율

## 5. 결과

### 5.1 Chunk 400/60

Validation **40개 전부**에서 gold chunk를 확보했습니다 (chunk 606개).

**Rerank 전 후보 확보 성능**

| Method | Recall@20 | Hit@20 | MRR@20 | Recall@50 | N |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.749 | 0.775 | 0.417 | 0.829 | 40 |
| Dense Retrieval (BGE-M3) | **0.933** | **0.950** | **0.602** | **0.975** | 40 |
| Hybrid Retrieval (RRF) | 0.912 | 0.925 | 0.560 | **0.975** | 40 |

**Rerank 후 최종 정렬 성능**

| Method | Recall@5 | Hit@5 | MRR@5 | nDCG@5 | N |
|---|---:|---:|---:|---:|---:|
| Hybrid Retrieval (RRF) | 0.699 | 0.750 | 0.544 | 0.549 | 40 |
| Hybrid + Reranker (bge-reranker-v2-m3) | **0.796** | **0.825** | **0.628** | **0.628** | 40 |
