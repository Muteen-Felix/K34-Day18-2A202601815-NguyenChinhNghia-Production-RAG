# PLAN.md — Lab 18: Production RAG Pipeline

## Tóm tắt đề bài
Implement 5 module của Production RAG Pipeline (M1-M5): Advanced Chunking → Enrichment → Hybrid Search (BM25+Dense+RRF) → Cross-encoder Reranking → RAGAS Evaluation. So sánh kết quả với naive baseline (paragraph chunking + dense-only). Viết failure analysis cho bottom-5 worst questions và reflection cá nhân.

## Danh sách công việc (theo thứ tự)

### BẮT BUỘC

- [x] **T0: Setup** — Cài dependencies, tạo .env, chạy Docker Qdrant
  - Tiêu chí xong: `pip install` thành công, Qdrant accessible trên port 6333

- [ ] **T1: M1 Chunking** — Implement 3 strategies trong `src/m1_chunking.py`
  - `chunk_semantic()`: split bằng sentence similarity (SentenceTransformer all-MiniLM-L6-v2)
  - `chunk_hierarchical()`: parent-child hierarchy (parent 2048 chars, child 256 chars)
  - `chunk_structure_aware()`: parse markdown headers → chunk theo section
  - File: `src/m1_chunking.py`
  - Tiêu chí xong: `pytest tests/test_m1.py` — 100% pass (10 tests)

- [ ] **T2: M2 Search** — Implement BM25 + Dense + RRF trong `src/m2_search.py`
  - `segment_vietnamese()`: underthesea word_tokenize + replace `_` → ` `
  - `BM25Search.index()` + `.search()`: BM25Okapi trên segmented text
  - `DenseSearch.index()` + `.search()`: bge-m3 + Qdrant query_points
  - `reciprocal_rank_fusion()`: merge ranked lists score(d) = Σ 1/(k+rank+1)
  - File: `src/m2_search.py`
  - Tiêu chí xong: `pytest tests/test_m2.py` — 100% pass (4 tests)

- [ ] **T3: M3 Rerank** — Implement cross-encoder reranking trong `src/m3_rerank.py`
  - `CrossEncoderReranker._load_model()`: load bge-reranker-v2-m3
  - `CrossEncoderReranker.rerank()`: predict scores, sort, return top-k
  - File: `src/m3_rerank.py`
  - Tiêu chí xong: `pytest tests/test_m3.py` — 100% pass (5 tests)

- [ ] **T4: M4 Eval** — Implement RAGAS evaluation trong `src/m4_eval.py`
  - `evaluate_ragas()`: 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  - `failure_analysis()`: diagnostic tree, bottom-N worst questions
  - File: `src/m4_eval.py`
  - Tiêu chí xong: `pytest tests/test_m4.py` — 100% pass (4 tests)
  - LƯU Ý: Dùng Google Gemini API thay vì OpenAI cho RAGAS evaluation

- [ ] **T5: M5 Enrichment** — Implement enrichment pipeline trong `src/m5_enrichment.py`
  - `summarize_chunk()`: tóm tắt chunk (Gemini hoặc extractive fallback)
  - `generate_hypothesis_questions()`: sinh câu hỏi giả định
  - `contextual_prepend()`: prepend context mô tả chunk
  - `extract_metadata()`: extract topic, entities, category
  - `_enrich_single_call()`: combined 1 API call/chunk
  - File: `src/m5_enrichment.py`
  - Tiêu chí xong: `pytest tests/test_m5.py` — 100% pass (9 tests)

- [ ] **T6: Pipeline** — Đảm bảo pipeline chạy end-to-end
  - File: `src/pipeline.py` (cần sửa nhỏ nếu dùng Gemini thay OpenAI)
  - Tiêu chí xong: `python src/pipeline.py` exit code 0, sinh `ragas_report.json`

- [ ] **T7: Failure Analysis** — Điền `analysis/failure_analysis.md`
  - Bottom-5 worst questions với diagnosis + fix + Error Tree
  - Tiêu chí xong: File có đủ 5 failures với diagnosis, root cause, suggested fix

- [ ] **T8: Reflection** — Viết `analysis/reflections/reflection_NguyenChinhNghia.md`
  - Phần 1: Mapping bài giảng → code (5 modules)
  - Phần 2: Khó khăn & giải quyết
  - Phần 3: Action plan cho project cá nhân
  - Tiêu chí xong: File có đủ 3 phần theo template ASSIGNMENT.md

### MỞ RỘNG (không làm trước phần bắt buộc)
- [ ] Bonus: RAGAS Faithfulness ≥ 0.85
- [ ] Bonus: Tất cả metrics ≥ 0.75
- [ ] Bonus: Enrichment combined mode (`_enrich_single_call()`)
- [ ] Bonus: Latency breakdown report

## Quyết định trong lúc làm

- **Google Gemini thay OpenAI**: User dùng Google API key. Sẽ dùng `google-generativeai` cho M5 enrichment và `langchain-google-genai` cho RAGAS LLM wrapper. Pipeline LLM generation cũng sẽ dùng Gemini.
- **DenseSearch model**: Giữ nguyên `BAAI/bge-m3` (EMBEDDING_DIM=1024) theo config.
- **Fallback cho không có API key**: Tất cả M5 functions có extractive fallback, M4 có zeros fallback.
