# ARCHITECTURE.md — Lab 18: Production RAG Pipeline

## Cấu trúc thư mục

```
lab18-production-rag/
├── src/
│   ├── m1_chunking.py      # 3 chunking strategies + load_documents
│   ├── m2_search.py        # BM25 + Dense (Qdrant) + RRF hybrid
│   ├── m3_rerank.py        # CrossEncoder reranking (bge-reranker-v2-m3)
│   ├── m4_eval.py          # RAGAS 4 metrics + failure analysis
│   ├── m5_enrichment.py    # Chunk enrichment (summarize, HyQA, contextual, metadata)
│   └── pipeline.py         # Ghép M1→M5→M2→M3→LLM→M4
├── config.py               # Shared constants (unchanged)
├── test_set.json           # 20 Q&A evaluation pairs
├── data/                   # 25 .md + 3 .pdf documents
├── tests/                  # 5 test files (source of truth)
├── analysis/               # Deliverables
│   ├── failure_analysis.md
│   └── reflections/reflection_NguyenChinhNghia.md
└── reports/
    ├── ragas_report.json
    └── naive_baseline_report.json
```

## Luồng dữ liệu

```
load_documents(data/) → list[dict{text, metadata}]
        ↓
M1: chunk_hierarchical() → (parents[], children[])
        ↓ (children converted to dict)
M5: enrich_chunks() → list[EnrichedChunk]
        ↓ (enriched_text + auto_metadata → dict)
M2: HybridSearch.index(chunks)
    ├── BM25Search.index()  → in-memory BM25Okapi
    └── DenseSearch.index() → Qdrant collection (bge-m3 embeddings)
        ↓
M2: HybridSearch.search(query)
    ├── BM25Search.search()  → top-20 BM25 results
    ├── DenseSearch.search() → top-20 dense results
    └── reciprocal_rank_fusion([bm25, dense]) → top-20 hybrid
        ↓
M3: CrossEncoderReranker.rerank(query, top-20) → top-3
        ↓
LLM: Gemini generate answer from top-3 contexts
        ↓
M4: evaluate_ragas(questions, answers, contexts, ground_truths)
    → 4 metric scores + per_question breakdown
    → failure_analysis(bottom-5)
    → save_report("ragas_report.json")
```

## Quyết định thiết kế

### 1. Google Gemini thay OpenAI
- User dùng Google API key, nên tất cả LLM calls sẽ dùng `google-generativeai` (M5 enrichment, pipeline LLM generation)
- RAGAS evaluation sẽ wrap Gemini bằng `langchain-google-genai` (ChatGoogleGenerativeAI) để tương thích RAGAS API
- Fallback: nếu không có API key, dùng extractive methods hoặc return defaults

### 2. M1: Semantic chunking model
- Dùng `all-MiniLM-L6-v2` (384 dims) cho semantic chunking — nhẹ, nhanh, đủ cho similarity grouping
- `BAAI/bge-m3` (1024 dims) dùng riêng cho dense search (M2) — mạnh hơn nhưng nặng hơn

### 3. M2: Vietnamese segmentation
- `underthesea.word_tokenize` nối từ ghép bằng `_` → phải replace `_` → ` ` để BM25 tokenize đúng
- DenseSearch dùng `qdrant_client.QdrantClient` → `query_points()` (qdrant-client >= 2.0)

### 4. M4: RAGAS version handling
- ragas >= 0.1.10 < 0.2 (theo requirements.txt) — API cũ dùng `evaluate()` + `Dataset.from_dict()`
- Cần check xem RAGAS version đang dùng support Google Gemini LLM wrapper không

### 5. Pipeline LLM
- Pipeline `run_query()` dùng Gemini (`gemini-2.0-flash`) thay `gpt-4o-mini`
- Cùng system prompt "Trả lời CHỈ dựa trên context"

## Giả định

1. **Docker Qdrant sẽ chạy trước khi test M2 Dense** — BM25 tests không cần Docker, chỉ Dense cần
2. **Google API key tương thích RAGAS** — RAGAS cần LLM wrapper, sẽ dùng langchain-google-genai
3. **Python 3.11+** — đã xác nhận 3.11.9
4. **bge-m3 + bge-reranker-v2-m3 models** — sẽ auto-download lần đầu, có thể mất vài phút
5. **Tên sinh viên**: Nguyễn Chính Nghĩa (từ repo name K34-Day18-2A202601815-NguyenChinhNghia)
6. **Bài tập cá nhân** — README nói "cá nhân", reflection file đặt tên reflection_NguyenChinhNghia.md
