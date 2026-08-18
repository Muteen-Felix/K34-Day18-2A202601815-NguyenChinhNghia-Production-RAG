# Individual Reflection — Lab 18

**Tên:** Nguyễn Chính Nghĩa  
**MSSV:** 2A202601815  
**Module phụ trách:** Tất cả (M1-M5) — Bài tập cá nhân

---

## 1. Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking (nhóm câu theo similarity) | M1 | `chunk_semantic()` | Dùng all-MiniLM-L6-v2, threshold 0.85 mặc định. Với text tiếng Việt, semantic chunking tạo các chunk có ngữ nghĩa chặt chẽ hơn basic paragraph splitting |
| Hierarchical chunking (Parent-Child) | M1 | `chunk_hierarchical()` | Parent 2048 chars, child 256 chars. Retrieve trên child (precision cao) nhưng trả parent cho LLM (context đầy đủ). Đây là strategy được khuyên dùng cho production |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Parse markdown headers (h1-h3) để chunk theo logical structure. Giữ nguyên tables, code blocks — không cắt giữa chừng |
| BM25 + Dense fusion (Hybrid Search) | M2 | `reciprocal_rank_fusion()` | BM25 giỏi exact match (từ khóa), Dense giỏi semantic match → RRF merge tận dụng cả hai. RRF không cần tuning trọng số |
| Vietnamese word segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_`, phải replace về space cho BM25. Đây là gotcha quan trọng khi làm NLP tiếng Việt |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 đọc cặp (query, doc) cùng lúc → chấm điểm chính xác hơn bi-encoder. Trade-off: chậm hơn nhưng precision cao hơn đáng kể |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | 4 metrics tách 2 trục: Retrieval (context_recall, context_precision) và Generation (faithfulness, answer_relevancy). Cho phép định vị chính xác tầng gây lỗi |
| Error Tree / Diagnostic Tree | M4 | `failure_analysis()` | Map worst metric → root cause → suggested fix. Thay vì debug cảm tính, dùng metric để "vặn đúng núm" |
| Contextual embeddings (Anthropic style) | M5 | `contextual_prepend()` | Prepend context mô tả chunk nằm ở đâu trong tài liệu. Anthropic benchmark: giảm 49% retrieval failure |
| Hypothesis Q&A | M5 | `generate_hypothesis_questions()` | Sinh câu hỏi giả định → bridge vocabulary gap giữa query user và tài liệu gốc |

## 2. Khó khăn & Cách giải quyết

### Khó khăn 1: Dependency conflicts giữa langchain versions
- **Lỗi:** `ragas>=0.1.10,<0.2` yêu cầu `langchain-core<0.3`, nhưng hệ thống đã có `langchain 1.3.14` + `langchain-core 1.5.3`
- **Debug:** Kiểm tra `pip show langchain langchain-core` → xác định version conflict
- **Giải quyết:** Cài ragas không giới hạn version (`pip install ragas` thay vì `ragas>=0.1.10,<0.2`), để pip tự resolve compatible version

### Khó khăn 2: flashrank không hỗ trợ Python 3.11
- **Lỗi:** `ERROR: No matching distribution found for flashrank` — requires Python <=3.11 specific builds
- **Debug:** Đọc error message, check PyPI page
- **Giải quyết:** Bỏ flashrank (FlashrankReranker trả rỗng), chỉ dùng CrossEncoderReranker — đây là module chính, flashrank chỉ là optional alternative

### Khó khăn 3: Google Gemini API thay OpenAI
- **Vấn đề:** User dùng Google API key, không có OpenAI key. Cần adapt M4 (RAGAS) và M5 (Enrichment)
- **Giải quyết:** Tạo `_call_llm()` abstraction trong M5, dùng `langchain-google-genai.ChatGoogleGenerativeAI` cho RAGAS wrapper, thêm `_generate_answer()` helper trong pipeline

## 3. Action Plan cho project cá nhân

## Project: Production RAG cho hệ thống hỏi đáp nội bộ

### Hiện tại
- RAG pipeline hiện tại: basic paragraph chunking + dense-only search
- Known issues: Trả lời sai khi có nhiều phiên bản tài liệu (v2023 vs v2024), thiếu context cho câu hỏi multi-hop

### Plan áp dụng
1. [x] Chunking strategy: Hierarchical chunking (parent 2048, child 256) — giữ precision khi search nhưng đủ context cho LLM
2. [x] Search: Hybrid (BM25 + Dense + RRF) — BM25 bắt exact keyword match mà Dense bỏ sót
3. [x] Reranking: CrossEncoder bge-reranker-v2-m3 — top-20 → top-3, tăng precision đáng kể
4. [ ] Evaluation: RAGAS 4 metrics + Error Tree — đo lường định lượng thay vì test thủ công
5. [ ] Enrichment: Contextual prepend + HyQA — giảm retrieval failure, bridge vocabulary gap
6. [ ] Metadata filtering: Thêm version/date filter để giải quyết temporal blind spot của embedding

### Timeline
- Tuần 1: Setup RAGAS evaluation pipeline, benchmark baseline
- Tuần 2: Implement hierarchical chunking + hybrid search
- Tuần 3: Add reranking + enrichment
- Tuần 4: Metadata filtering + conflict resolution cho versioned documents

## 4. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Problem solving | 4 |
| Áp dụng kiến thức | 4 |
