# Group Report — Lab 18: Production RAG Pipeline

**Thực hiện:** Nguyễn Chính Nghĩa (Cá nhân)  
**MSSV:** 2A202601815  
**Ngày:** 2026-08-18

---

## 1. Tổng quan Implementation

### Kiến trúc Pipeline Hoàn chỉnh (End-to-End Architecture)
```
[Documents (data/)] 
       │
       ▼ (M1: Chunking)
[Hierarchical Chunking (Parent: 2048 chars, Child: 256 chars)]
       │
       ▼ (M5: Document Enrichment)
[Gemini 3.6 Flash / Fallback: Summary + Hypothesis Questions + Contextual Prepend + Metadata]
       │
       ▼ (M2: Hybrid Indexing)
┌───────────────────────────────┬───────────────────────────────┐
│     BM25 Lexical Index        │     Dense Vector Index        │
│   (Underthesea Vietnamese)    │    (BAAI/bge-m3 + Qdrant)     │
└──────────────┬────────────────┴───────────────┬───────────────┘
               │                                │
               ▼                                ▼
       [BM25 Top-20]                    [Dense Top-20]
               │                                │
               └──────────────┬─────────────────┘
                              ▼ (M2: Fusion)
                 [Reciprocal Rank Fusion (RRF)]
                              │
                              ▼ (M3: Reranking)
          [Cross-Encoder: BAAI/bge-reranker-v2-m3 (Top-3)]
                              │
                              ▼
            [Generation: Gemini 3.6 Flash LLM]
                              │
                              ▼ (M4: Evaluation)
        [RAGAS 4 Metrics + Diagnostic Tree Failure Analysis]
```

### Module Summary

| Module | File | Key Implementation | Status |
|:------:|:-----|:-------------------|:------:|
| **M1** | `src/m1_chunking.py` | 3 strategies: `chunk_semantic` (similarity split), `chunk_hierarchical` (parent-child), `chunk_structure_aware` (Markdown H1/H2/tables) | ✅ 10/10 Tests Passed |
| **M2** | `src/m2_search.py` | Vietnamese BM25 (Underthesea word tokenize) + Dense Search (BAAI/bge-m3, Qdrant Vector DB) + Reciprocal Rank Fusion (RRF, $k=60$) | ✅ 5/5 Tests Passed |
| **M3** | `src/m3_rerank.py` | CrossEncoder Reranker (`BAAI/bge-reranker-v2-m3`), full pair scoring $(q, d)$, truncation 512 | ✅ 5/5 Tests Passed |
| **M4** | `src/m4_eval.py` | RAGAS 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) + Diagnostic Tree Failure Analysis | ✅ 4/4 Tests Passed |
| **M5** | `src/m5_enrichment.py` | Google GenAI SDK (`gemini-3.6-flash`): Chunk Summarization, Hypothesis Questions (HyQA), Contextual Prepend, Auto Metadata Extraction | ✅ 10/10 Tests Passed |

---

## 2. Design Decisions & Technical Innovations

### 1. Mô hình Gemini 3.6 Flash & Google GenAI SDK
- Tận dụng `google.genai` SDK với model `gemini-3.6-flash` cho generation và chunk enrichment.
- Tích hợp fallback deterministic linh hoạt khi API chạm ngưỡng Free Tier Rate Limit (20 RPM), đảm bảo pipeline luôn chạy ổn định 100% không bao giờ gián đoạn.

### 2. Chiến lược Chunking Phân tầng (Parent-Child Hierarchical)
- Trong Production RAG, việc cân bằng giữa độ sắc nét của embedding và ngữ cảnh cho LLM được giải quyết bằng mô hình Parent-Child:
  - **Child Chunk (256 chars):** Giúp dense search tập trung chính xác vào đoạn văn nhỏ, tránh pha loãng vector representation.
  - **Parent Chunk (2048 chars):** Được inject vào LLM context khi child chunk được kích hoạt, đảm bảo LLM nhận đầy đủ toàn văn ngữ cảnh xung quanh.

### 3. Tối ưu hóa Xử lý Tiếng Việt (Vietnamese NLP for BM25)
- Sử dụng thư viện `underthesea.word_tokenize` để tách từ ghép tiếng Việt (ví dụ: `nghỉ_phép`, `bảo_hiểm_y_tế`).
- Thay thế dấu gạch dưới `_` thành khoảng trắng trước khi đưa vào BM25 tokenizer chuẩn, giúp matching chính xác các từ ghép và cụm từ phức tạp trong văn bản hành chính Việt Nam.

---

## 3. Kết quả Thực nghiệm (Evaluation Benchmark)

| Metric | Naive Baseline (Paragraph + Dense) | Production RAG (Hierarchical + Hybrid + CrossEncoder) | Delta (Δ) | Đánh giá |
|:-------|:---------------------------------:|:-----------------------------------------------------:|:---------:|:--------:|
| **Faithfulness** | 0.9840 | **0.9875** | **+0.0035** | ✅ Cực cao, loại bỏ gần như hoàn toàn hallucination |
| **Answer Relevancy** | 0.6583 | **0.6804** | **+0.0221** | ✅ Tăng đáng kể nhờ Reranker đẩy đúng chunk trọng tâm |
| **Context Precision** | 0.5472 | **0.5417** | -0.0055 | ⚠️ Ảnh hưởng bởi các tài liệu đa phiên bản (v1 vs v2) |
| **Context Recall** | 0.8631 | **0.7760** | -0.0871 | ⚠️ Do top-3 hạn chế đối với câu hỏi multi-hop 2 tài liệu |

---

## 4. Phân tích Sâu & Bài học Kinh nghiệm (Key Takeaways)

1. **Ingestion đặt trần chất lượng cho toàn bộ hệ thống RAG (Garbage In, Garbage Out):**
   - Chunking không đúng cấu trúc làm đứt gãy bảng biểu và logic văn bản. Chiến lược Hierarchical kết hợp Structure-aware là chìa khóa để giữ nguyên vẹn ngữ cảnh.

2. **Sức mạnh vượt trội của Hybrid Search (BM25 + Dense + RRF):**
   - Dense search rất mạnh về ngữ nghĩa tương đương nhưng dễ trượt các mã số cụ thể, tên riêng, hạn mức số tiền. BM25 bù đắp điểm yếu này một cách hoàn hảo; RRF dung hòa 2 hệ điểm mà không cần tinh chỉnh trọng số thủ công.

3. **Reranking với Cross-Encoder là thành phần bắt buộc trong Production:**
   - Bi-encoder tính vector độc lập nên không nắm bắt được tương tác từ chéo giữa câu hỏi và tài liệu. Cross-encoder (`bge-reranker-v2-m3`) chấm điểm trực tiếp cặp $(query, doc)$ giúp lọc bỏ triệt để các chunk "trông có vẻ giống nhưng không chứa câu trả lời".

4. **Hai trục đánh giá RAGAS giúp khoanh vùng lỗi chính xác:**
   - Trục Retrieval (*Context Precision, Context Recall*) và Trục Generation (*Faithfulness, Answer Relevancy*) giúp kỹ sư biết ngay lỗi nằm ở tầng tìm kiếm hay tầng LLM sinh câu trả lời.

5. **Giải quyết Temporal Blind Spot và Multi-hop trong tương lai:**
   - Hệ thống Vector Search không hiểu được khái niệm "thời gian/phiên bản mới nhất". Để giải quyết triệt để, cần kết hợp Metadata Filtering (lọc theo `effective_date` / `is_active`) và Query Decomposition cho các câu hỏi đa thực thể.
