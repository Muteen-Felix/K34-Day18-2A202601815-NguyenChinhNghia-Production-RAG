# Failure Analysis — Lab 18: Production RAG

**Tên:** Nguyễn Chính Nghĩa  
**MSSV:** 2A202601815 — Bài tập cá nhân

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:-------------:|:----------:|:---:|
| **Faithfulness** | 0.9840 | 0.9875 | +0.0035 |
| **Answer Relevancy** | 0.6583 | 0.6804 | +0.0221 |
| **Context Precision** | 0.5472 | 0.5417 | -0.0055 |
| **Context Recall** | 0.8631 | 0.7760 | -0.0871 |

> **Nhận xét tổng quan:** Production RAG Pipeline đạt độ trung thực (*Faithfulness*) xuất sắc (**0.9875**) và độ liên quan câu trả lời (*Answer Relevancy*) tăng rõ rệt (+0.0221) nhờ kết hợp Hybrid Search (BM25 + Dense Qdrant + RRF) và CrossEncoder Reranker (`bge-reranker-v2-m3`). Các trường hợp Context Precision/Recall thấp chủ yếu đến từ các câu hỏi đa phiên bản tài liệu (conflict versioning) và câu hỏi tính toán/multi-hop.

---

## Bottom-5 Failures (Phân tích lỗi theo Diagnostic Tree)

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** `phòng phê duyệt. Từ 5.000.000 VNĐ trở lên: cần thêm phê duyệt Kế toán trưởng.`
- **Worst metric:** `context_precision` (0.2778) — Avg: 0.5888
- **Error Tree:** Output sai ngưỡng → Context đúng? → Context retrieve nhiều chunk về các hạn mức mua sắm dưới 50 triệu (Trưởng phòng, Kế toán trưởng) thay vì trúng chính xác mốc trên 50 triệu (CEO) → Noise trong top-k context.
- **Root cause:** Mức độ chi tiết theo khoảng số liệu (Numerical Thresholds) — Vector search không hiểu cấu trúc phân tầng số học `>= 50M`, chỉ matching từ khóa "mua thiết bị", "phê duyệt".
- **Suggested fix:** Cải thiện chunking có table structure preservation hoặc metadata filter phân loại hạn mức ngân sách (`budget_tier: high`).

### #2
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** `Senior (P3-P4) | 20.000.000 - 35.000.000 | Lead (P5) | 35.000.000 - 50.000.000 | Manager (M1-M2) | 45.000.000 - 70.000.000 | Lương thử việc: 85%...`
- **Worst metric:** `answer_relevancy` (0.4945) — Avg: 0.6871
- **Error Tree:** Output trích xuất toàn bộ bảng lương → LLM answer có đúng? → Context chứa bảng lương nhiều cấp bậc, LLM liệt kê cấp bậc Senior/Lead/Manager thay vì tập trung riêng vào Junior.
- **Root cause:** Multi-step numerical reasoning — Cần bước 1: lấy max lương Junior (20M), bước 2: nhân 85%. LLM trả về raw context bảng lương thiếu bước suy luận cuối.
- **Suggested fix:** Áp dụng Chain-of-Thought (CoT) Prompting và Instruction Tuning ("Hãy thực hiện tính toán số học rõ ràng từng bước trước khi kết luận").

### #3
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** `khích sử dụng passphrase dài hơn 16 ký tự. ## Xác thực đa yếu tố (MFA) Tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ...`
- **Worst metric:** `context_precision` (0.3333) — Avg: 0.6886
- **Error Tree:** Output đúng nội dung cốt lõi nhưng bị lẫn thông tin passphrase của chunk bên cạnh → Context precision giảm do top-3 retrieve chunk có phần passphrase không liên quan trực tiếp.
- **Root cause:** Chunk boundary noise — Chunk chứa cả phần cuối của mục mật khẩu và phần đầu của mục MFA.
- **Suggested fix:** Structure-aware chunking theo Markdown H2 headers (`## Xác thực đa yếu tố`) để phân tách hoàn toàn mục MFA ra khỏi mục độ dài mật khẩu.

### #4
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** `# Chính sách mật khẩu (Phiên bản hiện hành) > Phiên bản: 2.0 | Ngày hiệu lực: 01/07/2024 | Phòng ban: CNTT ## Yêu cầu mật khẩu Mật khẩu phải có tối thiểu 12 ký tự...`
- **Worst metric:** `context_precision` (0.5000) — Avg: 0.6918
- **Error Tree:** Context retrieve cả bản v1.0 (8 ký tự) và v2.0 (12 ký tự) → Context chứa thông tin mâu thuẫn → Precision giảm vì 1 trong 2 chunk là lỗi thời.
- **Root cause:** Temporal Blind Spot (Xung đột phiên bản) — Cả hai tài liệu đều có embedding tương đồng cao về chủ đề mật khẩu.
- **Suggested fix:** Metadata filtering theo `status: active` / `version: max`, hoặc Ingestion tagging loại bỏ/đánh dấu văn bản đã hết hiệu lực.

### #5
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** `thay thế hoàn toàn phiên bản 1.0 ban hành ngày 01/01/2023. ## Thâm niên công tác Nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm làm việc liên tục. Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3)...`
- **Worst metric:** `context_recall` (0.6087) — Avg: 0.7090
- **Error Tree:** Multi-hop question → Context chỉ retrieve được chunk chính sách nghỉ phép và thâm niên, thiếu chunk về dải lương Senior (nằm ở tài liệu quy chế lương riêng biệt) → Context Recall thấp.
- **Root cause:** Multi-Hop Retrieval Blind Spot — Câu hỏi ghép 2 thực thể từ 2 tài liệu độc lập ("ngày phép" + "dải lương"). Top-3 retrieval đơn truy vấn chỉ ưu tiên ngữ nghĩa của vế đầu.
- **Suggested fix:** Query Decomposition (Pre-RAG) — Tách câu hỏi thành 2 sub-queries: `Sub-Q1: "Senior 9 năm thâm niên được bao nhiêu ngày phép?"` và `Sub-Q2: "Dải lương của nhân viên cấp Senior là bao nhiêu?"`, sau đó hợp nhất kết quả tìm kiếm.

---

## Case Study (Phân tích chuyên sâu)

**Question chọn phân tích:** *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*

### Error Tree Walkthrough:
```
[User Question: "Nhân viên được nghỉ bao nhiêu ngày phép năm?"]
        │
        ▼
[Retrieve Chunks] ──► Retrieve đồng thời cả 'chinh_sach_nghi_phep_v2023.md' (12 ngày)
        │             và 'chinh_sach_nghi_phep_v2024.md' (15 ngày)
        ▼
[Context Assessment] ──► Context chứa 2 thông tin mâu thuẫn trực tiếp (12 vs 15)
        │
        ▼
[LLM Generation] ──► LLM dễ nhầm lẫn hoặc trích xuất phiên bản cũ nếu không có chỉ dẫn thời gian
        │
        ▼
[Root Cause] ──► Temporal Blind Spot trong Vector Search (không phân biệt thời gian hiệu lực)
```

### Biện pháp xử lý:
1. **Tại Ingestion:** Bổ sung metadata `version`, `effective_date`, `is_active` cho từng chunk trong M1 & M5.
2. **Tại Retrieval:** Áp dụng Qdrant payload filter `filter={"is_active": True}` để loại bỏ hoàn toàn tài liệu hết hiệu lực.
3. **Tại Augmentation:** Thêm NLI Conflict Resolver để phát hiện mâu thuẫn trong top-k chunks trước khi đưa vào context prompt.

---

## Kế hoạch tối ưu tiếp theo (Next Iterations)
1. **Metadata Filtering:** Tích hợp bộ lọc Qdrant theo phòng ban và phiên bản tài liệu.
2. **Query Decomposition:** Thêm module tiền xử lý phân rã câu hỏi multi-hop phức tạp.
3. **HyDE (Hypothetical Document Embeddings):** Sinh câu trả lời giả định trước khi vector search để cải thiện Recall cho các câu hỏi suy luận.
