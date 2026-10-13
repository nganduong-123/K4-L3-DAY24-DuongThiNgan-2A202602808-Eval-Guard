# Nhóm thực hiện K4 Track 3B — Day 24

- **Hình thức:** Dự án cá nhân
- **Thành viên duy nhất:** Dương Thị Ngân
- **Mã sinh viên:** 2A202602808
- **Repository:** https://github.com/nganduong-123/K4-L3-DAY24-DuongThiNgan-2A202602808-Eval-Guard
- **Người thực hiện và nộp bài:** Dương Thị Ngân — 2A202602808
- **Hạn nộp:** 11:59 ngày 13/10/2026 (Asia/Ho_Chi_Minh, UTC+07:00)
- **Commit nộp ban đầu:** `c1aa4f0`
- **Phiên bản dùng để chấm:** commit mới nhất trên nhánh `main`

## Thành viên và phân công

| STT | Họ và tên | Mã sinh viên | Phân công |
| --- | --- | --- | --- |
| 1 | Dương Thị Ngân | 2A202602808 | Thực hiện toàn bộ Phase A (RAGAS), Phase B (LLM-as-a-Judge), Phase C (Guardrails), kiểm thử, phân tích kết quả, viết báo cáo và nộp bài |

Người tích hợp mã nguồn, chạy batch đánh giá, kiểm tra artifact và nộp liên kết VLearn: **Dương Thị Ngân — 2A202602808**.

## Môi trường và dữ liệu thực nghiệm

- Hệ điều hành: Windows 10.
- Python: 3.11.9, môi trường ảo `.venv`.
- Vector database: Qdrant 1.9.7.
- Corpus sau xử lý: 102 chunks.
- Tập đánh giá: 50 câu hỏi; toàn bộ câu trả lời được sinh bằng mô hình `openai/gpt-oss-20b`.
- RAGAS: sử dụng checkpoint Gemini Flash Lite, tiếp tục bằng các evaluator Groq (GPT-OSS/Qwen) và sentence-transformer local; các lỗi quota/parse được ghi nhận trung thực trong báo cáo.
- Hai PDF dạng ảnh không có text layer được ghi nhận là giới hạn dữ liệu đầu vào.

## INDIVIDUAL

### DuongThiNgan-2A202602808

- **Phần việc được giao và đã thực hiện:** toàn bộ mã nguồn, cấu hình, kiểm thử, thí nghiệm và tài liệu của ba phase.
- **Minh chứng:** mã nguồn trong `src/`, cấu hình trong `guardrails/`, kết quả trong `artifacts/` và báo cáo trong `doc/reports/`.
- **Kiểm thử:** unit test và checker của bài lab; các thí nghiệm live được chạy để tạo artifact, không tự điền số liệu giả.
- **Nội dung đã thực hành:** xây dựng bộ dữ liệu đánh giá RAG, đánh giá bằng RAGAS, pairwise/binary LLM judge, kiểm tra position bias, phát hiện PII/prompt injection và đo latency guardrail.
- **Giới hạn:** quota/rate limit của evaluator bên ngoài và các PDF scan không có text layer; pipeline có retry, rate limiter và checkpoint để tránh mất tiến độ.
- **Hỗ trợ AI:** OpenAI Codex được sử dụng để rà soát starter code, hỗ trợ triển khai, gỡ lỗi, chạy kiểm thử/thí nghiệm và hoàn thiện báo cáo. Kết quả được đối chiếu bằng test, checker và artifact thực tế.

## Cập nhật sau khi nộp liên kết

Liên kết repository đã được nộp lên VLearn lúc **22:47:40 ngày 12/10/2026**. Các commit hoàn thiện tiếp theo được đẩy lên cùng repository và nhánh `main` trước hạn nộp.
