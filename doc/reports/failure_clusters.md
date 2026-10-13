# Phân tích lỗi Phase A

- **Hình thức:** cá nhân — Dương Thị Ngân, 2A202602808
- **Nguồn:** `artifacts/ragas_50q.json`
- **Run timestamp:** 2026-10-13T04:20:17Z
- **Generation:** 50/50 câu dùng `openai/gpt-oss-20b`, không dùng fallback câu trả lời.
- **Evaluators:** `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` và local `sentence-transformers/all-MiniLM-L6-v2`.

Do quota/rate limit của evaluator trong lượt cuối, RAGAS trả `NaN` cho một số metric; pipeline ghi các trường hợp không đo được là 0 thay vì tự tạo điểm. Vì vậy bảng dưới là bằng chứng chạy thật nhưng các mean, đặc biệt ở multi-hop/adversarial, là lower bound và không nên được diễn giải như chất lượng tuyệt đối của câu trả lời.

## Điểm theo distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
| --- | ---: | ---: | ---: |
| faithfulness | 0.9250 | 0.1208 | 0.0000 |
| answer_relevancy | 0.4860 | 0.1798 | 0.1911 |
| context_precision | 0.8083 | 0.3167 | 0.0000 |
| context_recall | 0.6750 | 0.4042 | 0.5083 |
| avg_score | 0.7236 | 0.2554 | 0.1749 |

## Bottom 10 và nguyên nhân

| Rank | ID | Distribution | Câu hỏi rút gọn | Avg | Worst metric | Diagnosis | Suggested fix |
| ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | 25 | multi_hop | Lương thử việc Junior cao nhất | 0.0000 | faithfulness | LLM hallucinating/evaluator unavailable | Siết prompt và chạy lại evaluator ổn định |
| 2 | 33 | multi_hop | Phụ cấp và phép của Manager 12 năm | 0.0000 | faithfulness | LLM hallucinating/evaluator unavailable | Tăng recall đa tài liệu và chạy lại |
| 3 | 40 | multi_hop | Manager thử việc gặp vi phạm bảo mật | 0.0000 | faithfulness | LLM hallucinating/evaluator unavailable | Ghép đúng policy thử việc và bảo mật |
| 4 | 37 | multi_hop | Tự xóa malware rồi chia sẻ trên Slack | 0.0082 | faithfulness | LLM hallucinating/evaluator unavailable | Truy hồi đủ các điều khoản sự cố |
| 5 | 43 | adversarial | Độ dài tối thiểu của mật khẩu | 0.0169 | faithfulness | Xung đột phiên bản/evaluator unavailable | Ưu tiên metadata phiên bản hiện hành |
| 6 | 49 | adversarial | So sánh phép v2023 và bản hiện hành | 0.0264 | faithfulness | Xung đột phiên bản/evaluator unavailable | Rerank theo effective date/status |
| 7 | 35 | multi_hop | Lương và phụ cấp Junior P1 thử việc | 0.0625 | faithfulness | LLM hallucinating/evaluator unavailable | Kết hợp bảng lương và thử việc |
| 8 | 47 | adversarial | Có nên tự xử lý malware không | 0.0696 | faithfulness | Phủ định/evaluator unavailable | Giữ nguyên từ khóa phủ định trong prompt |
| 9 | 39 | multi_hop | So sánh password policy v1/v2 | 0.0721 | faithfulness | Xung đột phiên bản/evaluator unavailable | Lọc version và nêu trạng thái thay thế |
| 10 | 22 | multi_hop | Laptop 30 triệu cần ai phê duyệt | 0.0772 | faithfulness | LLM hallucinating/evaluator unavailable | Tăng recall và rerank tài liệu mua sắm |

Đối chiếu nội dung cho thấy hai loại vấn đề khác nhau. ID 25 và 33 trả “Không tìm thấy” dù context chứa công thức 85% lương Junior và quy tắc phép/thâm niên; đây là lỗi generation không tổng hợp nhiều đoạn. Ngược lại ID 43 trả đúng “12 ký tự”, ID 47 giữ đúng phủ định và quy trình báo trong một giờ, ID 49 phân biệt đúng 12 ngày của v2023 với 15 ngày của v2024. Điểm rất thấp của các câu đúng này chủ yếu phản ánh evaluator bị quota/parse failure, nên cần rerun trên cùng artifact khi quota được khôi phục trước khi kết luận pipeline hallucinate.

## Matrix worst metric và nhóm yếu nhất

| Worst metric | factual | multi_hop | adversarial | Tổng |
| --- | ---: | ---: | ---: | ---: |
| faithfulness | 1 | 17 | 10 | 28 |
| answer_relevancy | 11 | 2 | 0 | 13 |
| context_precision | 5 | 1 | 0 | 6 |
| context_recall | 3 | 0 | 0 | 3 |

Nhóm yếu theo mean là **adversarial = 0.1749**. Worst metric xuất hiện nhiều nhất là **faithfulness, 28/50 câu**. Không dùng quy mô nhóm 20/20/10 để chọn nhóm yếu; kết luận dựa trên mean. Với adversarial, version conflict và câu hỏi phủ định làm judge cần phân biệt tài liệu đã thay thế, đồng thời quota evaluator làm nhiều faithfulness score bằng 0. Cải tiến có thể kiểm chứng là thêm metadata `effective_date/status`, rerank bản hiện hành, buộc generator trích nguồn/phiên bản và chạy lại đúng 50 câu với một evaluator pool ổn định.

## Bẫy adversarial và giới hạn

Factual đạt mean 0.7236, cao hơn rõ rệt multi-hop và adversarial. Tuy nhiên điểm thấp không tự chứng minh hệ thống đã nhận ra bẫy: ID 43, 47 và 49 cho thấy answer thực tế đã xử lý đúng version/negation nhưng evaluator không luôn ghi nhận được. Bộ test chỉ có 50 câu và 10 adversarial, sử dụng nhiều evaluator do giới hạn quota; so sánh giữa distribution có nhiễu từ model và lỗi parse. Lần chạy production nên cố định một evaluator/version, lưu error count riêng thay vì trộn với điểm 0, và mở rộng mẫu adversarial trước khi đặt quality gate.
