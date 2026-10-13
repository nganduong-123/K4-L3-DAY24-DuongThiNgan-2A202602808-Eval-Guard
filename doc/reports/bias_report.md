# Báo cáo LLM-as-a-Judge và bias

- **Hình thức:** cá nhân — Dương Thị Ngân, 2A202602808
- **Nguồn:** `artifacts/judge_results.json`
- **Thời điểm chạy:** 2026-10-12T16:18:59Z
- **Mô hình sinh câu trả lời:** `openai/gpt-oss-20b`
- **Mô hình judge:** `openai/gpt-oss-120b`

## Pairwise, swap-and-average

Năm cặp được chấm hai lần. Ở lượt hai, vị trí A/B được đảo và kết quả dưới đây đã được quy đổi lại về đáp án gốc.

| Question ID | Độ dài A | Độ dài B | Winner pass 1 | Winner pass 2 | Kết quả cuối | Nhất quán vị trí |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 | 60 | 75 | B | B | B | Có |
| 5 | 15 | 43 | tie | tie | tie | Có |
| 12 | 138 | 68 | A | tie | tie | Không |
| 21 | 268 | 104 | B | B | B | Có |
| 41 | 103 | 48 | A | A | A | Có |

Position-bias count là **1/5**, tương đương **0.20**. Trường hợp không nhất quán là câu 12; cơ chế swap-and-average đưa trường hợp này về `tie`, nhờ vậy không giữ một lựa chọn phụ thuộc vị trí. Trong ba cặp có winner dứt khoát, đáp án dài hơn thắng hai lần, nên verbosity win rate là **2/3 = 0.667**. Mẫu chỉ có năm cặp nên đây là tín hiệu cần theo dõi, chưa đủ để kết luận judge luôn thiên vị câu dài.

## Hiệu chuẩn nhị phân với nhãn người

Judge chỉ nhận `question`, `model_answer` và `reference`; nhãn và ghi chú của người không được đưa vào prompt.

| Question ID | Nhãn judge | Nhãn người | Khớp |
| --- | ---: | ---: | --- |
| 1 | 1 | 1 | Có |
| 5 | 0 | 0 | Có |
| 12 | 1 | 1 | Có |
| 21 | 1 | 1 | Có |
| 23 | 1 | 1 | Có |
| 29 | 0 | 0 | Có |
| 33 | 1 | 1 | Có |
| 41 | 0 | 0 | Có |
| 46 | 1 | 1 | Có |
| 50 | 0 | 0 | Có |

Observed agreement là **10/10 = 1.00**. Cả hai bộ nhãn có 6 nhãn dương và 4 nhãn âm, nên chance agreement là `0.6 × 0.6 + 0.4 × 0.4 = 0.52`. Cohen’s κ là `(1.00 - 0.52) / (1 - 0.52) = 1.00`, thuộc mức almost perfect. Kết quả tốt nhưng tập hiệu chuẩn chỉ có 10 câu; khi triển khai nên mở rộng nhãn người, theo dõi theo từng loại lỗi và định kỳ đánh giá lại sau khi đổi prompt/model.

## Kết luận và biện pháp giảm bias

Luôn giữ hai lượt swap-and-average, lưu cả reasoning và điểm của từng lượt, đồng thời coi kết quả `tie` là cần review thay vì ép winner. Với verbosity bias, prompt phải tách accuracy, completeness và conciseness; báo cáo nên theo dõi riêng tỷ lệ winner dài hơn trên tập lớn hơn. Cohen’s κ cần được đo lại trên mẫu ẩn, có nhiều người gán nhãn, thay vì dùng 10 câu hiện tại như bảo đảm chất lượng tuyệt đối.
