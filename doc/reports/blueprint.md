# Blueprint CI/CD cho RAG và guardrails

- **Hình thức:** cá nhân — Dương Thị Ngân, 2A202602808
- **Nguồn kết quả:** `artifacts/ragas_50q.json`, `artifacts/judge_results.json`, `artifacts/guard_results.json`
- **Guard run:** 2026-10-13T02:43:01Z
- **Models:** generation `openai/gpt-oss-20b`; judge `openai/gpt-oss-120b`; guard `qwen/qwen3.8-27b`; evaluator gồm Gemini Flash Lite, GPT-OSS, Qwen và local `sentence-transformers/all-MiniLM-L6-v2`.

Đây là đề xuất triển khai CI/CD dựa trên số đo của lab, chưa phải workflow production đã được bật.

## Kiến trúc và failure action

Luồng phục vụ đề xuất là: `input → Presidio → NeMo input rail → retrieval/rerank → generation → NeMo output rail → response`. Presidio phát hiện và ẩn email, số điện thoại, CCCD/CMND trước khi dữ liệu đi xa hơn. NeMo input chặn jailbreak, prompt injection và nội dung ngoài phạm vi; output rail chặn PII, mật khẩu và dữ liệu nhân viên trước khi trả kết quả.

Trong request production, một tầng chặn sẽ dừng sớm và trả thông báo an toàn. Log chỉ giữ category, quyết định, latency, version model/prompt và request ID; không ghi nguyên văn PII hoặc secret. API key nằm trong secret manager của CI/runtime, không nằm trong repo hay artifact. Khi evaluator hoặc guard LLM lỗi mạng, hệ thống fail closed đối với dữ liệu nhạy cảm, ghi metric lỗi và cho phép retry có giới hạn; không tự thay bằng điểm giả.

## Phạm vi và số đo latency

Task 12 đo riêng hai input layer theo thứ tự, không short-circuit: engine được khởi tạo trước, có hai warm-up thật không tính vào mẫu, sau đó đo 20 mẫu bằng `perf_counter`. Percentile dùng nội suy tuyến tính trên raw samples và tổng input guard được tính ở từng sample.

| Layer / scope | P50 ms | P95 ms | P99 ms | Tình trạng |
| --- | ---: | ---: | ---: | --- |
| Presidio | 0.141 | 0.287 | 0.298 | Đạt, chi phí rất nhỏ |
| NeMo input | 2355.138 | 3959.908 | 27467.336 | Vượt budget |
| Input guard sequential | 2355.298 | 3960.186 | 27467.482 | Vượt budget |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi Task 12 |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi Task 12 |
| Full request có short-circuit | chưa đo | chưa đo | chưa đo | Không suy ra từ input guard |

Target input-layer P95 là **500 ms**; kết quả **3960.186 ms**, do đó `latency_budget_ok=false`. Bottleneck là NeMo gọi remote LLM; một raw sample khoảng 33.34 giây làm P99 tăng mạnh. Hướng tối ưu là dùng classifier nhỏ/local cho rule rõ ràng, cache policy prompt, giới hạn timeout và chỉ gọi LLM cho trường hợp không chắc chắn. RAG, output rail và full request cần benchmark riêng với warm-up, timestamp và tối thiểu 100 mẫu trước khi đặt SLO production.

## CI gates đề xuất

- Offline: `python -m pytest tests/ -m "not integration" -q` và `python scripts/check_lab.py` phải trả mã 0.
- RAG: faithfulness tối thiểu 0.75 và mean bốn metric tối thiểu 0.65 trên đúng 50 câu cố định.
- Judge: Cohen’s κ tối thiểu 0.60; position inconsistency phải được báo cáo, không ép các `tie` thành winner.
- Guard: ít nhất 18/20 attack pass cho gate CI đề xuất; 8/8 clean phải allowed và 4/4 output phải đúng. Mốc lab tối thiểu vẫn là 15/20.
- Latency: chỉ bật gate full-guard P95 dưới 500 ms sau khi đã đo đúng full scope; số đo hiện tại của input layer chưa đạt nên pipeline sẽ cảnh báo, chưa được ghi là pass.

Một job đề xuất chạy `pytest`, tạo artifact live trong môi trường staging có secrets, sau đó chạy checker và một script gate đọc JSON. Script trả nonzero nếu metric thấp hơn `MIN_FAITHFULNESS`, `MIN_RAGAS_MEAN`, `MIN_KAPPA`, `MIN_ATTACK_PASS` hoặc nếu có false positive trên clean controls. Artifact và report được lưu theo commit SHA để truy vết; pull request không được dùng artifact của commit cũ.

## Monitoring và cải tiến

Theo dõi theo cửa sổ ngày/tuần: RAGAS theo distribution, tỷ lệ position inconsistency, Cohen’s κ trên mẫu được người review, attack pass rate, clean false-positive rate, output block rate, error/timeout rate và P50/P95/P99 từng layer. Chỉ lấy mẫu production đã loại PII; case false positive được chuyển cho người phụ trách chính sách, gắn nhãn rồi thêm vào regression set. Cảnh báo khi clean false-positive lớn hơn 0, attack pass dưới 90%, κ dưới 0.60 hoặc latency P95 vượt SLO hai cửa sổ liên tiếp.

- RAGAS mean theo distribution: factual **0.7236**, multi-hop **0.2554**, adversarial **0.1749**; nhóm thấp nhất là adversarial. Một số metric bị quota/parse failure được ghi 0, nên đây là lower bound cần rerun khi quota ổn định.
- Cohen’s κ: **1.00** trên 10 câu hiệu chuẩn.
- Guard: **20/20 attacks**, **8/8 clean**, false-positive rate **0.00**, **4/4 output**.

Bộ 50/20/8/4 còn nhỏ và cùng một miền chính sách, vì vậy chưa đại diện đầy đủ cho drift ngoài thực tế. Các bước tiếp theo có thể kiểm chứng là tăng negative/negation cases, bổ sung PDF scan bằng OCR, rerun sau mỗi thay đổi chunking/reranker, chuyển off-topic classifier sang local model để giảm P95, và định kỳ lấy mẫu ẩn do ít nhất hai người gán nhãn để đo lại κ.
