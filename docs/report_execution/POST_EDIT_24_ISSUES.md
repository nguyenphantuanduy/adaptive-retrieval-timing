# Nghiệm thu 24 chỉnh sửa báo cáo DACN ART

Đối chiếu thứ tự ưu tiên theo `docs/ART_DACN_IMPLEMENTATION_PLAN.md` và các quyết định bổ sung trong `docs/ART_DACN_POST_EDIT_REVIEW.md`. Điểm xuất phát là nhánh `dev`, HEAD `9ada1db46f028a5a9c942c354d311ded5ecd00f1`. Ba tệp đặc tả chưa được Git theo dõi đã có sẵn.

| Issue | Trạng thái | Vị trí và kết quả |
| --- | --- | --- |
| 01 | Hoàn tất | §1.2, Phụ lục A và `generate_case.py`. Hiển thị cả năm đoạn Vanilla RAG và đủ năm đoạn theo từng lượt ART, đúng thứ tự lưu. |
| 02 | Hoàn tất | §1.1. Bổ sung lý do cân nhắc chi phí và tài liệu gây nhiễu. |
| 03 | Hoàn tất theo Issue 23 | §1.2. Giải thích hai bước Michael Knight → David Hasselhoff → Snapper Foster. |
| 04 | Hoàn tất | §1.3. Câu hỏi và mục tiêu nghiên cứu. |
| 05 | Hoàn tất | §1.4. Phạm vi thiết kế, kiểm chứng và đánh giá. |
| 06 | Hoàn tất | §1.5. Đóng góp về giả thuyết, cơ chế và hiện thực. |
| 07 | Hoàn tất | §2.2, Bảng 2.1. Dùng reflection tokens cho Self-RAG và cập nhật các ô mô tả. |
| 08 | Hoàn tất theo bổ sung ký hiệu | §3.1. Dùng điểm ứng viên \(d_t\), điều kiện vị trí và truy vấn hợp lệ, phân biệt hệ số ép của probe. |
| 09 | Hoàn tất có hiệu chỉnh theo nguồn | §3.2. Tách dữ liệu probe, huấn luyện và đánh giá. Teacher tạo chuỗi đích ngoại tuyến. ART Collect dùng chuỗi đó để chọn quỹ đạo truy hồi trong huấn luyện. |
| 10 | Hoàn tất | §3.3.1, `graphics/attention.tex`. Vẽ lại hai nhánh RoPE và mask, trộn phân bố trước khi nhân với giá trị. |
| 11 | Hoàn tất | §3.3.2. Mô tả LM Loss, hệ số 0.05, tập các lần cộng điểm kể cả ở nhánh kết thúc, và đường gradient qua Replay. |
| 12 | Hoàn tất | §4.1. E5, mean pooling, chuẩn hóa L2, FAISS float32 và ánh xạ corpus. |
| 13 | Hoàn tất có hiệu chỉnh theo nguồn | §4.2.1. Theo `REVIVE_training/training_v2.py` dòng 293–304, chuỗi đích và EOS được nối trước khi tokenize, riêng với câu hỏi. Mô tả nhãn -100, tập tham số, batch và accumulation. Giữ 0.75 epoch theo xác nhận của tác giả. |
| 14 | Hoàn tất | §§4.3.1–4.3.2 và `graphics/workflow.tex`. Sơ đồ và mô tả Collect, Replay, inference, exploration và cache theo model. |
| 15 | Hoàn tất | §5.1.1. Thiết lập probe, cỡ mẫu và khác biệt giữa bốn điều kiện. |
| 16 | Hoàn tất | §5.1.2. Đối chiếu cùng 500 ID, câu hỏi và đáp án chuẩn trên từng bộ. Mô tả checkpoint và định nghĩa chỉ số. |
| 17 | Hoàn tất | §5.2.1. Đối chiếu CSV H1 và mô tả cả bốn đường, đáy trên lưới và giới hạn teacher forcing. |
| 18 | Hoàn tất | §5.2.2 và hình trade-off. Đối chiếu 10 cấu hình có metric. Dùng số truy vấn như đại lượng tải và mô tả chiều biến động F1 theo từng bộ. |
| 19 | Hoàn tất theo Issue 23 | §5.3.1. Thảo luận ID 348 theo trace và giới hạn suy diễn từ một mẫu. |
| 20 | Hoàn tất | §5.3.2. So sánh cơ chế IRCoT, Self-RAG, R-Search và giới hạn đối chứng cục bộ. |
| 21 | Hoàn tất | §§6.1–6.3. Kết luận, giới hạn, Tool Selection và huấn luyện phối hợp Retriever–Generator. |
| 22 | Hoàn tất | `refs/references.bib`. Bảy mục và URL venue, đồng bộ bảy khóa trích dẫn. |
| 23 | Hoàn tất | §§1.2, 5.3.1, Phụ lục A. ID 348, ART mặc định, EM 0/0/1 theo thứ tự No-RAG, Vanilla RAG, ART. |
| 24 | Hoàn tất | §B.1. Hai URL là hyperlink trong PDF. Trạng thái GitHub và kế hoạch Hugging Face theo xác nhận tác giả. |

## Kiểm chứng

- Tái sinh tài sản từ `report/dacn/generate_assets.py` và `report/dacn/generate_case.py`. Đối chiếu nguyên văn ba câu trả lời, năm đoạn Vanilla RAG, hai truy vấn ART và mười đoạn ART từ các JSONL gốc. Ca ID 348 ở bản ghi 349 của ba tệp. Đáp án chuẩn là Snapper Foster.
- Hai bảng benchmark `graphics/local-table.tex` và `graphics/sweep-table.tex` khớp nội dung tại HEAD. Metric và dữ liệu thực nghiệm được giữ nguyên. Các miền nội dung của H1/H2 dựa trên CSV gộp và notebook H2 giản lược.
- `scripts/build-hcmut-template.ps1 -Report` trả về 0. PDF `build/latex/dacn/main.pdf` có 35 trang. Log cuối không có file thiếu, tham chiếu/trích dẫn chưa giải, cảnh báo box hoặc glyph mất. Cảnh báo môi trường còn lại gồm MiKTeX chưa kiểm tra cập nhật, mẫu ngắt từ tiếng Việt và cảnh báo lớp memoir.
- Render bằng `scripts/inspect_pdf.py`. Kiểm tra trực quan các trang qua ảnh ghép, rồi kiểm tra trang Phụ lục B sau lần sửa phân trang cuối. Bộ ảnh cuối và báo cáo kiểm tra nằm tại `build/latex/dacn/post-edit-verified`. Không có trang bị đánh dấu rỗng hoặc không trích được chữ.
- Kiểm tra văn bản PDF, bảy khóa citation và hai hyperlink phụ lục. Tên case cũ và URL arXiv không xuất hiện trong bản đọc. `REVIVE_training` còn trong đường dẫn kỹ thuật.
- Kiểm tra độc lập trạng thái truy cập GitHub chưa thành công. Hugging Face là địa chỉ dự kiến. Hai ngoại lệ câu chữ được nêu tại Issue 09 và 13. Không thực hiện commit, push, inference hoặc đánh giá mới.

## Tệp đã sửa

- `report/dacn/chapters/main/`: `01-introduction.tex`, `02-background.tex`, `03-methodology.tex`, `04-implementation.tex`, `05-evaluation.tex`, `06-conclusion.tex`.
- `report/dacn/chapters/back/`: `appendix-a.tex`, `appendix-b.tex`.
- `report/dacn/`: `generate_assets.py`, `generate_case.py`.
- `report/dacn/graphics/`: `attention.tex`, `workflow.tex`, `case-baseline.tex`, `case-outputs.tex`, `case-trace.tex`, `h1.pdf`, `h2-cost.pdf`, `h2-gradient.pdf`, `tradeoff.pdf`.
- `report/dacn/refs/references.bib` và tài liệu nghiệm thu này.
