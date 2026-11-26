# ART DACN — Kế hoạch chỉnh sửa tổng hợp (24 ý đã duyệt)

**Trạng thái:** Được phép triển khai sau khi Codex đọc và đối chiếu nguồn.  
**Nguồn yêu cầu đầy đủ:** `docs/ART_DACN_POST_EDIT_REVIEW.md` (bản gốc được cung cấp cùng kế hoạch này).  
**Nguồn kiểm chứng case:** `docs/ART_CASE_STUDY_INVESTIGATION.md` và JSONL gốc.  
**Phạm vi:** Chỉ báo cáo LaTeX, mã/script sinh tài sản báo cáo và tài liệu kiểm chứng cần thiết. Không thay đổi model, benchmark, kết quả thực nghiệm, hoặc dữ liệu gốc.

## 0. Chuẩn bị và quy tắc bắt buộc

1. Đặt **cả file kế hoạch này** và **file review gốc** vào `docs/` của repository với đúng tên ở trên, hoặc sửa đường dẫn trong prompt cho phù hợp. Không suy diễn rằng chúng đã có trong repo.
2. Đọc **toàn bộ 24 issue và các addendum** của review log trước khi sửa. File review là nguồn chính cho các đoạn **COPY EXACTLY** và các tiêu chí kỹ thuật; kế hoạch này quy định **thứ tự và cách xử lý xung đột**, không thay thế nội dung chi tiết.
3. Kiểm tra `git status`, cấu trúc LaTeX, build commands, các script sinh hình/bảng, và các file nguồn dữ liệu. Không ghi đè thay đổi chưa commit của người dùng; nếu có thay đổi trùng phạm vi, dừng và báo.
4. Các issue ghi **VERBATIM/COPY EXACTLY**: giữ nguyên nội dung đã duyệt, chỉ chuyển cú pháp sang LaTeX, ngoại trừ các đoạn **bị supersede bởi Issue 23**. Các issue ghi **CONTENT BRIEF**: tự viết văn phong học thuật, kiểm tra code/data trước khi phát biểu.
5. Không chế số liệu, citation, evidence, inference trace, hay thí nghiệm. Không chạy lại inference/GPU nếu không được yêu cầu. Nếu phát hiện mâu thuẫn giữa đoạn đã duyệt và bằng chứng gốc, **không âm thầm sửa số liệu**: báo rõ và đề xuất xử lý.
6. Phân biệt kết quả hệ thống ART (fine-tuned) với baseline (checkpoint gốc); không quy kết nhân quả riêng cho retrieval timing. H1/H2 là probe với hệ số trộn cung cấp trực tiếp, không phải minh chứng Decision Head đã học chính sách tối ưu.

## 1. Quy tắc ưu tiên / xung đột đã xác định

- **Issue 23 supersedes nội dung case cũ trong Issues 01, 03 và 19.** Không chép nguyên văn đoạn về Nim/Bindi Irwin/Steve Irwin ở Issue 03 hoặc 19. Thay bằng diễn giải **MuSiQue ID 348 — Michael Knight → David Hasselhoff → Snapper Foster**, dựa trên trace gốc. Issue 01 **vẫn có hiệu lực về nguyên tắc hiển thị đầy đủ các passage**, áp dụng cho case mới. Nếu case đã được Codex thay trước đó, kiểm chứng và chỉ bổ sung phần còn thiếu, không thay lại vô ích.
- **Issue 22 amendment là quyết định cuối:** giữ **7 nguồn**, loại FAISS [3], E5 [7], Qwen2.5 Technical Report [8] theo số thứ tự gốc; dùng **7 venue URLs** trong amendment, không dùng arXiv. Không xóa mô tả kỹ thuật FAISS/E5/Qwen2.5 trong nội dung.
- **Issue 14 notation addendum supersedes quy ước cũ:** ưu tiên `d` / `d_t` cho decision score và vai trò attention mixing khi phù hợp code; kiểm tra nhất quán với Issues 08, 11, công thức, sơ đồ và caption. Không đổi ký hiệu cơ học nếu dẫn đến sai nghĩa.
- **Issue 24 là phiên bản cuối của B.1:** GitHub đã công khai, Hugging Face **dự kiến** công khai. Không khẳng định HF đã public.
- Mọi chỉ dẫn “chưa chỉnh báo cáo” trong review log là **trạng thái lịch sử**; yêu cầu triển khai hiện tại cho phép sửa các mục đã duyệt. Không tự mở rộng sang vấn đề chưa duyệt.

## 2. Thứ tự triển khai

### Giai đoạn A — Baseline và xác minh dữ liệu

- Ghi nhận branch, commit SHA, working-tree status và quy trình build hiện tại.
- Kiểm tra báo cáo PDF/LaTeX hiện hành, xác định các issue nào đã được thực hiện (đặc biệt Issue 23), lập checklist `pending / done / blocked`.
- Đọc mã nguồn liên quan Decision Head, Attention Mixing, Collect–Replay, inference, token masking, checkpoint; đọc artifacts H1/H2 và benchmark. Đối chiếu các nhận định định lượng trước khi viết.
- Kiểm tra `docs/ART_CASE_STUDY_INVESTIGATION.md` và dữ liệu ID 348; nếu không có file, báo thiếu thay vì tạo case giả.

### Giai đoạn B — Case study và chứng cứ (Issues 23, 01; điều chỉnh 03, 19)

- Hoàn thiện thay thế case ở §1.2, §5.3.1, Phụ lục A.1/A.2, `report/dacn/generate_case.py` và `graphics/case-*.tex` nếu chưa thực hiện.
- No-RAG: `Jesse James Byron` (EM 0); Vanilla: `Peter Bergman` (EM 0); ART: `Snapper Foster` (EM 1), đối chiếu JSONL trước khi đưa vào.
- ART retrieval 1: *Michael Knight (Knight Rider)* → David Hasselhoff; retrieval 2 có David Hasselhoff trong query → *Snapper Foster*. Chỉ kết luận bằng chứng theo trace thực tế.
- Giữ **hai truy vấn nguyên văn và đủ 5 passages mỗi lượt**, đúng thứ tự; nếu rút gọn, hiển thị **từng** passage và dấu `...`, không ẩn toàn bộ tài liệu. Kiểm tra Vanilla passages được trình bày nếu có.
- Rà soát tất cả tham chiếu đến case cũ; tránh thay nội dung unrelated nếu là tên người xuất hiện trong nguồn trích dẫn độc lập.

### Giai đoạn C — Chương 1–3 (Issues 02–11)

- **Chương 1:** Issue 02 §1.1 (verbatim); Issue 03 §1.2 **chỉ giữ mục tiêu giải thích hai-hop, viết lại theo case 348**; Issue 04 §1.3 (verbatim); Issue 05 §1.4 (verbatim); Issue 06 §1.5 (verbatim).
- **Chương 2:** Issue 07 — thuật ngữ Self-RAG/reflection tokens và bảng Related Work (verbatim các ô được duyệt).
- **Chương 3:** Issue 08 §3.1 Decision Head và thời điểm quyết định (verbatim nếu phù hợp addendum notation); Issue 09 §3.2 phân biệt dữ liệu probe / training / evaluation (verbatim); Issue 10 §3.3.1 vẽ TikZ Attention Mixing rõ hai nhánh, RoPE/masks và công thức trộn; Issue 11 §3.3.2 diễn giải mục tiêu học và gradient path theo brief, không tự chế đạo hàm.
- Rà soát nhất quán score `d_t`, ngưỡng `τ`, mixing, H1/H2; nếu đoạn verbatim mâu thuẫn ký hiệu được duyệt sau, ưu tiên addendum và ghi nhận điều chỉnh.

### Giai đoạn D — Chương 4 (Issues 12–14)

- Issue 12 §4.1 mô tả E5/FAISS/corpus (verbatim).
- Issue 13 §4.2.1 chuẩn bị chuỗi, nhãn -100, số lớp được fine-tune, batch/accumulation (verbatim, xác minh code/checkpoint).
- Issue 14 §4.3.1–4.3.2: phân biệt Collect (discrete/no-grad), Replay (differentiable training) và inference; xử lý đúng retrieval, exploration, cache. Dùng ảnh `ART Retrieval Workflow Diagram.png` trong project làm tham chiếu, có thể vẽ lại TikZ paper-quality; xác minh mọi con số/ký hiệu từ code, không coi ảnh là nguồn kỹ thuật tối hậu.

### Giai đoạn E — Chương 5–6 (Issues 15–21, ngoại trừ case cũ)

- Issue 15 §5.1.1: thiết lập H1/H2 và cỡ mẫu (verbatim).
- Issue 16 §5.1.2: baseline 500 mẫu/bộ, top-5, checkpoint khác nhau, Acc/EM/F1 (verbatim); không chép chữ chỉnh sửa ngẫu nhiên nếu có trong giao diện soạn thảo.
- Issue 17 §5.2.1: **đọc plot/data rồi tự phân tích** xu hướng bốn đường H1, không áp đặt kết luận trước khi kiểm chứng.
- Issue 18 §5.2.2: **đọc plot/data rồi tự phân tích** số retrieval và F1; không ép kết luận trade-off, không bàn lựa chọn `τ=0.4297` ở đây, không gọi retrieval count là latency/compute thực đo.
- Issue 19 §5.3.1: **KHÔNG dùng bản Steve Irwin verbatim**, dùng case 348 theo Giai đoạn B.
- Issue 20 §5.3.2: cơ chế IRCoT/Self-RAG/R-Search và giới hạn đối chứng (verbatim).
- Issue 21 §§6.1–6.3: kết luận, hạn chế, hướng phát triển (verbatim); Tool Selection và joint Retriever–Generator là hướng nghiên cứu tiềm năng.

### Giai đoạn F — Tài liệu tham khảo và Phụ lục B (Issues 22, 24)

- Issue 22: bỏ 3 nguồn, giữ 7 nguồn với URL venue **đúng như amendment** trong review log; kiểm tra bibliography keys, citation dangling, đánh số sau build. Không dùng arXiv thay thế.
- Issue 24: thay nguyên văn B.1, link GitHub `https://github.com/nguyenphantuanduy/adaptive-retrieval-timing`, HF dự kiến `https://huggingface.co/adaptive-retrieval-timing`; tạo hyperlink trong PDF.

### Giai đoạn G — Kiểm thử, nghiệm thu và báo cáo

1. Chạy script sinh bảng/hình/case bằng quy trình repo hiện có; xác minh generated assets khớp nguồn. Tránh sửa trực tiếp file generated nếu script sẽ ghi đè.
2. Biên dịch PDF bằng lệnh hiện có, đủ số lần cho cross-references/bibliography; xử lý lỗi LaTeX, undefined citations/refs, overfull boxes đáng kể.
3. Kiểm tra trực quan các trang thay đổi: hình TikZ, Figure 4.1, case trong Chương 1/5 và Phụ lục A, bảng/đồ thị H1/H2, bibliography, B.1, phân trang và font tiếng Việt.
4. Dùng tìm kiếm toàn repo kiểm tra tên case cũ trong **nguồn báo cáo đang dùng**, placeholder bỏ passages, các URL arXiv còn sót trong 7 references, citation thiếu, các ký hiệu `s_t`/`d_t` mâu thuẫn.
5. Đối chiếu số liệu và nhận định với artifacts; xác minh không thay đổi JSONL, code model, training, benchmark, checkpoint hoặc dữ liệu gốc.
6. Báo cáo theo bảng 24 issue: `done / already done / blocked`, file và vị trí đã sửa, bằng chứng kiểm chứng, build status, PDF output path, các khác biệt có chủ đích do Issue 23 supersede.
7. **Không commit, push, mở PR hoặc chạy GPU experiments** nếu chưa được yêu cầu. Nếu vướng vấn đề không thể giải quyết mà không làm sai yêu cầu đã duyệt, dừng phần liên quan và báo người dùng; tiếp tục các phần độc lập khi an toàn.

## 3. Tiêu chí hoàn thành

- 24 ý được kiểm tra đầy đủ, không còn case cũ trong báo cáo hiện hành; các issue superseded được ghi rõ thay vì coi là bỏ sót.
- Case 348 và mọi evidence/metrics khớp JSONL; mỗi lượt retrieval hiển thị đủ top-5 passages hoặc từng excerpt có dấu lược bỏ.
- Các đoạn được duyệt verbatim được giữ nguyên, trừ chỗ xung đột với quyết định mới hơn hoặc bằng chứng thực tế; mọi ngoại lệ được báo cáo.
- Sơ đồ, ký hiệu, mô tả training/inference và kết luận khoa học nhất quán; không có tuyên bố nhân quả quá mức.
- 7 references với official venue links; GitHub public / HF planned rõ ràng.
- PDF build thành công và được kiểm tra trực quan; không sửa dữ liệu hoặc code thí nghiệm.

## 4. Ghi chú bàn giao

Đây là **kế hoạch điều phối**, không phải bản thay thế cho các đoạn nguyên văn trong `ART_DACN_POST_EDIT_REVIEW.md`. Codex **phải đọc cả hai**. Đặc biệt, các addendum ở Issues 14 và 22 cùng quyết định Issue 23 có độ ưu tiên cao hơn hướng dẫn cũ trong cùng log.
