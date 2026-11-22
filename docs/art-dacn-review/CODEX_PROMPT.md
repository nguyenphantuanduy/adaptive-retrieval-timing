# Prompt for Codex (paste into Codex chat)

Bạn đang làm việc trong repository `adaptive-retrieval-timing`. Tôi đã hoàn tất review học thuật báo cáo Đồ án chuyên ngành ART và **cho phép bạn triển khai chỉnh sửa báo cáo LaTeX** theo bộ tài liệu sau (đặt trong `docs/art-dacn-review/`):

1. `docs/art-dacn-review/README.md`
2. `docs/art-dacn-review/IMPLEMENTATION_BRIEF.md` — **yêu cầu triển khai chính**
3. `docs/art-dacn-review/EVIDENCE_AND_ACCEPTANCE.md` — **checklist kiểm chứng/nghiệm thu**
4. `docs/art-dacn-review/ART_DACN_REVIEW_LOG.md` — **toàn bộ quyết định U-001–U-015, R-001–R-008**, bao gồm bản Abstract tham chiếu được duyệt.

**Đọc hết bốn file trước khi sửa.** Sau đó tự khám phá cấu trúc repository, LaTeX report, official HCMUT template, source code, experiment notebooks, raw benchmark artifacts, demo-v3 artifacts, evaluation scripts và bibliography. Không mặc định đường dẫn `result/demo-v3` đúng; hãy tự tìm kết quả demo-v3 thực tế. Đọc và kiểm chứng trực tiếp nguồn trước khi đưa thông tin vào báo cáo.

**Thứ tự ưu tiên:** HCMUT template > academic-paper conventions > quyết định đã chốt. Giữ tên phương pháp ART. Luận điểm trung tâm: nghiên cứu retrieval timing được học gián tiếp từ trade-off giữa LM Loss và retrieval regularization, không cần nhãn trực tiếp cho retrieval decisions hay RL. H1/H2 chỉ kiểm chứng giả định nền tảng; benchmark đánh giá hệ thống ART hoàn chỉnh. Phân biệt `d` và `tau`, retrieval count và total cost, system-level F1 improvement và causal effect. Giữ primary `tau=0.4297`, sweep best chỉ exploratory. R-007 đã **CLOSED / NO CHANGE**, không thêm phân tích/thí nghiệm timing riêng. R-008 chốt Conclusion tự tin nhưng đúng phạm vi evidence.

**Phạm vi thực thi:** Chỉnh TeX nội dung từ front matter đến phụ lục/references theo brief; vẽ sơ đồ hữu ích bằng TikZ source riêng, compile PDF vector và `\includegraphics` khi phù hợp; đồng bộ thuật ngữ, trích dẫn, số liệu và ví dụ MuSiQue xuyên suốt. Giữ nguyên raw model outputs/passages/GT/trajectories khi trích dẫn, không paraphrase như nguyên văn. Không tự tạo số liệu, citation, sample hoặc kết quả; không chạy lại training/evaluation, không sửa experimental artifacts. Không khẳng định tài nguyên GitHub/HF đã public. Không đưa lịch sử v1/v2 vào method narrative.

**Workflow bắt buộc:**
- Giai đoạn 1: audit và liệt kê nguồn bằng chứng cùng blockers; tạo kế hoạch chỉnh sửa theo U/R IDs. Nếu có blocker ở một phần, tiếp tục các phần độc lập có đủ evidence và báo rõ phần chưa hoàn thành.
- Giai đoạn 2: triển khai các chỉnh sửa đã được duyệt. Không hỏi lại những quyết định đã có trong brief; chỉ hỏi khi thiếu dữ liệu thiết yếu mà repo không cung cấp hoặc phát hiện xung đột không thể tự giải quyết an toàn.
- Giai đoạn 3: build báo cáo bằng toolchain thực tế; kiểm tra PDF render, captions, refs, citations, overfull boxes, hình và bố cục; sửa lỗi.
- Giai đoạn 4: báo cáo file đã sửa, PDF đầu ra, checklist pass/fail, các nguồn bằng chứng và blockers còn lại. Đánh dấu từng U/R đã thực hiện hay blocked; không báo `DONE` khi chưa kiểm chứng.

**Chất lượng ưu tiên hơn độ dài:** viết mạch lạc, ngắn gọn, có logic nghiên cứu; tránh lặp, tránh thổi phồng nhưng cũng không tự hạ thấp đóng góp của ART. Hãy bắt đầu bằng cách đọc bộ tài liệu và audit repository, sau đó triển khai đến khi build/validation hoàn tất.
