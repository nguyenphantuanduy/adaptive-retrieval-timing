# Prompt cho Codex — Triển khai chỉnh sửa báo cáo ART DACN

Bạn đang làm việc tại root của repository `adaptive-retrieval-timing`.

Hãy đọc **đầy đủ, trước khi chỉnh sửa**:

1. `docs/ART_DACN_IMPLEMENTATION_PLAN.md` — kế hoạch, thứ tự triển khai, xử lý xung đột và tiêu chí nghiệm thu.
2. `docs/ART_DACN_POST_EDIT_REVIEW.md` — toàn bộ **24 issue đã được duyệt**, các đoạn thay thế nguyên văn và các addendum.
3. `docs/ART_CASE_STUDY_INVESTIGATION.md` — điều tra MuSiQue ID 348; đối chiếu thêm các JSONL nguồn.

**Nhiệm vụ:** Triển khai đầy đủ các chỉnh sửa đã duyệt cho báo cáo LaTeX theo kế hoạch, tái sinh assets cần thiết, build PDF, kiểm tra kết quả và báo cáo trạng thái từng issue. Bạn **được phép chỉnh sửa báo cáo ngay**, không cần hỏi lại về 24 ý đã duyệt. Không sửa code model, dữ liệu hay kết quả thực nghiệm.

**Quy tắc đặc biệt:** Issue 23 (case MuSiQue 348) thay thế nội dung case cũ ở Issues 03 và 19; Issue 01 vẫn yêu cầu hiển thị đủ từng passage. Issue 22 amendment yêu cầu **7 tài liệu** với official venue URLs. Issue 14 addendum chuẩn hóa ký hiệu decision score `d` khi đúng code. Issue 24 phân biệt GitHub public với Hugging Face dự kiến public. Không được dùng đoạn verbatim cũ để đưa Steve Irwin trở lại báo cáo.

Bắt đầu bằng cách kiểm tra `git status`, đọc tài liệu và xác định những thay đổi đã thực hiện sẵn; không ghi đè chỉnh sửa chưa commit của người dùng. Tiến hành theo các giai đoạn A–G. Khi có xung đột giữa dữ liệu gốc và nội dung đã duyệt, không tự chế bằng chứng; ghi rõ blocker và giải quyết các phần độc lập. Không chạy GPU experiments, commit, push hay tạo PR.

**Đầu ra:** Báo cáo bằng tiếng Việt gồm checklist 24 issue (done/already done/blocked), danh sách file thay đổi, nguồn dữ liệu đã kiểm chứng, kết quả build và kiểm tra PDF, những vấn đề còn tồn tại. Dừng sau khi hoàn tất và chờ tôi review diff.
