# ART DACN — Collaborative Review & Decision Log

> Status: ACTIVE — discussion only. No report edits authorized by this file.
> Source under review: `build/latex/dacn/main.pdf` (29-page PDF supplied by author, November 2026).
> Public-facing name: **ART (Adaptive Retrieval Timing)**. Internal legacy identifiers may remain only where needed for reproducibility.

## Working agreement

1. **Author findings first.** Record and discuss every issue the author raises before turning to assistant findings.
2. Discuss **one issue at a time**: observation → evidence → possible resolutions → author decision.
3. Do **not** treat an assistant suggestion as approved. Only the author's explicit confirmation makes a decision final.
4. Do **not** edit the report or instruct Codex to edit it during the review stage.
5. Preserve scientific accuracy, existing verified results, and the agreed ART naming convention.
6. If a claim requires code, notebooks, experiment artifacts or a paper not currently available, mark it as **needs verification** rather than guessing.
7. When the review is complete, compile only **DECIDED** items into an implementation brief for Codex.
8. **Writing hierarchy:** Follow the official HCMUT template wherever it prescribes content, headings, or formatting. For aspects the template leaves unspecified, use accepted academic-paper writing conventions; then apply the author-approved project-specific choices. Never override a template requirement in the name of paper style.

## Status definitions

- `OPEN`: Issue recorded, no resolution agreed.
- `DISCUSSING`: Under active discussion.
- `DECIDED`: Author approved a specific action, not yet implemented.
- `BLOCKED`: Awaiting evidence or an essential decision.
- `DONE`: Implemented and independently checked.
- `REJECTED`: Author explicitly decided not to change it.

## Review queue

### A. Author-identified issues — PRIORITY FIRST

### U-001 — Thuật ngữ chuyên ngành bị Việt hóa quá mức

- **Origin:** Author
- **Location:** Chương 2, mục 2.1, trang PDF 12; cần rà soát toàn báo cáo sau khi chốt quy ước.
- **Status:** DECIDED
- **Observed problem:** Một số thuật ngữ AI/ML phổ biến bị dịch thuần Việt khiến văn bản thiếu tự nhiên và khó liên hệ với code/paper: “bộ nhớ đệm khóa và giá trị”, “bộ truy hồi dày đặc”, “chú ý nhân quả”, “bộ giải mã”.
- **Evidence / source:** Đoạn mục 2.1 do tác giả trích và PDF đã cung cấp.
- **Discussion and options:** Đề xuất giữ nguyên những thuật ngữ kỹ thuật chuẩn như KV cache, Dense Retriever, Causal Attention, Decoder; tiếp tục dùng tiếng Việt cho thuật ngữ đã tự nhiên như mô hình, huấn luyện, truy vấn, hàm mất mát. Cần thống nhất quy tắc lần xuất hiện đầu, viết hoa và glossary trước khi sửa hàng loạt. Ví dụ viết lại trong chat là minh họa, chưa được duyệt.
- **Author's decision:** Đã chốt: báo cáo vẫn viết bằng tiếng Việt, nhưng giữ nguyên tiếng Anh cho thuật ngữ kỹ thuật AI/ML phổ biến (bao gồm Attention, Decoder, Embedding, Inference, Training, KV cache, Dense Retriever, Causal Attention, Attention Mixing). Dùng tiếng Việt cho từ đã thông dụng, tự nhiên như mô hình, truy vấn, dữ liệu, kết quả. Không dịch máy móc; thống nhất thuật ngữ xuyên suốt và giải thích ngắn ở lần đầu khi cần. Không đổi ý nghĩa khoa học.
- **Implementation notes for Codex:** Khi kết thúc review, rà soát toàn bộ report, thay các bản dịch gượng ép bằng thuật ngữ tiếng Anh chuẩn; không thay đổi công thức, số liệu, trích dẫn hoặc tên ART. Chỉ thực hiện khi có brief tổng hợp.
- **Validation after changes:** Pending


### U-002 — Lời cam đoan phải tuân thủ template HCMUT

- **Origin:** Author
- **Location:** Trang Lời cam đoan, PDF p. 2.
- **Status:** DECIDED
- **Observed problem:** Nội dung lời cam đoan đã bị viết lại, không bám nguyên mẫu chính thức.
- **Author's decision:** Khôi phục nguyên văn lời cam đoan từ template gốc; chỉ điều chỉnh đại từ số nhiều thành số ít cho đồ án cá nhân, tên GVHD (PGS. TS. Quản Thành Thơ), sinh viên (Nguyễn Phan Tuấn Duy), và ngày tháng khi được xác nhận. Giữ đủ các cam kết về trích dẫn, gian lận, tác quyền và bản quyền. Không tự diễn giải lại.
- **Implementation notes for Codex:** Lấy trực tiếp từ LaTeX template; build và kiểm tra trang PDF. Chưa được phép sửa cho đến đợt tổng hợp.
- **Validation after changes:** Pending

### U-003 — Tóm tắt cần ngắn gọn, đúng cấu trúc abstract khoa học

- **Origin:** Author
- **Location:** Tóm tắt, PDF p. 4.
- **Status:** DECIDED
- **Observed problem:** Tóm tắt hiện tại quá nặng thuật ngữ dịch thuần Việt và chi tiết cơ chế, chưa tối ưu về độ ngắn gọn và cấu trúc.
- **Author's desired outcome:** Tóm tắt tiếng Việt–Anh hybrid, không quá dài, nhưng đủ problem, project objective, method, implemented work, and results.
- **Discussion and options:** Assistant đề xuất 3 đoạn khoảng 200–250 từ: problem/objective; ART method and H1/H2; benchmarks and F1 improvements. Tác giả đã phê duyệt hướng chuẩn hóa và bản tóm tắt ngắn gọn được đề xuất trong chat; sử dụng bản này làm nội dung tham chiếu khi triển khai.
- **Author's decision:** APPROVED (2026-11-22). Chuẩn hóa Tóm tắt theo 3 đoạn, khoảng 200–250 từ, Vietnamese–English hybrid; trình bày Problem/Objective → Method + H1/H2 → Results; tránh công thức và chi tiết kỹ thuật không cần thiết. Dùng bản tóm tắt đã được đề xuất và đồng ý trong chat làm bản tham chiếu; giữ chính xác các số liệu, không suy diễn nhân quả từ so sánh system-level.
- **Implementation notes for Codex:** Đợi brief tổng hợp mới sửa. Dùng bản nội dung đã được tác giả đồng ý (ghi lại dưới đây), đồng bộ thuật ngữ theo U-001, giữ số liệu và tránh causal attribution quá mức.
- **Approved reference draft:**

  **Tóm tắt**

  Trong bài toán **Multi-hop Question Answering (Multi-hop QA)**, việc xác định các tài liệu cần thiết thường phụ thuộc vào kết quả của những bước suy luận trung gian. Các phương pháp Retrieval-Augmented Generation (RAG) với retrieval cố định có thể bỏ sót bằng chứng quan trọng hoặc thực hiện những lượt truy hồi không cần thiết, làm tăng chi phí suy luận. Đồ án tập trung nghiên cứu khả năng học **thời điểm retrieval thích ứng** mà không cần nhãn giám sát trực tiếp cho từng quyết định hoặc sử dụng Reinforcement Learning.

  Đồ án đề xuất **Adaptive Retrieval Timing (ART)**, một phương pháp sử dụng Decision Head kết hợp với Attention Mixing để học tín hiệu retrieval thông qua Language Modeling Loss. Quá trình huấn luyện áp dụng cơ chế Collect–Replay nhằm truyền gradient đến Decision Head. Bên cạnh việc xây dựng và huấn luyện mô hình, đồ án thực hiện hai thí nghiệm H1 và H2 để khảo sát ảnh hưởng của supporting documents, distractor documents và mixing coefficient đến LM Loss và gradient.

  ART được đánh giá trên ba benchmark **MuSiQue, HotpotQA và 2WikiMultiHopQA**, mỗi benchmark gồm 500 câu hỏi. Tại retrieval threshold 0.4297, hệ thống cải thiện F1 lần lượt **20.92, 8.63 và 8.49 điểm phần trăm** so với Vanilla RAG. Thí nghiệm threshold sweep cũng cho thấy sự đánh đổi giữa chất lượng câu trả lời và số lượt retrieval. Các kết quả bước đầu cho thấy tiềm năng của ART trong bài toán adaptive retrieval, đồng thời cung cấp bằng chứng thực nghiệm về tín hiệu học từ Attention Mixing.

  **Từ khóa:** Adaptive Retrieval Timing, Multi-hop Question Answering, Retrieval-Augmented Generation, Attention Mixing, Collect–Replay.

- **Validation after changes:** Pending

### B. Assistant-identified candidates — DEFERRED

These are **review candidates only**, not accepted findings or instructions to Codex. Discuss after the author finishes their list.

| ID | Location (PDF) | Candidate issue | Status |
|---|---|---|---|
| A-01 | pp. 14–15, Chapter 3 | Explain the learning mechanism and gradient path in more depth | OPEN |
| A-02 | pp. 16–18, Chapter 4 | Clarify collect/replay and inference algorithm; consider verified pseudocode | OPEN |
| A-03 | pp. 19–24, Chapter 5 | Deepen interpretation of benchmark and threshold results | OPEN |
| A-04 | pp. 12–13, Chapter 2 | Sharpen comparison with related work and research gap | OPEN |
| A-05 | p. 21, Figs. 5.2–5.3 | Check readability and consistency of H2 figure labels | OPEN |
| A-06 | p. 22, Table 5.1 | Ensure Acc vs EM/F1 metric definitions are sufficiently clear | OPEN |
| A-07 | Main chapters, pp. 10–25 | Consider whether the technical discussion is too compressed; do not add filler | OPEN |

### U-004 — Bổ sung chi phí latency của retrieval trong phần động lực

- **Origin:** Author
- **Location:** Chương 1, mục 1.1 (và liên hệ 1.2).
- **Status:** DECIDED
- **Observed problem:** Lập luận retrieval tại mọi bước mới đề cập distractor/noise và context length; thiếu thời gian thực thi và chi phí pipeline retrieval.
- **Author proposal:** Đề cập latency, tài nguyên tính toán và các bước có thể có như query rewriting, embedding, vector search trên corpus lớn, reranking. Đây là các ví dụ pipeline RAG nói chung, không mặc nhiên khẳng định ART hiện thực đầy đủ những bước này.
- **Evidence needed:** Kiểm tra implementation và số đo nếu muốn tuyên bố chi phí thực nghiệm; nếu chưa đo, chỉ phát biểu định tính.
- **Author decision:** APPROVED (2026-11-22). Bổ sung chi phí latency và compute của retrieval pipeline (query rewrite, embedding, search trên corpus lớn, reranking) theo lập luận định tính; không gán các bước này cho implementation ART hay khẳng định có đo latency khi chưa có evidence.
- **Implementation notes for Codex:** Chưa được phép chỉnh báo cáo.

### U-005 — Case study MuSiQue nhất quán xuyên các chương

- **Origin:** Author
- **Location:** Chương 1 mục 1.2; tái sử dụng ở chương mô tả ART và chương thực nghiệm.
- **Status:** DECIDED
- **Author proposal:** Tìm trong `result/demo-v3` một sample MuSiQue ưu tiên thỏa No-RAG fail, Vanilla RAG fail, ART pass; ART retrieve được passage cần thiết ở thời điểm hợp lý. Trình bày nguyên văn question, ground truth, generated output của baseline và ART, passage được retrieve và retrieval trajectory; tuyệt đối không tự sáng tác hoặc paraphrase model output như trích dẫn nguyên văn.
- **Validation requirements:** Xác minh sample ID và kết quả theo metric; đảm bảo các hệ thống dùng cùng question và đúng output artifact; kiểm tra passage nào được thấy tại từng thời điểm, có thật sự hỗ trợ answer không; không chọn chỉ vì ART đúng nếu retrieval timing không minh họa được cơ chế. Phân biệt nguyên văn (verbatim) với lời phân tích tác giả. Nếu output dài, cân nhắc trích đoạn có dấu lược bỏ minh bạch và dẫn phụ lục đầy đủ.
- **Presentation options:** LaTeX-native figure/table/flow preferred for reproducibility and exact text; rendered illustration optional only if it faithfully reproduces verified content and remains legible. Do not generate fictitious retrieved passages or outputs with image generation.
- **Author decision:** APPROVED (2026-11-22). Chọn sample MuSiQue chất lượng cao từ result/demo-v3, cùng câu hỏi và kết quả No-RAG fail, Vanilla RAG fail, ART pass, có bằng chứng retrieval đúng lúc và passage hữu ích. Chương 1 CHỈ hiển thị No-RAG và Vanilla RAG, question, ground truth, outputs và retrieved passages nguyên văn dưới dạng diagram LaTeX, phần phân tích tách biệt bên dưới. Chương 3–5 mới tiết lộ và phân tích ART trên chính sample đó. Không bịa hay paraphrase raw outputs/passages. Sample ID sẽ do Codex tìm và xác minh bằng artifact, chưa chốt trước. Nếu không có case đạt tiêu chí, báo lại thay vì chế dữ liệu.
- **Implementation notes for Codex:** Chưa được phép chỉnh báo cáo hoặc lựa chọn sample không kiểm chứng.

## Decision register

- **U-001 (DECIDED):** Văn phong tiếng Việt với thuật ngữ AI/ML chuẩn bằng tiếng Anh; áp dụng cả Tóm tắt, từ khóa, bảng và hình. Chờ đợt sửa tổng hợp.
- **U-002 (DECIDED):** Lời cam đoan theo nguyên mẫu HCMUT, chỉ cá nhân hóa các trường cần thiết.
- **U-003 (DECIDED):** Chuẩn hóa Tóm tắt theo bản 3 đoạn được tác giả đồng ý; Vietnamese–English hybrid, đầy đủ vấn đề, mục tiêu, phương pháp, H1/H2 và kết quả. Chờ đợt sửa tổng hợp.
- **U-004 (DECIDED):** Bổ sung định tính chi phí latency/compute của retrieval pipeline ở mục 1.1, phân biệt hệ thống RAG nói chung và ART hiện thực.
- **U-005 (DECIDED):** Chương 1 dùng diagram LaTeX nguyên văn chỉ No-RAG và Vanilla RAG; phân tích riêng dưới hình. Dùng lại cùng MuSiQue sample để giới thiệu ART ở chương sau; Codex phải xác minh artifacts, không bịa.
- **U-006 (DECIDED):** Chuẩn hóa các mục 1.1–1.4 thành research narrative; mục 1.3 thể hiện research objective/question và ba trọng tâm khoa học. HCMUT template ưu tiên cao nhất; paper conventions cho phần chưa quy định.
- **U-007 (DECIDED):** Mục 1.4 trình bày Scope & Contributions đầy đủ, công bằng, theo academic paper standards và research storytelling nếu phù hợp; xem xét đóng góp thực nghiệm H1/H2, phương pháp ART, và đánh giá hệ thống, không áp đặt cấu trúc ba nhóm hoặc phóng đại novelty/causality.
- **U-008 (DECIDED):** Related Work chọn lọc IRCoT/Self-RAG/R-Search; research gap về learning mechanism; methodology-oriented, không SOTA claim.
- **U-009 (DECIDED):** Chương 3 chuẩn paper và sát implementation; sơ đồ TikZ biên dịch riêng; mô tả cấu hình cuối, không kể v1/v2.

## Issue entry template

### U-001 — [Short title]

- **Origin:** Author
- **Location:** Chapter / page / figure / table
- **Status:** OPEN
- **Observed problem:**
- **Evidence / source:**
- **Discussion and options:**
- **Author's decision:** Not decided
- **Implementation notes for Codex:** Not authorized yet
- **Validation after changes:** Pending

## Change history

- 2026-11-22: Initialized collaborative review log. Author findings take precedence; assistant findings deferred. No report modifications authorized.

- 2026-11-22: Ghi nhận U-001 từ tác giả; đang thảo luận, chưa chốt quyết định.

- 2026-11-22: Bổ sung U-002 (DECIDED) và U-003 (DISCUSSING); dự thảo abstract ngắn gọn đang chờ tác giả duyệt.

- 2026-11-22: Tác giả phê duyệt U-003; chuyển DECIDED, lưu nguyên bản Tóm tắt tham chiếu và tiêu chí triển khai. Chưa sửa báo cáo.

- 2026-11-22: Ghi nhận U-004 (retrieval latency/cost) và U-005 (MuSiQue case study xuyên suốt) ở trạng thái DISCUSSING; chưa có artifact demo-v3 để kiểm chứng.

- 2026-11-22: Tác giả chốt U-004 và U-005; Chương 1 chỉ baseline No-RAG/Vanilla RAG, diagram nguyên văn và phân tích riêng, ART trình bày sau.

### U-006 — Chuẩn hóa mục 1.3 và mạch nghiên cứu Chương 1

- **Origin:** Author
- **Location:** Chương 1, trọng tâm mục 1.3; mở rộng rà soát tính nhất quán giữa các mục 1.1–1.4.
- **Status:** DECIDED
- **Observed problem:** Mục 1.3 chỉ liệt kê H1, H2 và benchmark, chưa thể hiện đầy đủ research objective và các đóng góp.
- **Author's decision (2026-11-22):** Tuân thủ cấu trúc/tiêu đề/nội dung bắt buộc của HCMUT template; các phần template không mô tả chi tiết thì vận dụng chuẩn trình bày academic paper. Chuẩn hóa các mục Chương 1 thành research narrative nhất quán, không bắt buộc trình bày tuần tự năm bước workflow dưới dạng danh sách. Mục 1.3 nêu research question/objective và ba trọng tâm: kiểm chứng tín hiệu học qua H1/H2; xây dựng ART (Decision Head, Attention Mixing, Collect–Replay); đánh giá trên benchmark và threshold sweep. Giữ tên mục “Mục tiêu” nếu template quy định; chỉ đổi thành “Mục tiêu và đóng góp nghiên cứu” nếu template cho phép.
- **Cross-section editorial plan:** 1.1 Motivation/research gap, bổ sung latency/cost theo U-004; 1.2 Problem formulation và MuSiQue baseline-only verbatim case study theo U-005; 1.3 Research objective/question and contributions; 1.4 Scope/contributions, phân biệt điều đã thực hiện với giới hạn và future work. Tránh trùng lặp đóng góp giữa 1.3 và 1.4; kiểm tra template trước khi quyết định cách phân bổ.
- **Scientific caution:** H1/H2 khảo sát tín hiệu, không chứng minh tuyệt đối retrieval timing; benchmark là system-level comparison, không khẳng định causal attribution. Không tự thêm thí nghiệm hoặc kết quả.
- **Implementation notes for Codex:** Chưa được phép chỉnh báo cáo cho tới brief tổng hợp. Khi triển khai, kiểm tra template và bằng chứng source; bảo toàn mọi quyết định U-001, U-004, U-005.
- **Validation after changes:** Pending

- 2026-11-22: Ghi nhận U-006 về cấu trúc storytelling cho mục 1.3; đang thảo luận.

- 2026-11-22: Tác giả phê duyệt chuẩn hóa các mục Chương 1 theo paper style trong phạm vi template cho phép; U-006 chuyển DECIDED.

### U-007 — Làm rõ và trình bày đầy đủ đóng góp nghiên cứu ở mục 1.4

- **Origin:** Author
- **Location:** Chương 1, mục 1.4 “Phạm vi và đóng góp” (PDF trang 11).
- **Status:** DECIDED
- **Observed problem:** Mục 1.4 hiện chỉ nêu ngắn gọn Attention Mixing, Collect–Replay, vòng inference và benchmark, chưa phản ánh đầy đủ đóng góp nghiên cứu của H1/H2 và chưa phân biệt rõ đóng góp khoa học với hoạt động triển khai.
- **Author question:** Việc thiết kế và kiểm chứng giả thuyết H1/H2 có được tính là research contribution hay không?
- **Discussion:** Có thể xem thiết kế kiểm chứng có hệ thống và các phát hiện về LM Loss/gradient theo supporting/distractor documents là một empirical contribution, với điều kiện mô tả đúng evidence và giới hạn; không gọi đó là chứng minh tối ưu hay khẳng định novelty tuyệt đối khi chưa đối chiếu literature. Nhóm đóng góp dự kiến: (1) empirical investigation H1/H2; (2) ART method gồm Decision Head/Attention Mixing/Collect–Replay; (3) system implementation and evaluation trên 3 benchmarks và threshold sweep. Đảm bảo mục 1.3 là objectives, mục 1.4 là actual contributions/scope để tránh lặp.
- **Author decision (2026-11-22):** Chuẩn hóa mục 1.4 theo chuẩn trình bày academic paper trong phạm vi HCMUT template cho phép; nêu **đầy đủ và fair** các đóng góp thực tế, gồm giá trị kiểm chứng H1/H2 nếu được evidence hỗ trợ, thiết kế/hiện thực ART, và đánh giá thực nghiệm; ưu tiên trình tự research storytelling khi tự nhiên. Đây là **định hướng**, không phải yêu cầu sao chép một bản nháp hay ép chia đúng ba nhóm. Tránh trùng lặp với mục 1.3; không phóng đại novelty, tính nhân quả, mức độ chứng minh hoặc kết quả.
- **Implementation notes for Codex:** Chưa được phép chỉnh báo cáo; tuân thủ **HCMUT Template → Academic Paper Standards → Quyết định riêng của chúng ta**; kiểm tra evidence trước khi phát biểu claim.
- **Validation after changes:** Pending

- 2026-11-22: Mở U-007 về đóng góp H1/H2 và hoàn thiện mục 1.4; đang chờ tác giả chốt.

- 2026-11-22: Tác giả chốt U-007 (DECIDED): chuẩn hóa Scope & Contributions theo paper standards, đầy đủ/fair và theo research storytelling khi phù hợp; không áp đặt cấu trúc, chưa giao Codex chỉnh báo cáo.


### U-008 — Định vị Related Work và toàn báo cáo theo nghiên cứu cơ chế học

- **Origin:** Author
- **Location:** Chương 2 (đặc biệt 2.2–2.3), đồng thời áp dụng xuyên suốt báo cáo.
- **Status:** DECIDED
- **Observed problem:** Related Work liệt kê quá nhiều paper, research gap chưa tập trung vào cơ chế học; cách kể chuyện thiên về F1/SOTA tạo ra so sánh thiếu kiểm soát.
- **Author's decision (2026-11-22):** Ưu tiên selective, comparative Related Work với Vanilla RAG làm baseline và IRCoT, Self-RAG, R-Search đại diện các hướng reasoning-interleaved, reflection-token supervision, và RL. Research gap đặt vào khả năng học retrieval timing qua LM Loss và differentiable attention mixing mà không cần nhãn retrieval-decision trực tiếp hay RL. Định vị ART là **methodology-oriented research**, không phải SOTA performance research; áp dụng nhất quán Abstract, Introduction, Related Work, Methodology, Experiments và Conclusion. Giữ Vietnamese–English hybrid theo U-001.
- **Scientific boundaries:** Không tuyên bố các công trình trước không thể học timing, ART là phương pháp đầu tiên, hay ART vượt các paper trên F1 khi chưa có controlled comparison. H1/H2 hỗ trợ tính khả thi của tín hiệu học cục bộ, không chứng minh tối ưu policy; benchmark là system-level evidence. Nêu rõ hạn chế thiếu controlled ablation.
- **Implementation notes for Codex:** Kiểm tra slide research gap, literature notes, original papers, source và evidence; chọn bảng đối chiếu theo learning mechanism, supervision signal, retrieval decision strategy, không theo leaderboard. Chưa sửa báo cáo cho đến khi có brief tổng hợp.
- **Validation after changes:** Pending

### U-009 — Chuẩn hóa Chương 3 và chỉ trình bày phương pháp cuối cùng

- **Origin:** Author
- **Location:** Chương 3, đặc biệt mô tả dữ liệu và các sơ đồ.
- **Status:** DECIDED
- **Observed problem:** Cần trình bày chuẩn paper, sát code/artifacts; mô tả training data còn nhắc phiên bản phát triển “v2”, không phù hợp với bản báo cáo cuối.
- **Author's decision (2026-11-22):** Giữ hướng nội dung Chương 3 vì không có lỗi lớn, nhưng chỉnh theo HCMUT Template → Academic Paper Standards → author decisions; dùng Vietnamese–English hybrid; giải thích mạch khoa học H1/H2 → cơ chế ART, nhất quán ký hiệu/công thức, mô tả đúng Decision Head, Attention Mixing, gradient path, Collect–Replay, retrieval cost và các giới hạn. Với sơ đồ phức tạp, Codex có thể vẽ bằng LaTeX/TikZ trong file độc lập, biên dịch thành PDF/ảnh vector và nhúng bằng `\includegraphics`, thay vì đặt code vẽ trực tiếp trong nội dung báo cáo; lưu source để tái lập.
- **Versioning decision:** Không dùng các nhãn v1/v2/demo-2/demo-3 để kể lịch sử phát triển trong phần trình bày phương pháp; mô tả cấu hình cuối cùng thực sự được sử dụng. Vẫn phải công bố các khác biệt experimental setup ảnh hưởng tới reproducibility hoặc diễn giải H1/H2 và benchmark; không được gộp sai các cấu hình.
- **Evidence checks for Codex:** Đối chiếu MuSiQue hypothesis probes, teacher Qwen3-8B, four few-shot examples, insufficient-evidence instructions, MuSiQue training data và ba benchmark MuSiQue/HotpotQA/2WikiMultiHopQA với code/config/artifacts trước khi khẳng định. Không thêm thành phần hoặc kết quả chưa xác minh.
- **Implementation notes for Codex:** Chỉ ghi nhận yêu cầu trong giai đoạn review; chưa được phép chỉnh báo cáo.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt U-008 về methodology-oriented positioning, selective Related Work và research gap.
- 2026-11-22: Tác giả chốt U-009 về Chương 3: chuẩn paper, đúng implementation, sơ đồ TikZ biên dịch riêng và bỏ lịch sử phiên bản khỏi narrative phương pháp.

### U-010 — Trực quan hóa kiến trúc và luồng hiện thực ART trong Chương 4

- **Origin:** Author
- **Location:** Chương 4 — Thiết kế và hiện thực hệ thống.
- **Status:** DECIDED
- **Author's decision (2026-11-22):** Giữ ưu tiên **HCMUT Template → Academic Paper Standards → Quyết định của tác giả**. Chương 4 tập trung vào thiết kế/hiện thực ART, không chuyển phần kiểm chứng giả thuyết H1/H2 hoặc kết quả thí nghiệm từ Chương 5 sang đây. Chương 3 giải thích phương pháp, Chương 4 giải thích implementation, Chương 5 cung cấp experimental evidence. Chỉ liên hệ H1/H2 ngắn gọn khi thực sự cần thiết.
- **Visual presentation priority:** Ưu tiên **sơ đồ trực quan, dễ đọc, dễ hiểu** để diễn giải kiến trúc và các luồng hoạt động thay vì chỉ mô tả dài bằng văn bản. Các ứng viên gồm: (1) ART overall architecture/data flow; (2) Attention Mixing với hai attention branches, mixing coefficient và gradient flow; (3) Collect–Replay/training pipeline; (4) autoregressive inference/retrieval workflow, bao gồm KV Cache nếu đúng với implementation. Không bắt buộc đủ bốn hình; Codex chọn hoặc gộp hình theo giá trị giải thích, tránh trùng lặp với Chương 3 và tránh rườm rà.
- **Diagram production:** Ưu tiên LaTeX/TikZ ở file source độc lập, biên dịch thành **PDF vector** rồi nhúng vào báo cáo bằng `\includegraphics`; giữ source để tái lập. Mỗi hình cần caption, nhãn/tham chiếu và phần giải thích tương ứng trong nội dung. Sơ đồ phải phản ánh chính xác source code/config/artifacts, không tự thêm thành phần, gradient path hoặc xử lý KV Cache chưa được kiểm chứng.
- **Other implementation guidance:** Trình bày rõ training và inference, Collect–Replay, Decision Head, Attention Mixing và các implementation details cần thiết để hiểu/tái lập phương pháp; không kể lịch sử v1/v2/demo. Không lặp công thức dài ở Chương 3 hay kết quả H1/H2 ở Chương 5.
- **Implementation notes for Codex:** Đây là định hướng và tiêu chí nghiệm thu, **không áp đặt chính xác số hình, layout hay đoạn văn**; Codex kiểm tra HCMUT template, source và artifacts trước khi triển khai. Chưa được phép chỉnh báo cáo cho tới khi kết thúc review và có brief tổng hợp.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt U-010, nhấn mạnh ưu tiên TikZ/vector diagrams cho Attention Mixing, kiến trúc và workflow nhằm tăng tính trực quan; chưa chỉnh sửa báo cáo.

### U-011 — Tái cấu trúc lập luận và minh chứng trong Chương 5

- **Origin:** Author
- **Location:** Chương 5, các mục 5.1–5.4.
- **Status:** DECIDED
- **Author's decision (2026-11-22):** Chỉnh Chương 5 theo thứ tự ưu tiên **HCMUT Template → Academic Paper Standards → Quyết định của tác giả**. Ưu tiên lập luận ngắn gọn, mạch lạc, tránh lặp hoặc diễn giải dài không tạo thêm giá trị; giữ đúng evidence và số liệu.
- **5.1 Experimental setup:** Trong phần thăm dò H1/H2, giải thích ngắn gọn đây là thiết lập **ART tối giản tập trung vào Attention Mixing** nhằm khảo sát cơ chế, không đồng nhất với full ART end-to-end. Codex phải tự đọc source code/config để xác định chính xác kiến trúc lite, các thành phần giữ/bỏ, điều kiện cố định, phép quét mixing coefficient, LM Loss và gradient; không suy diễn chi tiết từ mô tả sơ bộ của tác giả. Nếu cần bổ sung chi tiết kiến trúc để tái lập, ưu tiên phụ lục. Viết lại cả “Hai phép thăm dò cơ chế” và “Đối chứng hệ thống” theo mạch mục đích → thiết lập → phép đo → tính công bằng/giới hạn, súc tích và nhất quán.
- **5.2 H1/H2 results:** Tổ chức lại theo giả thuyết → kết quả quan sát → giải thích → phạm vi kết luận, tránh diễn giải trùng lặp. Phân biệt forced mixing coefficient d trong probes và decision threshold τ ở inference; không khẳng định τ=0.4297 là tối ưu được chứng minh. Giữ nguyên evidence thực tế, cải thiện hình/nhãn nếu cần.
- **5.3 QA quality and retrieval load:** Nội dung nhìn chung ổn, chỉ chỉnh nhẹ cho mạch lạc/chính xác.
- **5.3.1 Qualitative comparison:** Thay ví dụ hiện tại bằng **chính mẫu MuSiQue thống nhất ở U-005/Chương 1**: No-RAG sai, Vanilla RAG sai, ART đúng, với bằng chứng retrieval timing có ý nghĩa. Trình bày câu hỏi, ground truth và **nguyên văn raw model outputs** của No-RAG, Vanilla RAG, ART; tuyệt đối không tự viết lại, diễn giải hoặc làm đẹp câu trả lời của model. Có thể trình bày song song 3 hệ thống để so sánh; bình luận ngắn, chỉ các điểm thực sự quan trọng. **Không mặc định kết quả nằm ở `result/demo-v3`: tác giả xác nhận đường dẫn đó không đúng. Codex tự tìm và xác minh vị trí artifacts/results của demo-v3 trong repo hoặc môi trường thực tế**, đối chiếu sample ID, raw generations, retrieved passages, ground truth, trajectories. Nếu không có mẫu đáp ứng đầy đủ, báo blocker; không bịa dữ liệu.
- **5.4 Literature comparison/discussion:** Bỏ Adaptive-RAG khỏi đối chiếu. So sánh fair theo đúng learning mechanism, supervision, optimization và experimental comparability, nhưng không tự hạ thấp đóng góp của ART. Phân tích rõ lợi thế thiết kế không cần retrieval-decision labels trực tiếp hoặc RL; có thể thảo luận training cost/computational requirements **chỉ trên cơ sở paper/code/config/measurements đã kiểm chứng**. Phân biệt chi phí thành phần/độ phức tạp phương pháp với chi phí GPU-hours hoặc tổng compute đã đo; không khẳng định ART rẻ hơn định lượng hoặc tổng thể nếu chưa có controlled measurements. Giao Codex chủ động xây dựng lập luận khoa học thuyết phục, có chứng cứ và giới hạn trung thực; không biến so sánh thành cuộc đua SOTA F1.
- **Cross-section narrative:** Scientific evidence (H1/H2) → system performance → qualitative evidence → research contribution. Đồng bộ với U-005, U-008, U-009 và U-010.
- **Implementation notes for Codex:** Đây là brief yêu cầu, không áp đặt câu chữ; Codex phải kiểm tra source/artifacts và template trước khi sửa. **Chưa được phép chỉnh báo cáo** trước khi hoàn tất review và brief tổng hợp.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt U-011 và đính chính rằng đường dẫn `result/demo-v3` không phải nơi chứa kết quả; Codex phải tự xác định đường dẫn artifacts demo-v3 đúng.

### U-012 — Chuẩn hóa Chương 6: Kết luận, hạn chế và hướng phát triển

- **Origin:** Author
- **Location:** Chương 6 — Kết luận và hướng phát triển.
- **Status:** DECIDED
- **Author's decision (2026-11-22):** Tuân thủ thứ tự **HCMUT Template → Academic Paper Standards → Quyết định của tác giả**, đồng bộ với research narrative và các kết quả thực sự được trình bày trong báo cáo. Kết luận phải đầy đủ nhưng súc tích, nhấn mạnh **đóng góp về learning mechanism**: research gap → H1/H2 và bằng chứng có điều kiện → Attention Mixing/Decision Head/Collect–Replay → hiện thực ART → kết quả đánh giá ba benchmark. Không biến F1/SOTA thành trọng tâm hay diễn giải H1/H2 thành chứng minh tối ưu retrieval policy.
- **Limitations:** Nêu hạn chế khoa học cụ thể và fair, phân biệt thiếu controlled ablations/đo lường với các kết quả đã thực sự xác nhận; không lặp disclaimer quá mức hoặc tự hạ thấp đóng góp. Tránh đưa các vấn đề quản lý phiên bản/artifacts thuần túy vào hạn chế khoa học nếu không ảnh hưởng trực tiếp đến reproducibility hay độ tin cậy kết quả.
- **Future directions:** Giữ **hai hướng đang có trong báo cáo**: (1) mở rộng sang quyết định chọn công cụ (*tool selection*); (2) nghiên cứu huấn luyện đồng thời retriever và generator. Đây chỉ là **hướng khảo sát/nghiên cứu tiềm năng**, chưa bảo đảm phù hợp, khả thi hay cải thiện hiệu năng; trình bày động cơ, câu hỏi mở và điều kiện cần kiểm chứng, không viết thành cam kết. Các kiểm chứng bổ sung (controlled ablations, threshold validation, computational cost) có thể nêu như bước cần thiết, không tự nhận đã thực hiện.
- **Implementation notes for Codex:** Kiểm tra chính xác Chương 6 hiện tại, HCMUT template, kết quả và source evidence trước khi chỉnh. Giữ lối viết mạch lạc, không áp đặt câu chữ. **Chưa chỉnh sửa báo cáo** cho tới khi hoàn tất review và có brief tổng hợp.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt U-012 (DECIDED).

### U-013 — Chọn lọc tài liệu tham khảo liên quan trực tiếp đến ART

- **Origin:** Author
- **Location:** References; đồng bộ với Related Work Chương 2 và Discussion Chương 5.
- **Status:** DECIDED
- **Author's decision (2026-11-22):** Chỉ giữ các tài liệu có vai trò rõ ràng trong báo cáo: cơ sở phương pháp ART thực sự sử dụng, mô hình/dataset/benchmark cần dẫn nguồn, hoặc related work liên quan trực tiếp để đối chiếu cơ chế học và retrieval timing. Không đưa paper tràn lan để tăng số lượng trích dẫn.
- **Priority related work:** Tối thiểu phải **xem xét và dẫn nguồn phù hợp cho Self-RAG, IRCoT, R-Search**; đây là các phương pháp đối chiếu quan trọng, **không đồng nghĩa ART trực tiếp dùng thuật toán của chúng**. Đồng bộ lựa chọn selective Related Work ở U-008; bỏ Adaptive-RAG khỏi phần đối chiếu theo U-011 nếu không còn vai trò trích dẫn độc lập thực sự cần thiết.
- **Citation integrity:** Mỗi mục References phải được trích dẫn thực sự và phục vụ một luận điểm cụ thể; mọi citation trong nội dung phải có mục thư mục tương ứng. Codex xác minh tên paper, tác giả, năm, venue và metadata từ nguồn đáng tin cậy/chính thức; tuân thủ định dạng HCMUT template. Không loại bỏ nguồn nền tảng cần thiết chỉ vì không trực tiếp nói về retrieval timing.
- **Implementation notes for Codex:** Đối chiếu toàn bộ citations, nội dung Chương 2/5, source và bibliography hiện tại; ưu tiên tính chính xác, chọn lọc, liên quan, không áp đặt số lượng references cứng. **Chưa chỉnh sửa báo cáo** cho tới khi hoàn tất review và có brief tổng hợp.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt U-013 (DECIDED) cùng U-012.

### U-014 — Chuẩn hóa phụ lục, minh chứng nguyên văn và thông tin hiện vật

- **Origin:** Author
- **Location:** Phụ lục A; Phụ lục B.1 “Hiện vật và quy trình”; Phụ lục B.2 “Giới hạn của phép đếm tải huấn luyện”.
- **Status:** DECIDED
- **Author's decision (2026-11-22):** Chuẩn hóa phụ lục theo **HCMUT Template → Academic Paper Standards → Quyết định của tác giả**. Chỉ giữ nội dung bổ trợ thật sự cần thiết cho kiểm chứng, giải thích hoặc tái lập nghiên cứu; diễn đạt ngắn gọn, đúng trọng tâm, tránh lặp lại phần chính và không kể lịch sử phát triển không cần thiết.
- **Appendix A — Examples/evidence:** Dùng ví dụ nhất quán với mẫu MuSiQue đã chọn ở U-005/U-011 khi phù hợp. Câu hỏi, ground truth, raw answers của No-RAG/Vanilla RAG/ART, retrieved passages và trajectory liên quan phải **trích nguyên văn từ artifacts đã xác minh**; tuyệt đối không tự diễn giải, viết lại hoặc “làm đẹp” output của model. Nếu cần lược bỏ vì quá dài, đánh dấu rõ phần lược bỏ và không làm sai ngữ cảnh. Nhận xét phân tích đặt bên dưới, ngắn gọn và dựa trên bằng chứng. Codex tự tìm vị trí thực tế của demo-v3 artifacts, không giả định `result/demo-v3`.
- **Appendix B.1 — Artifacts and workflow:** Tác giả đã có các địa chỉ: GitHub `https://github.com/nguyenphantuanduy/adaptive-retrieval-timing.git`; Hugging Face organization `https://huggingface.co/adaptive-retrieval-timing`; model/checkpoint `https://huggingface.co/adaptive-retrieval-timing/ART-REVIVE`. **Các tài nguyên chưa public theo xác nhận của tác giả**; không mô tả là đã công khai hoặc người đọc có thể truy cập/tải/tái lập ngay. Codex xác minh phạm vi hiện vật, tình trạng truy cập và các bước reproducibility trước khi viết. Giữ literal technical identifier `ART-REVIVE` ở đường dẫn/checkpoint khi cần, nhưng public-facing method name vẫn là ART. Trình bày ngắn gọn những hiện vật và bước thiết yếu để kiểm chứng/tái lập, không lặp Chương 4 hay liệt kê đường dẫn/log nội bộ rườm rà; không hứa thời điểm public.
- **Appendix B.2 — Limits of training-load count:** Viết lại theo logic **đại lượng được đếm → phạm vi phép đếm → các chi phí không được bao gồm → kết luận được phép rút ra**. Codex đối chiếu code/config/artifacts để xác minh chính xác từng thành phần được/không được tính; không đồng nhất số trajectories, steps hoặc tokens với tổng GPU-hours/FLOPs/computational cost. Tránh nhắc lại dài dòng những hạn chế đã nêu ở Chương 5/6.
- **Implementation notes for Codex:** Đây là tiêu chí nội dung, không áp đặt câu chữ hay số trang; phải đọc template, phụ lục hiện tại và artifacts trước khi chỉnh. **Chưa được phép chỉnh báo cáo** cho tới khi hoàn tất review và lập brief tổng hợp.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt U-014 (DECIDED), hoàn tất 14 quyết định trong vòng review do tác giả khởi xướng.

### U-015 — Nguyên tắc chỉnh sửa bao trùm toàn bộ báo cáo

- **Origin:** Author
- **Location:** Toàn bộ báo cáo, từ trang đầu đến phụ lục và tài liệu tham khảo.
- **Status:** DECIDED (nguyên tắc bao trùm do tác giả xác lập ngày 2026-11-22)
- **Priority:** **HCMUT Template → Academic Paper Standards → Các quyết định review của tác giả**. Kiểm tra template gốc và bảo đảm không thiếu bất kỳ thành phần bắt buộc nào. Không tự ý bỏ, đổi cấu trúc hoặc viết lại các văn bản quy định như Lời cam đoan; chỉ điều chỉnh các trường cá nhân/ngôi xưng thực sự cần thiết theo U-002.
- **Language:** Văn phong học thuật tiếng Việt, kết hợp English AI/ML terminology tự nhiên và nhất quán theo U-001. Không Việt hóa máy móc các thuật ngữ kỹ thuật phổ biến.
- **Evidence integrity:** Mọi example, câu hỏi, ground truth, raw model outputs, retrieved passages và trajectory phải được đối chiếu artifacts gốc và trích nguyên văn. Không tự ý dịch, diễn đạt lại, sửa hay làm đẹp câu trả lời của model. Nhận xét của tác giả tách biệt, ngắn gọn; lược bỏ nội dung dài phải đánh dấu rõ và không làm sai nghĩa.
- **Diagrams:** Ưu tiên sơ đồ cho kiến trúc, Attention Mixing, Collect–Replay, training/inference flows và các nội dung thật sự được hưởng lợi từ hình minh họa. Dùng LaTeX/TikZ trong source độc lập, biên dịch thành vector image (ưu tiên PDF) và nhúng vào báo cáo; bảo đảm đúng code, dễ đọc, ít chữ, có caption/tham chiếu. Không vẽ hình cho đủ số lượng hay thay mọi giải thích bằng hình.
- **Argumentation:** Mạch lập luận rõ, logic, súc tích, đúng trọng tâm, tránh lặp và tránh kéo dài chỉ để tăng trang. Mọi khẳng định phải tương xứng bằng chứng; trung thực về giới hạn nhưng không tự hạ thấp đóng góp phương pháp ART. Kiểm chứng các chi tiết kỹ thuật bằng code, experiment artifacts, template và nguồn trích dẫn, không tự bịa.
- **Implementation notes for Codex:** Các nguyên tắc này áp dụng xuyên suốt và điều phối U-001 đến U-014; không được ưu tiên một chỉnh sửa cục bộ nếu xung đột với template. **Chưa chỉnh sửa báo cáo** cho tới khi có brief triển khai cuối cùng và tác giả cho phép.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả xác lập bộ nguyên tắc chỉnh sửa bao trùm (U-015).

### R-001 — Threshold selection và test-set selection bias

- **Origin:** Independent scientific review
- **Location:** Chương 5, đặc biệt bảng benchmark mặc định, threshold sweep và bảng đối chiếu paper.
- **Status:** DECIDED (tác giả đồng ý phương án A, 2026-11-22)
- **Decision:** Dùng **ART tại threshold vận hành chung τ = 0.4297** làm **primary evaluation** trên MuSiQue, HotpotQA và 2WikiMultiHopQA. Các threshold tối đa hóa F1 được quan sát trên từng benchmark trong sweep chỉ là **exploratory/sensitivity analysis**, không trình bày như kết quả kiểm thử độc lập của một cấu hình đã được chọn trước.
- **Observed sweep points (verify against artifacts):** MuSiQue τ=0.60, F1=32.08; HotpotQA τ=0.50, F1=49.68; 2WikiMultiHopQA τ=0.4297, F1=46.30. Không thay đổi, bỏ hoặc chế thêm số liệu; chỉ chuẩn hóa nhãn và cách diễn giải.
- **Scientific rationale:** Chọn threshold theo F1 trên chính test set có nguy cơ selection bias. Phân biệt rõ primary fixed-threshold evaluation và best-observed exploratory results. Không khẳng định τ=0.4297 là tối ưu được chứng minh; không lẫn forced mixing coefficient d của H1/H2 với inference threshold τ.
- **Paper comparison:** Ưu tiên dùng kết quả fixed τ=0.4297 cho ART ở bảng đối chiếu; nếu nêu thêm best-observed sweep, ghi rõ là exploratory và không dùng như bằng chứng thắng paper khác trong so sánh độc lập. Vẫn lưu ý khác biệt hệ thống/baselines theo U-008 và U-011, không tự hạ thấp đóng góp learning mechanism.
- **Implementation notes for Codex:** Đọc bảng, text và artifacts gốc trước khi chỉnh; đồng bộ toàn bộ abstract, bảng, discussion và conclusion khi nhắc kết quả; tuân thủ U-015. **Chưa sửa báo cáo** cho đến khi kết thúc review và tác giả cho phép.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chấp thuận đề xuất R-001, phương án A.

### R-002 — Baseline fairness và giới hạn quy kết nhân quả cho retrieval timing

- **Origin:** Independent scientific review
- **Location:** Chương 5.1.2, benchmark results, Discussion/Limitations và các phát biểu tổng kết liên quan.
- **Status:** DECIDED (tác giả chọn phương án A, 2026-11-22)
- **Decision — phương án A (thực hiện trong lần chỉnh báo cáo này):** Giữ nguyên các benchmark và số liệu hiện có; trình bày đối chiếu No-RAG, Vanilla RAG và ART như **system-level comparison**. Nêu rõ ART dùng checkpoint đã fine-tune, trong khi các baseline dùng base checkpoint; prompt, answer extraction và retrieval configuration có thể khác. Vì vậy không quy toàn bộ chênh lệch F1 cho riêng retrieval timing hoặc Attention Mixing.
- **Evidence interpretation:** Phân biệt **H1/H2** (bằng chứng có điều kiện về learning signal qua Attention Mixing) với **end-to-end benchmark** (chất lượng hệ thống trong thiết lập thực nghiệm). Khẳng định các kết quả được hỗ trợ, không suy diễn kết luận nhân quả chưa kiểm chứng. Nêu thiếu controlled ablation như một giới hạn khoa học cụ thể, súc tích, không lặp disclaimer hay tự hạ thấp đóng góp cơ chế ART.
- **Deferred — phương án B:** Nếu sau này có thời gian/tài nguyên, tác giả có thể thực hiện **controlled ablations** (ví dụ so sánh trên checkpoint và thiết lập được kiểm soát, nếu khả thi về kiến trúc). Đây là **công việc có thể cân nhắc sau**, không phải yêu cầu bắt buộc cho vòng chỉnh báo cáo hiện tại; không khẳng định đã chạy hoặc chắc chắn sẽ chạy.
- **Implementation notes for Codex:** Đối chiếu lại code, evaluation scripts, prompts, checkpoints và artifacts để mô tả chính xác các khác biệt; đồng bộ với U-008, U-011, U-012, U-015 và R-001. **Chưa chỉnh sửa báo cáo** cho tới khi review hoàn tất và tác giả cho phép.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chọn phương án A cho R-002; phương án B chỉ cân nhắc sau nếu có thời gian.

### R-003 — Retrieval timing như hệ quả của trade-off trong objective

- **Origin:** Independent scientific review + clarification from author.
- **Location:** Abstract, Chương 1 (research thesis), Chương 3 (objective/learning mechanism), Chương 5 (H1/H2 và retrieval count), Chương 6, Phụ lục B.2 khi liên quan.
- **Status:** DECIDED (tác giả đồng ý và làm rõ ý tưởng cốt lõi, 2026-11-22).
- **Core research framing (author):** Đề tài nghiên cứu liệu **hành vi retrieval timing có thể được học như một hệ quả của trade-off** giữa LM Loss và retrieval regularization, thay vì cần retrieval-decision labels trực tiếp hoặc RL. Đây là research hypothesis / cơ chế tối ưu cần được hỗ trợ bằng thực nghiệm, không phải định lý bảo đảm tối ưu toàn cục.
- **Mechanistic intuition:** Với một decision score d trong dạng đơn giản L(d)=L_LM(d)+λd, gradient cục bộ là ∂L/∂d=∂L_LM/∂d+λ. Khi lợi ích cục bộ từ retrieved evidence đủ lớn (∂L_LM/∂d < −λ), tín hiệu gradient descent có xu hướng tăng d; nếu không đủ lớn, penalty có xu hướng kéo d xuống. Trong training thực tế, cập nhật diễn ra qua tham số Decision Head và Attention Mixing; không hứa từng score sẽ thay đổi đúng chiều ở mọi bước. Codex xác minh cách áp dụng regularizer và các vị trí j từ code thực tế.
- **Scientific precision:** Retrieval regularization/surrogate penalty khác với discrete retrieval count và khác với wall-clock latency, GPU-hours, FLOPs hay total computational cost. H2 local gradient analysis không tự chứng minh threshold tối ưu hay tối ưu hóa compute thực tế. Retrieval count là metric inference được đo riêng.
- **Writing guidance:** Nêu rõ và nhất quán luận điểm trade-off → learning signal → emergent/learned retrieval timing trong phần động cơ, methodology và kết luận; tránh lặp dài dòng, tránh phóng đại. Giữ trọng tâm đóng góp là **learning mechanism**. Đối chiếu nội dung mô tả đề tài gốc nếu Codex có thể truy cập; không giả vờ đã xác minh nguyên văn mô tả đó nếu chưa đọc.
- **Implementation notes for Codex:** Kiểm tra source code, training objective và experimental artifacts trước khi sửa; tuân thủ U-015. **Chưa chỉnh sửa báo cáo** cho tới khi review hoàn tất và tác giả cho phép.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt R-003 và nhấn mạnh retrieval timing là hành vi học được như hệ quả của trade-off, đúng định hướng mô tả đồ án.

### R-004 — Phân biệt kiểm chứng giả định nền tảng và đánh giá ART

- **Origin:** Independent scientific review + clarification from author.
- **Location:** Chương 3 (đặt giả thuyết), Chương 5.1–5.3 (thiết lập, H1/H2 và benchmark), Chương 6 (kết luận) khi cần.
- **Status:** DECIDED (tác giả chốt, 2026-11-22); **gộp vào nguyên tắc lập luận khoa học hiện có**, không tạo thêm hạng mục thí nghiệm bắt buộc.
- **Author's clarification:** H1 và H2 chỉ kiểm chứng **hai giả định nền tảng của ART** về quan hệ giữa LM Loss, Attention Mixing/decision signal; chúng **không trực tiếp chứng minh hiệu quả của toàn bộ ART**. Việc đánh giá hệ thống ART đã được huấn luyện thuộc về **ART end-to-end experiments**.
- **Evidence roles:** H1 khảo sát LM Loss khi thay đổi mixing coefficient trong các điều kiện bằng chứng; H2 khảo sát gradient của LM Loss theo mixing coefficient và khả năng cung cấp learning signal trong các điều kiện đã kiểm tra. ART benchmarks đánh giá chất lượng Multi-hop QA và retrieval behavior/count của hệ thống hoàn chỉnh.
- **Claim boundaries:** Không dùng H1/H2 để khẳng định ART end-to-end đã được chứng minh hiệu quả; không dùng benchmark để suy ra toàn bộ tính chất gradient hay quy kết mọi cải thiện F1 riêng cho retrieval timing (theo R-002). Dùng ngôn ngữ tương xứng với bằng chứng, không làm giảm giá trị đóng góp cơ chế.
- **Writing guidance:** Trình bày mạch lạc **giả định nền tảng → H1/H2 verification → cơ chế ART/Collect–Replay → ART system evaluation**, theo vị trí các chương đã thống nhất; không lặp dài dòng, không cần thí nghiệm mới hay đoạn limitations bổ sung chỉ vì R-004.
- **Implementation notes for Codex:** Tuân thủ U-008, U-011, U-015, R-002, R-003; kiểm tra code và artifacts trước khi diễn giải. **Chưa chỉnh sửa báo cáo** cho tới khi hoàn tất review và tác giả cho phép.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt R-004 theo hướng phân biệt rõ H1/H2 foundation verification và ART system evaluation.

### R-005 — Diễn giải rõ evaluation metrics kế thừa từ R-Search

- **Origin:** Independent review + author clarification.
- **Location:** Chương 5.1.2 (evaluation metrics), bảng kết quả và bibliography khi cần.
- **Status:** DECIDED (tác giả chốt, 2026-11-22).
- **Author's clarification:** Cách tính Acc, EM, F1 hiện tại được tác giả triển khai theo R-Search. Không thay đổi metric hay tính lại benchmark chỉ vì cách đặt tên dễ gây hiểu nhầm.
- **Decision:** Yêu cầu Codex kiểm tra evaluation script thực tế và đối chiếu paper/code R-Search trước khi mô tả nguồn gốc metric. Giữ nguyên định nghĩa, công thức và số liệu đã xác minh; viết lại lời giải thích cho dễ hiểu, đặc biệt nêu chính xác Acc đang đo gì và phân biệt với EM. Trích dẫn R-Search khi xác minh được. Không tự giả định paper dùng cùng implementation nếu chưa đối chiếu.
- **Scope:** Chỉnh cách diễn giải evaluation protocol, không đổi phương pháp ART, không thêm thí nghiệm, không chỉnh báo cáo cho tới khi tác giả cho phép.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả xác nhận metric theo R-Search và yêu cầu Codex chỉ đối chiếu, diễn giải rõ ràng hơn.

### R-006 — Phân biệt retrieval count và total computational cost

- **Origin:** Independent scientific review; author decision.
- **Location:** Chương 5 (benchmark retrieval count và thảo luận), Chương 6 và các phát biểu về efficiency/cost khi liên quan.
- **Status:** DECIDED — phương án A (tác giả chốt, 2026-11-22).
- **Decision:** Giữ nguyên các kết quả và số liệu thực nghiệm hiện có. Diễn giải **retrieval count** là số lần gọi retrieval được ghi nhận trong thiết lập đánh giá, phản ánh hành vi sử dụng retrieval; **không đồng nhất** chỉ số này với wall-clock latency, GPU-hours, FLOPs hoặc tổng computational cost.
- **Claim boundaries:** Không suy luận rằng số lần retrieval cao/gấp k lần có nghĩa tổng chi phí cao/gấp k lần. Tổng chi phí còn có thể phụ thuộc vào generation, embedding, search, context length và cấu hình hệ thống; chỉ nêu những thành phần phù hợp với pipeline thực tế, không giả vờ đã đo các thành phần chưa được đo.
- **Implementation notes for Codex:** Đối chiếu code và artifacts để xác nhận định nghĩa retrieval count; chỉ chuẩn hóa câu chữ trong báo cáo, không thay metric, không chạy thêm thí nghiệm. Đồng bộ với R-003 và U-015. **Chưa chỉnh sửa báo cáo** cho tới khi tác giả cho phép.
- **Validation after changes:** Pending

- 2026-11-22: Tác giả chốt phương án A cho R-006.

### R-007 — Retrieval timing versus retrieval frequency
- **Status:** CLOSED / NO CHANGE (author decision, 2026-11-22).
- **Decision:** No new timing-specific experiment or qualitative analysis. Existing U-005/U-011 MuSiQue case study already addresses the desired illustration. Codex must implement that agreed example faithfully, using only verified trajectories and without fabricating decision scores. Do not add an independent R-007 task.

### R-008 — Calibrate conclusions to available evidence
- **Status:** DECIDED — option A (author decision, 2026-11-22).
- **Decision:** Preserve a confident, concise conclusion about ART's learning-mechanism contribution and supported benchmark outcomes. Keep H1/H2 foundation evidence, method proposal, and system-level evaluation distinct. Do not claim globally optimal retrieval timing, causal attribution of all F1 gains to timing, or lower total compute without evidence. No new experiments; harmonize Abstract, Results and Conclusion. Merge with U-012, R-002 and R-004.

### Review closure
- **Status:** REVIEW COMPLETE (2026-11-22). U-001–U-015 and R-001–R-006, R-008 are approved implementation requirements; R-007 is closed with no additional work. The author has now authorized preparation of the implementation brief and Codex execution prompt. Implementation and validation remain pending.
