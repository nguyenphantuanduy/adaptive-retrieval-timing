# ART DACN — Post-edit review and correction requests

## Purpose
This is a **new, separate** rolling review log for issues identified while inspecting Codex's edited ART DACN report. Record each user-approved observation sequentially. Do not implement changes until the user requests a consolidated Codex prompt or explicitly asks for execution.

## Issue 01 — MuSiQue example: do not omit retrieved passages

**Location:** User identifies section **1.2 — Phát biểu bài toán**; screenshot shows running header **1.3 — Mục tiêu**. Codex must locate the actual example in LaTeX and verify placement rather than assuming the section numbering is wrong.

**Observed problem:** A boxed example shows the question, ground truth, No-RAG and Vanilla RAG outputs, and a partial `Vanilla RAG, passage 1`, followed by the editorial placeholder `[Lược phần còn lại: 4 đoạn khác không hiển thị.]`. This omission makes the evidence incomplete and the presentation inappropriate for an academic report.

**Required changes:**
1. **Every retrieved passage must be represented individually and in original order.** Prefer displaying each passage in full. Do not replace entire passages with an editorial placeholder such as `4 đoạn khác không hiển thị`.
2. Preserve exact raw passage content from the verified experiment artifacts; do not invent, paraphrase, summarize, or silently alter it.
3. For long passages, first try natural line wrapping, page breaks, and a modestly smaller but readable academic-paper-style font. Prefer a breakable LaTeX environment (e.g. `longtable`, `tcolorbox` with `breakable`, or suitable text blocks). If full text remains impractically long, include **every passage separately**, quote several verbatim representative lines from each and then mark the omitted continuation with `...`; explicitly label such excerpts as shortened. Never omit a passage entirely.
4. Prefer full evidence even when it requires extra pages. Only use per-passage ellipses as a fallback, not the default. Maintain readable typography, spacing, headings, and captions.
5. Rebuild the PDF and inspect the actual pages for overflow, clipped text, awkward breaks, and correct placement/caption.
6. Verify the question, ground truth, outputs, passage counts and passage order against original artifacts before editing. The screenshot alone is not proof of the complete passage set.

**Status:** Recorded; awaiting more review issues and later consolidated Codex prompt.

### Confirmation after inspecting uploaded 35-page PDF (2026-11-22)
User reiterates that this omission is unacceptable. Confirmed in PDF page 11 (printed page 2), Figure 1.1, and also PDF pages 33–34 (printed pages 24–25), Appendix A.2: placeholders omit four Vanilla RAG passages, four ART first-query passages, parts of ART second-query passages, and its fourth/fifth passages. **Scope of Issue 01 explicitly includes both the chapter example and Appendix A.2, and any equivalent omission placeholders elsewhere in this case.** Do not omit whole passages. Prefer full text with multi-page typesetting; per-passage verbatim excerpts ending in `...` are permitted only when necessary for length, with all passages still individually shown and clearly labeled as abbreviated. This is an editorial/layout correction, not permission to fabricate missing retrieved evidence. Source: user-uploaded `main(1).pdf`.

### User clarification — formatting priority (2026-11-22)
The user explicitly allows **smaller readable font**, as used in research papers, when passages are lengthy. The preferred solution remains **all passages in full**. If this becomes too long, a permissible fallback is **a few verbatim lines from EACH passage followed by `...`**, not omission of entire passages or a placeholder announcing unseen passages. Every passage must remain identifiable in the original order; indicate excerpting transparently. This clarification supersedes earlier absolute prohibitions on ellipses in Issue 01.

## Issue 02 — Section 1.1: replace retrieval-cost paragraph verbatim

**Location:** Section **1.1 — Lý do chọn đề tài**.

**Problem:** The existing paragraph unnecessarily qualifies and explains its own motivation in conversational/metadiscursive terms (especially “Đây là động lực nghiên cứu, còn tác động thực tế phụ thuộc vào hiện thực và cấu hình hệ thống.”). The list of potential RAG pipeline stages should support the claim that unnecessary retrieval adds processing overhead, within one cohesive, formal academic paragraph.

**User-approved replacement (COPY EXACTLY, no paraphrasing):**

> Ngược lại, việc truy hồi quá thường xuyên không chỉ làm gia tăng lượng tài liệu gây nhiễu (*distractor documents*) trong ngữ cảnh mà còn tạo thêm chi phí xử lý. Trong một pipeline RAG, mỗi lần truy hồi có thể liên quan đến nhiều công đoạn như *query rewriting*, tạo *embedding*, tìm kiếm vector trên tập dữ liệu lớn và *reranking*. Do đó, các quyết định truy hồi không cần thiết có thể làm gia tăng khối lượng tính toán, mức sử dụng tài nguyên và độ trễ của hệ thống, đồng thời ảnh hưởng đến chất lượng suy luận do sự xuất hiện của các tài liệu không liên quan.

**Codex instructions:** Replace the corresponding original paragraph with the exact approved Vietnamese text above; retain italic emphasis for technical terms in LaTeX as appropriate, but do not alter wording or add disclaimers. Preserve surrounding argument and formatting. No new experiments or metric changes.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 03 — Section 1.2: concise multi-hop reasoning analysis (verbatim replacement)

**Location:** Section **1.2 — Phát biểu bài toán**, the explanatory paragraph immediately following the MuSiQue question/example. Replace the existing paragraph beginning “No-RAG và Vanilla RAG đều không khớp đáp án chuẩn...”

**Problem:** The current text merely states answer mismatches and identifies an intermediate entity, without explicitly explaining the two-hop reasoning dependency or why a single initial Vanilla RAG retrieval is inadequate for this case. The user requires formal, concise academic Vietnamese.

**User-approved replacement (COPY EXACTLY, no paraphrasing):**

> Đối với câu hỏi “Who is the father of Nim's actress in Return to Nim's Island?”, quá trình suy luận đòi hỏi hai bước liên tiếp: (1) xác định diễn viên thủ vai Nim trong *Return to Nim's Island* là Bindi Irwin; (2) xác định cha của Bindi Irwin là Steve Irwin.
>
> Trong khi đó, Vanilla RAG chỉ thực hiện một lượt truy hồi dựa trên câu hỏi ban đầu. Mặc dù tài liệu truy hồi cung cấp thông tin về thực thể trung gian Bindi Irwin, hệ thống không thể thực hiện truy hồi bổ sung dựa trên thực thể này và đã đưa ra câu trả lời không chính xác. Ví dụ cho thấy hạn chế của chiến lược truy hồi cố định trong các bài toán suy luận nhiều bước, qua đó đặt ra yêu cầu xác định thời điểm truy hồi thích hợp dựa trên trạng thái suy luận của mô hình.

**Codex instructions:** Replace the corresponding paragraph with exactly these two approved paragraphs, preserving wording and technical names; translate italic emphasis into LaTeX markup only. Verify the raw five Vanilla RAG passages and actual retrieved evidence before retaining any claim that the second hop lacks sufficient evidence; do not invent an unsupported causal explanation. If the raw artifacts contradict that claim, flag the discrepancy for user review instead of silently rewriting the approved text. Do not modify the report yet; include this issue in the later consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 04 — Section 1.3: adaptive retrieval timing emerges from the optimization trade-off (verbatim replacement)

**Location:** Section **1.3 — Mục tiêu**. Replace the existing paragraph beginning “Câu hỏi nghiên cứu là liệu mô hình có thể học thời điểm truy hồi từ sự đánh đổi giữa LM Loss...”

**Problem:** The research question must characterize **adaptive retrieval timing as an emergent consequence** of optimizing the trade-off between language modeling loss and retrieval regularization. Explicitly specify that the method does **not require direct supervision labels for retrieval timing** (not that it lacks supervision for the language-generation task).

**User-approved replacement (COPY EXACTLY, no paraphrasing):**

> Câu hỏi nghiên cứu đặt ra là liệu mô hình có thể học được thời điểm truy hồi thích ứng như một hệ quả của quá trình tối ưu hóa sự đánh đổi giữa mục tiêu sinh ngôn ngữ (*LM Loss*) và chi phí truy hồi được biểu diễn thông qua *retrieval regularization*, mà không cần nhãn giám sát trực tiếp về thời điểm truy hồi hoặc Reinforcement Learning. Để trả lời câu hỏi này, đồ án khảo sát các giả thuyết nền tảng H1/H2, xây dựng phương pháp kết hợp Decision Head, Attention Mixing và Collect–Replay, đồng thời đánh giá hiệu quả trên ba bộ dữ liệu benchmark và phân tích độ nhạy theo ngưỡng quyết định (*threshold sweep*).

**Codex instructions:** Replace the corresponding original paragraph with this **exact** approved text. Convert Markdown italics to appropriate LaTeX markup without changing words. Do not generalize “nhãn giám sát trực tiếp về thời điểm truy hồi” to “nhãn giám sát trực tiếp” or claim that retrieval regularization directly measures real-world latency/computational cost. Do not implement until the later consolidated prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 05 — Section 1.4: exploratory future research directions (verbatim replacement)

**Location:** Section **1.4 — Phạm vi**. Replace the existing paragraph beginning “Đồ án chuyên ngành nghiên cứu thời điểm truy hồi cho Multi-hop QA...”

**Problem:** The original wording presents tool selection and joint retriever–generator training as if they are definite next-stage work. The report should present them as potential research extensions whose feasibility and potential benefits remain to be investigated, without promising implementation or success.

**User-approved replacement (COPY EXACTLY, no paraphrasing):**

> Đồ án chuyên ngành tập trung nghiên cứu thời điểm truy hồi thích ứng cho bài toán Multi-hop QA, từ khảo sát các giả thuyết nền tảng đến xây dựng và đánh giá hệ thống. Các hướng mở rộng như lựa chọn công cụ truy hồi và huấn luyện đồng thời retriever–generator có thể được xem xét trong Đồ án tốt nghiệp nhằm đánh giá tính khả thi và tiềm năng cải thiện hiệu quả của phương pháp.

**Codex instructions:** Replace the corresponding original paragraph with the exact approved text, preserving wording and technical terms; only adapt LaTeX markup as needed. Do not convert the exploratory language into a firm commitment. Do not modify the report until the consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 06 — Section 1.5: explicitly identify the teacher model (verbatim replacement)

**Location:** Section **1.5 — Đóng góp của đồ án**, replace the paragraph beginning “Đồ án chuyên ngành đóng góp một khảo sát thực nghiệm có kiểm soát...” and ending with the ambiguous reference to “giáo viên”.

**Problem:** The phrase “dữ liệu văn bản từ giáo viên” can be misconstrued as referring to the human thesis supervisor. The intended source is an AI **teacher model** used in the Collect–Replay training process. State its role precisely and present the experimental, methodological, and implementation contributions concisely.

**User-approved replacement (COPY EXACTLY, no paraphrasing):**

> Đồ án chuyên ngành đóng góp một khảo sát thực nghiệm có kiểm soát về ảnh hưởng của bằng chứng truy hồi và hệ số Attention Mixing đối với LM Loss và gradient. Về phương pháp, đồ án xây dựng cơ chế học quyết định truy hồi thông qua Decision Head và Attention Mixing khả vi, kết hợp quy trình Collect–Replay sử dụng teacher model để tạo quỹ đạo suy luận phục vụ huấn luyện. Về hiện thực, hệ thống tích hợp quyết định truy hồi vào quá trình sinh tự hồi quy và đánh giá trên ba bộ dữ liệu Multi-hop QA, với các phép so sánh ghép cặp cùng No-RAG và Vanilla RAG.

**Codex instructions:** Replace the corresponding paragraph with this exact approved wording, adapting only LaTeX syntax as necessary. Preserve the explicit term **teacher model**, not “giáo viên” or “dữ liệu văn bản từ giáo viên”. Verify the description of the teacher model's role against the actual Collect–Replay implementation; flag any discrepancy for user review rather than silently changing the approved text. Do not modify the report until the consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 07 — Chapter 2 related-work table: standardize Self-RAG reflection-token terminology (verbatim cells)

**Location:** Chapter 2 related-work comparison table, **Self-RAG** row (previous PDF review identified the table as Table 2.1; user refers to “bảng 2.2”, so locate by table content rather than assuming numbering). Also check nearby prose for inconsistent translation.

**Problem:** The literal Vietnamese translation “token phản tư” / “các token phản tư” is awkward and potentially misleading in a technical report. Preserve the standard English term *reflection tokens* and describe their role as special control tokens. Do not suggest human-like introspection.

**User-approved replacement for the Self-RAG row (COPY EXACTLY, except LaTeX formatting):**

| Cách tiếp cận | Quyết định truy hồi | Tín hiệu cho cơ chế quyết định |
|---|---|---|
| Self-RAG | Sinh các token điều khiển (*reflection tokens*) | Học có giám sát trên các *reflection tokens* |

**Codex instructions:** Replace the corresponding two Self-RAG table cells with the exact approved wording. Adapt italic formatting to LaTeX as needed. Search the rest of the report for “token phản tư” and replace with consistent English technical terminology where context warrants; do not make unrelated content changes. Confirm the actual table number from source, and do not change report files until the consolidated prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 08 — Section 3.1: clarify last-token Decision Head and Attention Mixing (verbatim replacement)

**Location:** Section **3.1 — Tổng quan hướng tiếp cận**, paragraph beginning “Decoder Qwen2.5-3B-Instruct...” and explaining the score, threshold, mixing coefficient, and H1/H2 probes.

**Problem:** The original description does not clearly distinguish the last-token hidden state of a segment, the Decision Head's predicted retrieval score, the mixing coefficient associated with newly retrieved evidence, and the directly prescribed coefficient used in H1/H2 probes.

**User-approved replacement (COPY EXACTLY in meaning and wording; convert mathematical notation into LaTeX only):**

> Với mỗi segment \(t\) gồm \(L\) token, Decoder Qwen2.5-3B-Instruct tạo các trạng thái ẩn và sử dụng trạng thái của token cuối cùng \(h_{t,L}\) làm đầu vào cho Decision Head. Thông qua phép chiếu tuyến tính và hàm sigmoid, Decision Head dự đoán điểm quyết định truy hồi:
>
> \[s_t = \sigma(w^\top h_{t,L} + b).\]
>
> Trong giai đoạn suy luận (inference), hệ thống thực hiện truy hồi khi \(s_t \geq \tau\). Khi truy hồi được thực hiện, điểm \(s_t\) được liên kết với quan sát mới và sử dụng làm hệ số Attention Mixing \(d\) trong các lần forward Decoder tiếp theo, cho đến khi quan sát đó được thay thế bởi kết quả truy hồi mới.
>
> Đối với các thực nghiệm kiểm chứng giả thuyết H1/H2, hệ số \(d\) được thiết lập trực tiếp thay vì dự đoán thông qua Decision Head, nhằm khảo sát ảnh hưởng của Attention Mixing đến LM Loss và gradient.

**Codex instructions:** Replace the target prose verbatim; adapt equation environments and citation placement to the LaTeX document, without changing technical meaning. Preserve that the score is computed from the **last token's hidden state of the segment**, that \(s_t\) both triggers retrieval and becomes the mixing coefficient associated with the **new observation**, and that H1/H2 use **directly specified \(d\)** independent of Decision Head. In workflow figures and other explanations, disambiguate the fresh decision score \(s_t\) from the previously cached mixing coefficient \(d\); threshold comparison is against \(s_t\). Check against implementation and flag discrepancies rather than silently rewriting approved text. Do not edit report until consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 09 — Section 3.2: separate H1/H2 probe, ART training, and evaluation data (verbatim replacement)

**Location:** Section **3.2 — Dữ liệu**, replace the paragraph beginning “Các phép thăm dò dùng mẫu kiểm chứng nhiều bước của MuSiQue...” and ending with the three evaluation benchmarks.

**Problem:** The current paragraph conflates data used for H1/H2 controlled probes, ART Collect–Replay training, and final benchmark evaluation. **Both H1/H2 and ART training use teacher-model-generated data**, but for H1/H2 only state that the data were generated by a teacher model; reserve the detailed Qwen3-8B/four-shot/Collect–Replay description for ART training. Do not call the model a human “giáo viên”.

**User-approved replacement (COPY EXACTLY, no paraphrasing):**

> **Dữ liệu kiểm chứng giả thuyết H1/H2.** Các thực nghiệm sử dụng mẫu Multi-hop QA từ MuSiQue với dữ liệu được tạo bởi teacher model, phân nhóm theo số bước suy luận và tỷ lệ tài liệu hỗ trợ so với tài liệu gây nhiễu. Trong mỗi điều kiện thực nghiệm, câu hỏi, tài liệu và đích dự đoán được giữ cố định, trong khi hệ số Attention Mixing được thay đổi có kiểm soát nhằm khảo sát sự biến thiên của LM Loss và gradient.
>
> **Dữ liệu huấn luyện ART.** Dữ liệu huấn luyện được xây dựng từ tập huấn luyện MuSiQue thông qua quy trình Collect–Replay. Trong giai đoạn Collect, teacher model Qwen3-8B được sử dụng để tạo quỹ đạo suy luận với bốn ví dụ few-shot và hướng dẫn xử lý trường hợp bằng chứng chưa đầy đủ. Các quỹ đạo thu thập được sử dụng trong giai đoạn Replay để tối ưu mô hình ART.
>
> **Dữ liệu đánh giá.** Hệ thống được đánh giá trên ba bộ dữ liệu Multi-hop QA gồm MuSiQue, HotpotQA và 2WikiMultiHopQA. Các mẫu đánh giá được tách khỏi dữ liệu huấn luyện và được sử dụng để so sánh ART với các baseline No-RAG và Vanilla RAG.

**Codex instructions:** Replace the original paragraph with these **three separate paragraphs**, using bold inline labels rather than new subsections. Preserve every approved sentence verbatim, adapting only LaTeX formatting and existing citations as appropriate. Explicitly retain teacher-model generation for **both** H1/H2 and ART training, but do not elaborate the teacher generation process in the H1/H2 paragraph. Check source artifacts/code for consistency; flag discrepancies rather than silently changing the approved wording. Do not edit the report until the consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 10 — Section 3.3.1: add a colorful TikZ diagram explaining Attention Mixing

**Location:** Section **3.3.1 — Attention Mixing**, adjacent to the existing mathematical explanation.

**Problem / objective:** The section's current equations and explanation are acceptable, but a publication-quality vector schematic would make the two attention branches and their mixing mechanism clearer.

**User-approved Codex requirements:**

1. **Preserve the existing prose and equations** in Section 3.3.1; add a figure rather than rewriting the section.
2. Draw a **colorful, academically styled TikZ/LaTeX vector diagram**, using the detailed TikZ snippet provided by the user in the review conversation as a **reference for visual styling and architecture**, not as code that must be copied verbatim.
3. Clearly distinguish the **with-document** and **without-document** attention branches, relevant RoPE and attention masks, score computation, softmax, the mixing coefficient \(d\), and the core mixing equation \(A=dA_d+(1-d)A_n\), followed by the attention output.
4. Prefer a **balanced two-parallel-branches-then-converge layout** over a single excessively wide row. Use a restrained but colorful paper-style palette, legible typography, and clear arrows. Do not allow overlapping text, arrows, or page margins; fit the report's page format.
5. Include implementation-level details such as Repeat KV, dropout, dtype casting, or fallback **only when they materially aid comprehension** and are confirmed by the actual source code. Verify especially the RoPE branches, document attention mask, and fallback behavior against the implementation. Never invent architectural details to match the reference diagram.
6. Provide a meaningful figure caption, LaTeX label, and a reference to the figure in the surrounding discussion. Ensure TikZ compiles successfully and the final PDF contains a sharp vector figure.
7. Do **not** modify the report now. This issue is for the later consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 11 — Section 3.3.2: clarify learning objective and H1/H2 (content guidance, NOT verbatim replacement)

**Location:** Section **3.3.2 — Mục tiêu học và giả thuyết**.

**Problem:** The current prose compresses LM Loss, retrieval regularization, a local derivative illustration, the ART optimization mechanism, and H1/H2 into an overly dense and difficult-to-follow passage. The user approves the *conceptual direction*, but explicitly does **not** approve copying ChatGPT-generated formulas because of the risk of mathematical or implementation mismatches.

**User-approved instructions for Codex (WRITE THE PROSE YOURSELF; VERIFY FORMULAS):**

1. Preserve equations and notation **only insofar as verified against the actual implementation**. Inspect code for LM Loss computation, masking/selected token positions, retrieval regularization, the set of decision points, and the configured value of lambda. Do not blindly copy formulas from the review conversation or invent new formulas.
2. Reorganize the explanation in this order: (a) objective of learning adaptive retrieval timing; (b) LM Loss and retrieval regularization and what each does; (c) their tradeoff; (d) differentiable gradient path through Attention Mixing to Decision Head.
3. Explain that adaptive retrieval behavior can **emerge as a consequence of optimizing the tradeoff**, without **direct labels for retrieval timing** or reinforcement learning for retrieval decisions. Do not imply the model is trained without any language-model supervision.
4. Explain H1/H2 briefly and separately: controlled probes of how LM Loss and its gradient respond to evidence and directly specified mixing coefficient d. They test foundational assumptions, **not** the complete ART training process and **not** optimality of the learned policy.
5. Simplify or remove the illustrative local derivative with respect to d if it is not needed. Do not suggest that ART's optimizer directly updates a free independent variable d; parameter gradients flow through the actual computational graph.
6. Avoid extraneous implementation asides (e.g., RoPE/mask details better handled in the Attention Mixing section) and self-justifying commentary. Write in concise, formal Vietnamese suitable for an academic report.
7. If any existing equation or interpretation contradicts code or cannot be verified, **flag the discrepancy for review** rather than silently changing the mathematics or empirical claims. Retain relevant citations and ensure mathematical consistency across Sections 3.3.1, 3.3.2, and 5.

**Mode:** Conceptual/content brief, **not** an exact-text substitution. Codex owns the final wording and equation verification. Do not edit the report until the consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 12 — Section 4.1: concise and coherent system requirements description

**Location:** Section **4.1 — Yêu cầu hệ thống**.

**Problem:** The original paragraph is technically acceptable but mixes system-level components with low-level retriever implementation details and reads as rambling. The user approved a concise exact replacement.

**Approved replacement (VERBATIM; adapt only LaTeX formatting):**

> Hệ thống ART gồm Decoder tùy biến từ Qwen2, bộ truy hồi tài liệu, bộ phân đoạn câu trực tuyến và bộ điều phối huấn luyện. Bộ truy hồi sử dụng E5 để mã hóa truy vấn, kết hợp FAISS để tìm kiếm các đoạn văn liên quan. Quá trình mã hóa áp dụng tiền tố truy vấn E5, mean pooling theo attention mask và chuẩn hóa L2; các vector float32 được sử dụng để tìm kiếm trên CPU. Mỗi kết quả truy hồi được ánh xạ đến đoạn văn tương ứng trong corpus và cung cấp cho Decoder dưới dạng quan sát.

**Codex instructions:** Replace the existing paragraph with the approved text **verbatim**, making only necessary LaTeX formatting changes. Preserve technical accuracy and existing relevant citations. The old sentence about the corpus and index having equal row counts may be relocated to an appropriate detailed implementation subsection if necessary to explain FAISS-to-corpus alignment, but should not clutter this overview. Verify implementation details against code and flag discrepancies instead of silently altering approved wording. Do not edit the report until the consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 13 — Section 4.2.1: data sequence preparation and trainable parameters

**Location:** Section **4.2.1 — Chuẩn bị chuỗi và tập tham số**.

**Problem:** “token quan sát chèn thêm” obscures that the observation contains retrieval results; “Theo tác giả” inappropriately attributes the report's own training configuration to an external author. The original prose also conflates data preparation with fine-tuning settings.

**Approved replacement (VERBATIM; adapt only LaTeX formatting):**

> Câu hỏi được định dạng theo mẫu hội thoại, trong khi chuỗi đích do teacher model sinh ra và token EOS được mã hóa riêng. Các token thuộc câu hỏi và nội dung tài liệu truy hồi được gán nhãn −100 để loại khỏi quá trình tính LM Loss. Bộ nạp dữ liệu kiểm tra kiểu chuỗi và loại bỏ các mẫu có chuỗi đích rỗng.
>
> Checkpoint ART được sử dụng trong đánh giá được huấn luyện trong 0.75 epoch. Quá trình fine-tuning chỉ cập nhật 15 lớp Decoder cuối trong tổng số 36 lớp cùng với Decision Head; các lớp còn lại được đóng băng nhưng vẫn tham gia lan truyền thuận. Kích thước micro-batch là 1, với gradient accumulation qua 32 bước.

**Codex instructions:** Replace the current paragraph with the approved two paragraphs **verbatim**, changing only LaTeX syntax/formatting. Do not reintroduce “Theo tác giả”; explicitly identify the inserted observation tokens as retrieval content. Verify token-label masking, EOS handling, trainable layers, training duration, batch size, and accumulation against the actual code/configuration; flag any mismatch for user review rather than silently rewriting the approved text. Do not edit the report until the consolidated Codex prompt.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 14 — Section 4.3: rewrite Collect–Replay and autoregressive inference; use supplied workflow diagram

**Location:** Section **4.3 — Hiện thực**, including **4.3.1 Collect–Replay khi training**, **4.3.2 Inference tự hồi quy**, and **Figure 4.1**.

**Problem:** Current prose reads like a code audit, mixing the conceptual workflow with low-level KV-cache, logging, and unmeasured edge-case remarks. Figure 4.1 does not clearly communicate the full training/inference distinction. The user approves restructuring and rewriting rather than an exact-text replacement.

**User-approved instructions for Codex (CONTENT BRIEF; WRITE PROSE YOURSELF):**

1. Rewrite Section 4.3 in concise, formal academic Vietnamese. Organize by the core algorithmic flow; avoid opening with a description of the figure itself.
2. In 4.3.1, first explain the purpose of Collect–Replay, then clearly separate **Collect** (build and store a retrieval trajectory; discrete retrieval and configured exploration, without gradients) from **Replay/Training** (replay stored prefixes/contexts, recompute differentiable decision scores, calculate LM Loss plus retrieval regularization, and backpropagate). Explain why this separation is needed and what information is preserved. Check the actual implementation before asserting the exact snapshot, mask, teacher-forcing, logits, or cache behavior.
3. In 4.3.2, explain the chronological inference flow: initialize input/context and caches; generate tokens; check candidate decision points (prefill and eligible sentence boundaries); compute decision score and apply threshold plus query-validity rules; retrieve and insert observation when accepted; update attention/context/cache state; continue generation. Explain KV Cache and Attention Mixing only to the extent needed for correctness. Distinguish training exploration from inference, which does not use it in the current implementation.
4. Remove unnecessary audit-style asides from the main narrative (e.g., unmeasured historical edge-case frequencies, differences between logging list denominators, commented-out retrieval quotas). Keep or relocate edge cases that materially affect algorithmic correctness; do not erase actual limitations or invent guarantees.
5. **Use the existing image named `ART Retrieval Workflow Diagram.png` in the project folder as the primary reference for Figure 4.1.** The user has also supplied the image in this conversation (mounted reference: `/mnt/data/ART Retrieval Workflow Diagram(1).png`). Codex should first locate the project image and either (a) use it directly if legible and appropriate for the report, or (b) redraw an equivalent clean, colorful, paper-quality TikZ figure suitable for A4 layout. The visual should clearly separate three stages: **1. Collect (Build Trajectory), 2. Training (Replay & Optimize), 3. Generation (Inference)**, with their key operations and the contrast between discrete retrieval and gradient-bearing replay. Avoid overcrowding; translate labels into consistent report terminology if redrawing.
6. **Do not blindly copy the image's notation or numerical settings.** In particular, reconcile its `d >= 0.4297` decision labels, `0.05 × Σ d_i` retrieval penalty, exploration-probability update, retrieval-context and KV-cache behavior with the verified code and the report's distinction between Decision Head score `s_t` and attention-mixing coefficient `d`. If there is a discrepancy, flag it for user review rather than silently adopting an incorrect diagram. The figure is a visual reference, not authoritative proof of implementation.
7. Keep Figure 4.1 caption accurate, cite/reference it naturally from the prose, and cross-check Sections 3.1, 3.3, 4.2, and 4.3 for consistency. Do not modify the report yet; implement only after the consolidated Codex prompt.

**Mode:** Approved conceptual rewrite + figure reference, **not** exact prose substitution.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

### Addendum to Issue 14 — Decision-score notation (user clarification, APPROVED)

The user clarifies that their code and technical descriptions commonly denote the **Decision Head's decision score as `d`** (decision score). Codex should **standardize notation consistently across the report, formulas, captions, and Figure 4.1**, using `d` for the decision score where this matches the verified implementation. The score may also serve as the Attention Mixing coefficient after retrieval; distinguish its roles and any time/observation indices clearly in prose. Do **not** force a distinction between `s_t` and `d` merely because earlier review guidance used `s_t`; treat the earlier Issue 14 warning as a consistency check, not a mandate to rename the decision score. Check actual code and flag any genuine semantic conflict rather than silently altering equations. This addendum supersedes the conflicting notation guidance in Issue 14 point 6 and should also inform Issues 08 and 11.

**Status:** APPROVED clarification by user on 2026-11-22.

## Issue 15 — Section 5.1.1: rewrite the two mechanism probes

**Location:** Section **5.1.1 — Hai phép thăm dò cơ chế**.

**Problem:** The original paragraph combines probe purpose, dataset setup, H1/H2 measurement procedures, and sample counts into a dense, hard-to-follow passage. User approved separating it into three coherent paragraphs.

**Approved replacement (VERBATIM; adapt only LaTeX formatting):**

> Để kiểm chứng hai giả thuyết H1 và H2, chúng tôi sử dụng Attention Mixing Probe, một phiên bản Decoder Qwen2.5 cho phép cung cấp trực tiếp tài liệu và hệ số trộn chú ý \(d\). Việc sử dụng tài liệu có sẵn giúp khảo sát riêng tác động của Attention Mixing, không phụ thuộc vào Decision Head hoặc quá trình truy hồi tự hồi quy.
>
> Dữ liệu thí nghiệm được xây dựng từ các tập xác thực MuSiQue với câu hỏi yêu cầu 2, 3 và 4 bước suy luận. Dựa trên nhãn đoạn hỗ trợ của bộ dữ liệu, mỗi mẫu được cung cấp ba đoạn văn theo bốn cấu hình: 0:3, 1:2, 2:1 và 3:0, tương ứng với số đoạn hỗ trợ và đoạn gây nhiễu. Các đoạn được lấy mẫu một lần cho mỗi câu hỏi với seed 42 và được giữ cố định khi thay đổi hệ số trộn. Cả hai phép thăm dò đều khảo sát 33 giá trị \(d=i/32\), với \(i=0,\ldots,32\).
>
> Đối với H1, mô hình được chạy ở chế độ suy luận để đo LM Loss trên các token đích tại từng giá trị \(d\). Đối với H2, các tham số mô hình được đóng băng và gradient của LM Loss theo hệ số \(d\) được ghi nhận thông qua hook, không thực hiện cập nhật tham số (learning rate bằng 0). Kết quả được tổng hợp trên 1.024 mẫu đối với các cấu hình 0:3, 1:2 và 2:1; riêng cấu hình 3:0 có 494 mẫu do điều kiện 2 bước không đủ ba đoạn hỗ trợ. Vì vậy, khi so sánh kết quả giữa các cấu hình, cần lưu ý sự khác biệt về thành phần mẫu và lượng thông tin hỗ trợ được cung cấp.

**Codex instructions:** Replace the existing Section 5.1.1 paragraph with the three approved paragraphs **verbatim**, adjusting only LaTeX syntax and typography. Retain `Attention Mixing Probe`, the manual mixing coefficient `d`, seed 42, the 33-point grid, the four support:distractor configurations, and the reported 1,024/494 sample counts. Verify experimental artifacts and code for these values and for H1 inference-mode LM Loss versus H2 gradient hook, frozen parameters, and zero learning rate. Flag any mismatch for user review; do not silently alter approved wording. Distinguish manually specified probe `d` from Decision Head output in full ART.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 16 — Section 5.1.2: simplify the system baseline comparison

**Location:** Section **5.1.2 — Đối chứng hệ thống**.

**Problem:** Original text reads like a code audit and overemphasizes ID-order checks, retrieval-title matching, historical index pinning, prompt internals, answer-extraction implementation, and exact normalization steps. The academic report should foreground datasets, comparable systems, evaluation metrics, and the important fine-tuned-versus-base-checkpoint limitation.

**Approved replacement (VERBATIM; adapt only LaTeX formatting):**

> Để đánh giá hiệu quả của hệ thống ART, chúng tôi tiến hành thực nghiệm trên ba bộ dữ liệu MuSiQue, HotpotQA và 2WikiMultiHopQA, mỗi bộ gồm 500 mẫu. Ba phương pháp được đánh giá trên cùng tập câu hỏi và đáp án tham chiếu nhằm bảo đảm tính nhất quán trong quá trình so sánh.
>
> Các phương pháp đối chứng bao gồm:
>
> - **No-RAG:** Mô hình sinh câu trả lời trực tiếp từ câu hỏi mà không sử dụng tài liệu truy hồi.
> - **Vanilla RAG:** Mô hình truy hồi một lần dựa trên câu hỏi ban đầu, sử dụng 5 đoạn văn liên quan để hỗ trợ sinh câu trả lời.
> - **ART:** Hệ thống quyết định thời điểm truy hồi trong quá trình sinh, với 5 đoạn văn được lấy ở mỗi lượt truy hồi.
>
> Cả ba phương pháp đều sử dụng kiến trúc nền Qwen2.5-3B-Instruct. Tuy nhiên, No-RAG và Vanilla RAG sử dụng checkpoint gốc, trong khi ART sử dụng checkpoint đã fine-tune. Vì vậy, kết quả so sánh phản ánh hiệu quả tổng thể của hệ thống ART, bao gồm cả quá trình fine-tuning và cơ chế truy hồi thích nghi, thay vì chỉ đo lường riêng tác động của việc lựa chọn thời điểm truy hồi.
>
> Hiệu quả trả lời câu hỏi được đánh giá thông qua ba chỉ số Accuracy (Acc), Exact Match (EM) và token-level F1, sử dụng quy tắc chuẩn hóa và so khớp đáp án theo bộ đánh giá R-Search. Kết quả được tính trung bình trên toàn bộ mẫu và biểu diễn dưới dạng phần trăm.

**Codex instructions:** Replace the original Section 5.1.2 narrative with the approved text verbatim, converting the three baseline bullets into appropriate LaTeX itemization. The user edited a writing-block copy by accidentally appending `oke` to the end of the first paragraph; that conversational approval is NOT part of the approved report prose and must not be included. Verify the three datasets, 500 examples each, baseline retrieval count, checkpoint differences, and evaluator behavior against code/artifacts; flag discrepancies rather than silently rewriting approved claims. In particular, `Acc` may be a substring/alias hit metric, so avoid implying conventional classification accuracy beyond the actual evaluator definition. Technical audit details (seed, prompts, index provenance, answer parsing, exact normalization) can be moved to reproducibility documentation or an appendix when materially necessary, not casually erased from research records. Keep the methodological limitation about different checkpoints explicit.

**Mode:** Approved exact prose replacement, with verification before application.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt.

## Issue 17 — Section 5.2.1 (H1): independently interpret the Attention Mixing plot

**Location:** Section **5.2.1 — Kiểm chứng H1 và H2**, specifically the H1 discussion of Figure 5.1.

**Problem:** Existing H1 discussion mostly contrasts endpoint LM Loss values at d=0 and d=1, overlooking the shapes of all four curves, their turning points, and how support:distractor composition relates to their behavior.

**Mode: CONCEPTUAL GUIDANCE, NOT VERBATIM REPLACEMENT.** Codex must inspect the actual Figure 5.1 and, where available, the underlying experimental results, then write its OWN concise, rigorous Vietnamese academic interpretation. Do NOT copy the assistant's previously proposed H1 paragraphs.

**User's intended analytical directions (hypotheses to check against the plot, not predetermined conclusions):**
- Consider all four curves across the entire d grid, including their initial decreasing regions, minima, and eventual increases; qualify the magnitude of any changes (especially the shallow 0:3 decrease).
- Investigate whether increasing the number of golden/supporting passages shifts the turning point/minimum toward larger d values (the user's phrase 'score d bắt đầu tăng muộn hơn' refers to the LM Loss curve turning upward at later d, NOT to the decision score itself increasing later).
- Compare the steepness of the LM Loss decreases and the depth of minima as supporting passages increase, noting the 2:1 and 3:0 conditions.
- Explain implications for the role of document evidence and the Attention Mixing coefficient without asserting causal isolation beyond the experimental design.
- Avoid inferring free-generation answer quality from teacher-forced target-token LM Loss; consider unequal sample compositions and the 3:0 missing 2-hop condition.
- Use numbers only after verifying them against the actual plotted data/artifacts; avoid reducing the discussion to endpoint values.

**Status:** APPROVED by user on 2026-11-22 as guidance for Codex to independently read the graph; pending later consolidated Codex prompt. Do not modify the report yet.

## Issue 18 — Section 5.2.2: concise interpretation of retrieval count and F1 sensitivity

**Location:** Section **5.2.2 — Chất lượng hỏi đáp và tải truy hồi**.

**Problem:** The existing discussion mixes code/logging audit details, verbose definitions, and threshold-by-threshold enumeration with the substantive experimental interpretation. The user wants a focused, readable results discussion, not a code review or a forced Pareto/trade-off narrative.

**Mode: CONCEPTUAL GUIDANCE, NOT VERBATIM REPLACEMENT.** Codex should independently inspect the relevant tables, figures, and experiment outputs, and rewrite the discussion in concise formal Vietnamese, supported by the observed data.

**Approved focus:**
1. Describe how **retrieval count** (the number of recorded retrieval queries per completed sample, including repeats, as implemented) changes with the decision threshold **τ**. Check the claimed decreasing trend against the actual experiments.
2. Describe how **F1** varies over those configurations. Emphasize that its observed variation is **not necessarily monotonic** with retrieval count; fewer retrievals do not necessarily reduce F1, but do not guarantee improved F1 either. Compare the three benchmarks only as supported by the evidence.
3. Keep the meaning and limitation of retrieval count succinct: it measures retrieval usage / query count, **not directly measured latency, FLOPs, or total compute cost**.
4. Remove or relocate incidental logging implementation notes, overly detailed threshold listings, and audit language from the main results discussion. Preserve any material caveat required for scientific accuracy.
5. **Do not force a trade-off or Pareto-front analysis.** Omit the non-dominance definition and Pareto enumeration unless essential to a figure/table that genuinely requires them; the user prefers straightforward empirical trends.
6. **Do not add a discussion of why the common threshold τ=0.4297 was selected**, and do not center this subsection on distinguishing the default threshold from benchmark-specific test-set maxima. This topic was explicitly removed from Issue 18 at the user's request.
7. Use the user's comments as direction for what to analyze, **not prescribed findings**. Verify any specific numerical claims against the experimental artifacts and do not imply causal relationships not established by the data.

**Status:** APPROVED by user on 2026-11-22; pending later consolidated Codex prompt. Do not modify the report yet.

## Issue 19 — Section 5.3.1: rewrite MuSiQue comparison example

**Location:** Section **5.3.1 — Ví dụ đối sánh trên MuSiQue**, replacing the paragraph beginning “Hai đối chứng có EM bằng 0; ART có EM bằng 1...”.

**Mode: USER-APPROVED VERBATIM REPLACEMENT.** Replace with the following exact Vietnamese text (only adapt LaTeX markup/typography, without changing meaning):

> Trong ví dụ này, cả No-RAG và Vanilla RAG đều không đưa ra đáp án chính xác (EM = 0), trong khi ART trả lời đúng “Steve Irwin” (EM = 1).
>
> Điểm khác biệt nằm ở khả năng truy hồi dựa trên thông tin được xác định trong quá trình suy luận. Ở lượt truy hồi đầu tiên, ART thu thập bằng chứng cho thấy Bindi Irwin là diễn viên đảm nhận vai Nim trong bộ phim *Return to Nim's Island*. Sau khi xác định được thực thể trung gian này, hệ thống tiếp tục thực hiện truy hồi với truy vấn chứa tên Bindi Irwin, qua đó thu được bằng chứng xác định Steve Irwin là cha của cô.
>
> Ví dụ minh họa lợi ích của cơ chế truy hồi thích nghi trong bài toán hỏi đáp nhiều bước: thay vì chỉ truy hồi dựa trên câu hỏi ban đầu, ART có thể sử dụng thông tin vừa tìm được để xây dựng truy vấn tiếp theo, từ đó bổ sung bằng chứng cần thiết để hoàn thành chuỗi suy luận.

**Verification and caveats:** Before applying, cross-check the original MuSiQue trace, retrieved passages, EM outputs and Appendix A. In particular, do not claim that the retrieved passage titled “My Steve” itself explicitly establishes the father–daughter relationship if it does not; the direct evidence is reportedly in other passages from the same retrieval round. Preserve complete trace/evidence in Appendix A. If any approved statement is contradicted by the raw artifacts, flag it for user review rather than silently inventing or modifying evidence. The main text should describe the two-hop retrieval progression, not narrate raw-output auditing.

**Status:** APPROVED by user on 2026-11-22; pending consolidated Codex prompt. Do not modify the report yet.

## Issue 20 — Section 5.3.2: rewrite mechanism comparison and evidence limitations

**Location:** Section **5.3.2 — So sánh cơ chế**.

**Mode: USER-APPROVED VERBATIM REPLACEMENT.** Replace the current subsection prose with the following approved Vietnamese text; adapt only LaTeX formatting and citation placement, without changing its meaning:

> Các phương pháp IRCoT, Self-RAG và R-Search tiếp cận bài toán truy hồi thích nghi theo những cơ chế khác nhau. IRCoT kết hợp truy hồi với các bước suy luận trung gian; Self-RAG sử dụng các reflection tokens được học từ tín hiệu giám sát để điều khiển truy hồi; trong khi R-Search tối ưu quá trình tìm kiếm thông qua tín hiệu phần thưởng.
>
> Khác với các phương pháp trên, ART kết hợp LM Loss với Retrieval Regularization trong cùng mục tiêu huấn luyện. Thông qua Attention Mixing, tín hiệu gradient từ quá trình dự đoán token có thể tác động gián tiếp đến Decision Head, giúp mô hình học cách điều chỉnh quyết định truy hồi mà không cần nhãn trực tiếp cho từng thời điểm truy hồi hoặc tối ưu chính sách bằng Reinforcement Learning. Tuy nhiên, ART vẫn sử dụng dữ liệu đích do teacher model tạo ra và quá trình Collect–Replay để tổ chức huấn luyện.
>
> Kết quả thực nghiệm cho thấy hiệu quả của hệ thống ART hoàn chỉnh trên các bộ dữ liệu được khảo sát. Tuy nhiên, do ART sử dụng checkpoint đã fine-tune, trong khi các baseline sử dụng checkpoint gốc, kết quả hiện tại chưa tách biệt được đóng góp riêng của cơ chế lựa chọn thời điểm truy hồi. Để đánh giá riêng tác động này, cần thực hiện thêm các thí nghiệm ablation với checkpoint và điều kiện đánh giá tương thích.

**Verification and presentation:** Check descriptions of IRCoT, Self-RAG and R-Search against the cited papers; preserve appropriate citations. Verify ART's LM Loss, retrieval regularization, gradient path, teacher supervision and Collect–Replay against the project implementation. Do not imply that ART was evaluated head-to-head against IRCoT, Self-RAG or R-Search under a shared controlled protocol. Avoid unrelated workload-count appendix remarks and code-audit details in this mechanism discussion. If approved wording conflicts with verifiable sources, flag the conflict instead of silently inventing support.

**Status:** APPROVED by user on 2026-11-22; pending consolidated Codex prompt. Do not modify the report yet.

## Issue 21 — Chapter 6: rewrite conclusion, limitations, and future directions

**Location:** Sections **6.1 Kết luận**, **6.2 Hạn chế**, and **6.3 Hướng phát triển**.

**Mode: USER-APPROVED VERBATIM REPLACEMENT.** Replace the existing prose in these three sections with the following exact approved Vietnamese text. Adjust only LaTeX markup, typography, section heading syntax, and citations as necessary; preserve the substance and paragraph order.

### 6.1 Kết luận

Đồ án nghiên cứu phương pháp học thời điểm truy hồi thông tin thích nghi (Adaptive Retrieval Timing) thông qua việc kết hợp LM Loss và Retrieval Regularization, không yêu cầu nhãn giám sát trực tiếp cho từng quyết định truy hồi hoặc sử dụng Reinforcement Learning.

Thông qua hai phép thăm dò H1 và H2, nghiên cứu khảo sát ảnh hưởng của Attention Mixing đến khả năng dự đoán token và tín hiệu gradient theo hệ số trộn. Trên cơ sở đó, hệ thống ART được xây dựng với ba thành phần chính: Decision Head, Attention Mixing khả vi và quy trình huấn luyện Collect–Replay.

Kết quả thực nghiệm trên MuSiQue, HotpotQA và 2WikiMultiHopQA cho thấy hệ thống ART cải thiện các chỉ số Acc, EM và F1 so với No-RAG và Vanilla RAG trong thiết lập đánh giá được khảo sát. Đồng thời, các thí nghiệm thay đổi ngưỡng quyết định cho thấy số lần truy hồi có thể được điều chỉnh, trong khi chất lượng trả lời không biến thiên đơn điệu theo số lần truy hồi.

Nhìn chung, kết quả bước đầu cho thấy tính khả thi của việc sử dụng tín hiệu từ mục tiêu mô hình ngôn ngữ để học cơ chế quyết định truy hồi, mở ra khả năng nghiên cứu sâu hơn về truy hồi thích nghi trong các hệ thống hỏi đáp nhiều bước.

### 6.2 Hạn chế

Mặc dù đạt được những kết quả bước đầu, nghiên cứu vẫn tồn tại một số hạn chế. Thứ nhất, các phép thăm dò H1 và H2 sử dụng tài liệu được cung cấp sẵn và đánh giá dưới chế độ teacher forcing, do đó chưa phản ánh đầy đủ hành vi của hệ thống trong quá trình sinh tự hồi quy.

Thứ hai, ART sử dụng checkpoint đã fine-tune, trong khi các phương pháp đối chứng sử dụng checkpoint gốc. Vì vậy, kết quả hiện tại chưa cho phép tách biệt hoàn toàn đóng góp của cơ chế lựa chọn thời điểm truy hồi khỏi ảnh hưởng của quá trình fine-tuning và các khác biệt trong thiết lập đánh giá.

Cuối cùng, nghiên cứu chủ yếu sử dụng số lần truy hồi để phản ánh mức độ sử dụng công cụ tìm kiếm, chưa đánh giá trực tiếp chi phí tính toán và độ trễ thực tế của toàn bộ hệ thống.

### 6.3 Hướng phát triển

Trong các nghiên cứu tiếp theo, hệ thống có thể được đánh giá thông qua các thí nghiệm ablation nhằm xác định rõ hơn đóng góp của từng thành phần, đồng thời bổ sung các phép đo chi phí tính toán và độ trễ thực tế.

Đối với giai đoạn Đồ án tốt nghiệp, hai hướng nghiên cứu tiềm năng được xem xét. Hướng thứ nhất là mở rộng cơ chế quyết định từ thời điểm truy hồi sang lựa chọn công cụ (Tool Selection) dựa trên trạng thái suy luận của mô hình. Hướng thứ hai là khảo sát khả năng huấn luyện phối hợp Retriever–Generator, nhằm nghiên cứu tác động của tín hiệu từ mô hình sinh đối với quá trình lựa chọn và truy hồi bằng chứng.

Hai hướng này hiện được xem là những định hướng nghiên cứu cần tiếp tục đánh giá tính khả thi, thay vì các cải tiến đã được xác nhận về hiệu quả.

**Verification and presentation:** Confirm the metric claims and experimental scope against the evaluation artifacts. Maintain the distinction between manually set mixing coefficients in H1/H2 and learned Decision Head behavior in ART. Do not claim isolated causal attribution to retrieval timing, direct timing labels, reinforcement-learning optimization, or established feasibility of the proposed future work. Preserve academic tone and existing section structure. Flag source conflicts for user review instead of silently changing approved content.

**Status:** APPROVED by user on 2026-11-22; pending consolidated Codex prompt. Do not modify the report yet.

---

## Issue 22 — Tài liệu tham khảo: loại nguồn ít liên quan và ưu tiên liên kết venue chính thức

**Scope:** Danh mục Tài liệu tham khảo và các trích dẫn liên quan trong toàn bộ nguồn LaTeX.

**Approved changes:**

- Remove current reference [3], Jeff Johnson et al., *Billion-scale Similarity Search with GPUs* (FAISS).
- Remove current reference [7], Liang Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5).
- Keep current references [1] Self-RAG; [2] 2WikiMultiHopQA; [4] foundational RAG; [5] MuSiQue; [6] IRCoT; [8] Qwen2.5 Technical Report; [9] HotpotQA; [10] R-Search.
- Rebuild and automatically renumber the resulting eight references; verify all `\\cite{...}` commands, bibliography keys, cross-references, and compiled PDF. Do not remove FAISS or E5 from descriptions of the actual implementation merely because their bibliographic entries are removed.
- **User's additional explicit requirement:** Prefer the canonical **published venue/publisher page**, not arXiv, for each retained reference. Examples: OpenReview for ICLR Self-RAG; ACL Anthology for COLING, TACL, ACL, EMNLP, Findings papers; the official NeurIPS proceedings page for the original RAG paper. Confirm exact official URLs and publication metadata rather than inventing links or venues.
- **Exception to resolve carefully:** *Qwen2.5 Technical Report* currently has an arXiv citation; a formally published venue version has not been verified. Do not fabricate a venue. Search for a verified official Qwen team publication/technical-report page as a non-arXiv source; if none can be verified, flag this exception to the user for a decision rather than silently retaining an arXiv URL or substituting a different paper.
- Retain existing bibliography metadata where accurate; correct verified metadata only. Ensure all links are usable and citation numbers correspond to the final order.

**Status:** APPROVED by user on 2026-11-22, including venue-first URL requirement; pending consolidated Codex prompt. Do not modify the report yet.

### Issue 22 — User-approved amendment: remove Qwen2.5 report; supply canonical URLs (2026-11-22)

**This amendment supersedes the earlier eight-reference count and the Qwen2.5 exception in Issue 22.** The user explicitly requested removing *Qwen2.5 Technical Report* (original [8]) as well. The final bibliography must contain **seven** retained sources, in the existing relative order, with the following publisher/venue URLs (no arXiv links):

| New reference number (subject to bibliography ordering) | Original number | Source | Canonical publication URL |
|---|---|---|---|
| [1] | [1] | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (ICLR 2024) | https://openreview.net/forum?id=hSyW5go0v8 |
| [2] | [2] | Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps (COLING 2020) | https://aclanthology.org/2020.coling-main.580/ |
| [3] | [4] | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020) | https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html |
| [4] | [5] | MuSiQue: Multihop Questions via Single-hop Question Composition (TACL 2022) | https://aclanthology.org/2022.tacl-1.31/ |
| [5] | [6] | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (ACL 2023) | https://aclanthology.org/2023.acl-long.557/ |
| [6] | [9] | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (EMNLP 2018) | https://aclanthology.org/D18-1259/ |
| [7] | [10] | R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning (Findings of ACL 2026) | https://aclanthology.org/2026.findings-acl.1896/ |

**Implementation instructions for Codex:** Remove original references [3] FAISS, [7] E5, and [8] Qwen2.5 Technical Report from the bibliography. Preserve accurate technical descriptions of FAISS, E5, and Qwen2.5 in the body even though their citations are removed. For remaining citations, use the URLs above and official BibTeX metadata from the venue pages; preserve stable citation keys where practical. Rebuild the PDF and check for dangling `\\cite` commands, numbering consistency, URLs, and bibliography formatting. Do not edit the report until the user issues the consolidated Codex implementation request. OpenReview may challenge automated browsers; the original official URL is still the intended link, and Codex should not substitute arXiv.

**Status:** APPROVED by user on 2026-11-22. The report itself remains unchanged.

---

## Issue 23 — Thay case study cũ bằng MuSiQue ID 348 (đã duyệt)

**Scope:** §1.2, §5.3.1, Phụ lục A.1/A.2, `report/dacn/generate_case.py`, và các tệp LaTeX sinh từ script (`graphics/case-*.tex`); rà soát mọi tham chiếu tới ví dụ cũ.

**Lý do:** Case cũ “Who is the father of Nim's actress in Return to Nim's Island?” (Bindi Irwin → Steve Irwin) có vấn đề về tính nhất quán giữa reasoning, tài liệu ART viện dẫn và bằng chứng trực tiếp. Không nên sử dụng một đáp án EM=1 để mặc nhiên khẳng định chuỗi suy luận có căn cứ.

**Quyết định đã duyệt:** Thay bằng **MuSiQue sample ID 348** theo báo cáo điều tra `docs/ART_CASE_STUDY_INVESTIGATION.md` (cần đối chiếu lại với JSONL gốc khi triển khai):

- Question: “Who did the actor who played Michael Knight play on Young and Restless?”
- Ground truth: `Snapper Foster`.
- No-RAG: `Jesse James Byron` (EM=0).
- Vanilla RAG: `Peter Bergman` (EM=0).
- ART: `Snapper Foster` (EM=1).
- **Retrieval 1:** Truy vấn xác định diễn viên đóng Michael Knight; tài liệu *Michael Knight (Knight Rider)* xác nhận **David Hasselhoff**.
- **Retrieval 2:** Truy vấn sau đưa **David Hasselhoff** vào truy vấn; tài liệu *Snapper Foster* xác nhận Hasselhoff đóng vai **Snapper Foster** trong *The Young and the Restless*.
- Vanilla RAG có bằng chứng cho quan hệ đầu tiên nhưng thiếu đoạn xác nhận quan hệ thứ hai trong top-5 được ghi nhận. Đây là minh họa hành vi truy hồi thích nghi, **không phải bằng chứng nhân quả riêng của retrieval timing** khi checkpoint và thiết lập đánh giá chưa được kiểm soát.

**Yêu cầu triển khai đã được người dùng cho phép thực hiện ngay (ngoại lệ so với các issue khác):**

1. Đọc báo cáo điều tra và các JSONL nguồn của No-RAG, Vanilla RAG và ART; dùng output và retrieval trace thực tế, không tự dựng hoặc chỉnh sửa dữ liệu thực nghiệm.
2. Cập nhật §1.2, §5.3.1, Phụ lục A.1/A.2 và `report/dacn/generate_case.py`; tái sinh `graphics/case-*.tex` tương ứng. Rà soát mọi chỗ còn nhắc Nim/Bindi Irwin/Steve Irwin.
3. Phụ lục A.1 phải thể hiện đúng các raw outputs (kể cả reasoning nếu có); Phụ lục A.2 giữ nguyên văn cả hai truy vấn và **đủ 5 tài liệu mỗi lượt theo thứ tự gốc**. Nếu rút gọn nội dung từng tài liệu, đánh dấu `...` minh bạch; không lược bỏ toàn bộ một tài liệu. Đây cũng là yêu cầu đã thống nhất tại Issue 01.
4. Kiểm tra sample ID, câu hỏi, ground truth, ba dự đoán, EM, hai truy vấn và mười passages so với JSONL; kiểm tra chuỗi evidence, trích dẫn, liên kết nội bộ, và biên dịch/kiểm tra các trang PDF bị ảnh hưởng.
5. Chỉ chỉnh các file cần thiết cho case study; không thực hiện các thay đổi khác trong review log; không commit/push nếu chưa được yêu cầu.
6. Báo cáo lại các file đã chỉnh, bằng chứng xác minh, trạng thái build và vấn đề còn tồn tại.

**Nguồn điều tra:** `docs/ART_CASE_STUDY_INVESTIGATION.md`, mục C.1, D, E; ART record `art-backup/demo-3/result/musique/rsearch_musique_art_top5_results.jsonl`, ID 348 (dòng 349); No-RAG và Vanilla RAG tương ứng trong `REVIVE_training/result/baselines/musique/`.

**Status:** APPROVED by user on 2026-11-22. User explicitly authorized Codex to implement this case-study replacement now; the review-log update itself does not edit the original report.

---

## Issue 24 — Cập nhật Phụ lục B.1: Hiện vật và quy trình (đã duyệt)

**Scope:** Phụ lục B.1 — “Hiện vật và quy trình”.

**Lý do:** Câu mô tả GitHub và Hugging Face đều chưa công khai đã lỗi thời. Người dùng xác nhận repository GitHub đã công khai; tài nguyên Hugging Face dự kiến công khai sau. Giữ phân biệt trạng thái hiện tại và kế hoạch tương lai.

**Nội dung thay thế nguyên văn đã duyệt:**

### B.1 Hiện vật và quy trình

Mã nguồn của đồ án được công khai tại GitHub: https://github.com/nguyenphantuanduy/adaptive-retrieval-timing. Các tài nguyên phục vụ thực nghiệm và đánh giá, bao gồm checkpoint ART, dự kiến được công khai trên Hugging Face tại https://huggingface.co/adaptive-retrieval-timing.

Kết quả dự đoán và các chỉ số đánh giá của ART được lưu tại `art-backup/demo-3`, trong khi kết quả của các phương pháp đối chứng được lưu tại `REVIVE_training/result/baselines`. Các bảng, hình minh họa và ví dụ phân tích được tái tạo từ những tệp kết quả đã lưu, theo quy trình mô tả trong `docs/report_execution/REVIEW_2026-11-22.md`.

Một số checkpoint và chỉ mục truy hồi từ các lần thực nghiệm trước chưa được ghi nhận bằng revision bất biến. Do đó, khả năng tái lập chính xác các lần chạy lịch sử vẫn phụ thuộc vào tính sẵn có của những hiện vật tương ứng.

**Hướng dẫn Codex:** Thay thế đúng nội dung B.1 bằng đoạn trên; giữ hai URL dưới dạng liên kết có thể truy cập trong PDF (theo quy ước LaTeX của dự án). Không khẳng định tài nguyên Hugging Face đã public; chỉ thay đổi trạng thái này khi người dùng xác nhận. Không sửa các phần khác chỉ vì cập nhật B.1. Nếu đường dẫn tài liệu quy trình thay đổi trong repo, kiểm tra trước khi sửa và báo lại.

**Status:** APPROVED by user on 2026-11-22. Chỉ cập nhật review log; chưa chỉnh báo cáo gốc.
