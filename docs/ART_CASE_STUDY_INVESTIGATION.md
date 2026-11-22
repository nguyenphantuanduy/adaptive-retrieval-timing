# Điều tra mẫu thay thế cho case study ART

**Khuyến nghị để tác giả duyệt: MuSiQue ID 348.** Hai lượt truy hồi tạo thành chuỗi kiểm tra được: Michael Knight → David Hasselhoff → Snapper Foster. Cả hai tài liệu ART viện dẫn đều xuất hiện đúng lượt và hỗ trợ phát biểu tương ứng. Vanilla RAG có mắt xích đầu tiên nhưng thiếu đoạn xác nhận vai diễn. Đây là minh họa về hành vi truy hồi quan sát được, chưa phải phép đo hiệu ứng nhân quả của retrieval timing.

## A. Nguồn dữ liệu và cách đối sánh

- Đọc chín tệp dự đoán: với mỗi `{dataset}` trong `musique`, `hotpotqa`, `2wikimultihopqa`, dùng `REVIVE_training/result/baselines/{dataset}/no_rag.jsonl`, `REVIVE_training/result/baselines/{dataset}/vanilla_rag_top5.jsonl` và `art-backup/demo-3/result/{dataset}/rsearch_{dataset}_art_top5_results.jsonl`. ART là cấu hình mặc định \(\tau=0.4297\), top-5, theo [bảng kê benchmark](BENCHMARK_ANALYSIS.md). Các lượt sweep không tham gia sàng lọc.
- Mỗi JSONL có `id`, `question`, `golden_answers` và raw output `pure_art_answer`. Ghép theo cặp (dataset, ID), kiểm tra ID duy nhất trong từng tệp và so sánh nguyên văn câu hỏi cùng toàn bộ danh sách đáp án chuẩn. Tính số dòng nguồn sau khi ghép; vị trí dòng không dùng làm khóa.
- Chấm EM từng mẫu bằng `compute_exact` (đổi 0/100 thành 0/1). Hai baseline dùng `clean_prediction` để bỏ token điều khiển; ART dùng `extract_answer` để lấy đáp án từ cặp thẻ hoàn chỉnh cuối, hoặc dùng fallback của evaluator. Cả ba phương án đề xuất đều có thẻ đáp án hợp lệ. Bảng bên dưới giữ raw output baseline và tách đáp án trích xuất ART khỏi các đoạn reasoning. [Baseline scorer](../REVIVE_training/evaluate_baseline.py) · [ART scorer](../REVIVE_training/evaluate_art_pure.py) · [EM](../REVIVE_training/utils.py)
- `pure_art_queries` chứa truy vấn theo thứ tự, `pure_art_passages` chứa các đoạn dạng chuỗi nối phẳng, và `pure_art_retrieval_scores` ghi điểm ứng viên. Không ghép điểm với truy vấn theo vị trí: số điểm và số truy vấn khác nhau. Mã generation lưu trace, và mã mô hình nối danh sách passages sau mỗi lần tìm. Với các mẫu được đề xuất, đúng năm đoạn mỗi truy vấn; bảng nhóm danh sách theo các khối năm liên tiếp. Log không lưu document ID trong từng chuỗi ART, mốc token, hay xác nhận độc lập từng lần chèn. Các tham chiếu ART dùng tiêu đề và vị trí 1-based trong mảng phẳng. Vanilla còn có `retrieved_passages` với `rank`, `title`, `document`, `score`, `id` và `index`. [Generation](../REVIVE_training/evaluate_art_pure_generation_top5.py) · [Ghi passages trong model](../shared/ART_Models/ART_REVIVE/model.py#L1376)
- Kiểm tra schema và tìm theo câu hỏi nguyên văn trong `shared/MuSiQue/data/musique_rsearch/`: bản R-Search test xác nhận ID/câu hỏi/đáp án của 348 và 261 nhưng không mang supporting-document labels. Hai bản `musique_ans_v1.0_test.jsonl` và `musique_full_v1.0_test.jsonl` có mảng `paragraphs`, nhưng không khớp nguyên văn các câu đề xuất. Không nối ID giữa hai định dạng. Đã định vị các loader HotpotQA và 2Wiki; chưa xác lập mapping supporting-fact chính thức cho các bản ghi đánh giá. Vì vậy đánh giá bên dưới dựa vào văn bản lưu trong trace, không gọi những đoạn này là gold supporting documents. [Loader HotpotQA](../shared/HotpotQA_Distractor/rsearch_hotpotqa_test_dataset.py) · [Loader 2Wiki](../shared/TwoWikiMultiHopQA/rsearch_2wikimultihopqa_test_dataset.py)
- Vị trí hiện tại của case cũ: `report/dacn/generate_case.py`, ba `graphics/case-*.tex`, Chương 1.2, Chương 5.3.1 và Phụ lục A.1/A.2. Các tệp đó chỉ được đọc.

## B. Thống kê tìm kiếm

| Bộ dữ liệu | Số bản ghi mỗi hệ thống | Câu ghép hợp lệ | EM (0, 0, 1) | Trace dùng được để sàng lọc | Qua bộ lọc ưu tiên |
|---|---:|---:|---:|---:|---:|
| MuSiQue | 500 | 500 | 92 | 92 | 28 |
| HotpotQA | 500 | 500 | 80 | 80 | 22 |
| 2WikiMultiHopQA | 500 | 500 | 89 | 89 | 30 |
| **Tổng** | **1.500** | **1.500** | **261** | **261** | **80** |

Đã đọc 4.500 bản ghi cho 1.500 câu. Trace được coi là dùng được nếu có truy vấn và các chuỗi không rỗng với đúng năm đoạn mỗi truy vấn. Bộ lọc ưu tiên chọn 2–3 lượt, thẻ đáp án hợp lệ và ít nhất một alias chuẩn hóa xuất hiện theo ranh giới từ trong đoạn thuộc lượt sau, nhưng không xuất hiện trong lượt đầu ART hay năm đoạn Vanilla RAG. Đây là bộ lọc ưu tiên thủ công, có thể bỏ sót bằng chứng diễn đạt khác câu hoặc alias không xuất hiện sát chữ. Trong 80 mẫu qua lọc, tôi xem câu hỏi, đầu ra và tiêu đề của 10 mẫu thay thế và đối chiếu thêm mẫu đang dùng; **6 mẫu** được đối chiếu đến câu bằng chứng để quyết định, gồm **3 mẫu mạnh** và **3 mẫu loại khỏi đề xuất**. Các mẫu còn lại chưa được kết luận về bằng chứng.

| Mẫu loại sau kiểm chứng | Lý do |
|---|---|
| MuSiQue 27 | Lượt sau có đoạn xác nhận quan hệ cha–con. Tuy nhiên ART viện dẫn *My Steve*, đoạn lưu chỉ xác nhận quan hệ vợ chồng giữa Terri và Steve Irwin. Đoạn trực tiếp hỗ trợ nằm ở các kết quả khác cùng lượt. |
| MuSiQue 299 | Trace lập luận từ mệnh đề “part of the Parramatta River catchment” thành “flows into the Parramatta River”. Phát biểu về lưu vực trong đoạn mà ART viện dẫn chưa đủ để kết luận dòng chảy trực tiếp. |
| MuSiQue 228 | ART gán thông tin cho một observation có tiêu đề *Secret City Records*. Kết quả thực sự mang tiêu đề *Wooden Arms* và nói về hãng phát hành album đó. Điều này chỉ hỗ trợ một quan hệ phát hành cụ thể và yếu hơn câu hỏi về hãng của nghệ sĩ. |

## C. Ba mẫu mạnh nhất

Tất cả các trích đoạn dưới đây là chuỗi con nguyên văn từ các đoạn đã lưu. Các tiêu đề liệt kê theo thứ tự retrieval. Cột EM dùng đầu ra đã xử lý bằng scorer nêu trên.

### 1. MuSiQue ID 348 — Vai diễn của David Hasselhoff

**Câu hỏi:** “Who did the actor who played Michael Knight play on Young and Restless?”  
**Đáp án chuẩn:** `Snapper Foster`.

| Hệ thống | Đầu ra hoặc đáp án trích xuất | EM |
|---|---|---:|
| No-RAG | `Jesse James Byron<\|im_end\|>` | 0 |
| Vanilla RAG | `Peter Bergman<\|im_end\|>` | 0 |
| ART | `Snapper Foster` trong `<answer>` | 1 |

| Lượt | Truy vấn nguyên văn | Năm tiêu đề theo thứ tự |
|---|---|---|
| 1 | To answer this question, we first need to identify the actor who played Michael Knight. | Michael Knight (Knight Rider); Michael Raymond-James; Michel Gill; Christopher Knight (actor); Michael Murphy (actor) |
| 2 | Next, we need to determine who David Hasselhoff played on the television series "Young and Restless." | Snapper Foster; Michael Knight (Knight Rider); Victor Newman; J. T. Hellstrom; Jerry Douglas (actor) |

- **Lượt 1, hạng 1, mảng phẳng #1:** “Michael Knight is a fictional character and the protagonist of the 1980s television series "Knight Rider", played by David Hasselhoff.” Đoạn cũng nhắc Larry Anderson đóng Michael Long ở cảnh đầu pilot. Chi tiết này không phủ nhận quan hệ mà ART sử dụng: diễn viên đóng nhân vật chính Michael Knight trong series là David Hasselhoff. Lượt 2 thêm tên này, vốn không có trong câu hỏi.
- **Lượt 2, hạng 1, mảng phẳng #6:** “Snapper Foster is a fictional character on the CBS daytime soap opera The Young and the Restless.” Câu tiếp theo xác nhận vai này do William Gray Espy đóng trước, “and David Hasselhoff from 1975 to May 1982.” Không có truy vấn tiếp theo; ART kết luận Snapper Foster.
- **Vanilla RAG:** `Michael Baldwin`, `Michael Knight (Knight Rider)`, `Robert Adamson (actor)`, `Michael Raymond-James`, `Peter Bergman`. Đoạn hạng 2 (ID `111830`) chứa quan hệ Michael Knight → David Hasselhoff. Các đoạn còn lại không cung cấp quan hệ David Hasselhoff → Snapper Foster. Đoạn hạng 1 nói về một nhân vật khác trong cùng soap opera.
- **Thẩm định:** Hai quan hệ đều có câu trực tiếp trong tài liệu và được ART nêu trong raw reasoning, theo đúng tên tài liệu. ART rút gọn “to May 1982” thành “to 1982”; điều đó không đổi vai diễn. Vai có nhiều diễn viên qua các thời kỳ; tên trong truy vấn bước hai xác định đúng người, nhờ đó câu trả lời không dựa vào suy luận rằng chỉ có một diễn viên. Hai lượt và hai đoạn hạng 1 giúp trình bày gọn.
- **Provenance:** Dòng 349, ID 348 trong [No-RAG](../REVIVE_training/result/baselines/musique/no_rag.jsonl#L349), [Vanilla RAG](../REVIVE_training/result/baselines/musique/vanilla_rag_top5.jsonl#L349) và [ART](../art-backup/demo-3/result/musique/rsearch_musique_art_top5_results.jsonl#L349). Raw reasoning và answer tag nằm trong `pure_art_answer`; hai đoạn chứng cứ là `pure_art_passages[0]` và `[5]` theo chỉ số JSON 0-based.

### 2. HotpotQA ID 292 — Ai phong tước hiệu “patricia” cho Marozia?

**Câu hỏi:** “Who gave the mother of Alberic II of Spoleto the title "patricia" of Rome?”  
**Đáp án chuẩn:** `Pope John X`.

| Hệ thống | Đầu ra hoặc đáp án trích xuất | EM |
|---|---|---:|
| No-RAG | `Pope Leo IV<\|im_end\|>` | 0 |
| Vanilla RAG | `Marozia<\|im_end\|>` | 0 |
| ART | `Pope John X` trong `<answer>` | 1 |

| Lượt | Truy vấn nguyên văn | Năm tiêu đề theo thứ tự |
|---|---|---|
| 1 | To answer this question, we first need to identify the mother of Alberic II of Spoleto. | Alberic II of Spoleto; Alberic I of Spoleto; Bertila of Spoleto; Macrina the Elder; Theobald II of Spoleto |
| 2 | Next, we need to determine who gave Marozia the title "patricia" of Rome. | Marozia; Patricia of Naples; Statilia Messalina; Basilina; Patricia Clementina |

- **Lượt 1, hạng 1, #1:** “Alberic II (912–954) was ruler of Rome from 932 to 954, after deposing his mother Marozia and his stepfather, King Hugh of Italy.” Suy luận ghi trong output dùng quan hệ mẹ–con; lượt 2 thêm tên Marozia.
- **Lượt 2, hạng 1, #6:** Đoạn *Marozia* ghi “was given the unprecedented titles "senatrix" ("senatoress") and "patricia" of Rome by Pope John X.” Các kết quả có từ *Patricia* chứa thực thể khác, không đổi quan hệ trong đoạn đích. Không có truy vấn tiếp theo.
- **Vanilla RAG:** `Alberic II of Spoleto`, `Patricia of Naples`, `Alberic I of Spoleto`, `Patricia Clementina`, `Macrina the Elder`. Kết quả hạng 1 (ID `1682476`) cung cấp tên Marozia nhưng không cung cấp tên người phong tước. Output Vanilla trả đúng thực thể trung gian, sai loại thực thể câu hỏi yêu cầu.
- **Thẩm định:** Hai tài liệu khớp phần giải thích của ART, chứng minh được chuỗi quan hệ từ chính văn bản đã truy hồi. Điểm mạnh là lỗi baseline dừng đúng ở mắt xích trung gian. Điểm yếu về biên tập: đổi case từ MuSiQue sang HotpotQA, cần chuẩn hóa lại các nhãn và pipeline tạo ví dụ trong báo cáo.
- **Provenance:** Dòng 293, ID 292 trong [No-RAG](../REVIVE_training/result/baselines/hotpotqa/no_rag.jsonl#L293), [Vanilla RAG](../REVIVE_training/result/baselines/hotpotqa/vanilla_rag_top5.jsonl#L293) và [ART](../art-backup/demo-3/result/hotpotqa/rsearch_hotpotqa_art_top5_results.jsonl#L293). Đoạn chứng cứ là `pure_art_passages[0]` và `[5]`; raw reasoning là `pure_art_answer`.

### 3. MuSiQue ID 261 — Từ Psycho Donuts đến Santa Clara County

**Câu hỏi:** “The city that Psycho Donuts is located in is in which county?”  
**Đáp án chuẩn:** `Santa Clara County`.

| Hệ thống | Đầu ra hoặc đáp án trích xuất | EM |
|---|---|---:|
| No-RAG | `Maricopa<\|im_end\|>` | 0 |
| Vanilla RAG | `Stanislaus County<\|im_end\|>` | 0 |
| ART | `Santa Clara County` trong `<answer>` | 1 |

| Lượt | Truy vấn nguyên văn | Năm tiêu đề theo thứ tự |
|---|---|---|
| 1 | The city that Psycho Donuts is located in is in which county? | Psycho Donuts; Colo, Iowa; Huxley, Iowa; Cuba, Missouri; Newman, California |
| 2 | To answer this question, we first need to determine the city where Psycho Donuts is located. | Psycho Donuts; Top Pot Doughnuts; Uncle Maddio's Pizza Joint; Czech Stop and Little Czech Bakery; Häagen-Dazs |
| 3 | Next, we need to identify the county in which Campbell, California is located. | Campbell, California; Tylers Corner, California; California; Maltby, California; Central Coast (California) |

- **Lượt 1, hạng 1, #1:** “Psycho Donuts is a donut shop in Campbell, California opened in March 2009.” Đoạn cho thực thể Campbell. Lượt 2 vẫn truy vấn về Psycho Donuts, chưa đưa Campbell vào.
- **Lượt 2, hạng 1, #6:** Đoạn *Psycho Donuts* lặp nguyên câu về Campbell. Lượt 3 mới dùng `Campbell, California`; bằng chứng đầu tiên đã có ở lượt 1 nên không ghi công khám phá mới cho lượt 2.
- **Lượt 3, hạng 1, #11:** Đoạn *Campbell, California* ghi “Campbell is a city in Santa Clara County, California, and part of Silicon Valley, in the San Francisco Bay Area.” Raw reasoning dùng quan hệ này và kết luận Santa Clara County.
- **Vanilla RAG:** Năm tiêu đề cùng thứ tự với lượt 1 ART; hạng 1 (ID `113536`) xác nhận Campbell. Không có đoạn xác nhận county cần tìm. So sánh này minh họa rõ khác biệt về bằng chứng sau truy vấn chứa thực thể trung gian, đồng thời cho thấy ART có thể truy hồi dư.
- **Thẩm định:** Chuỗi hai quan hệ trực tiếp và khớp nguồn được nêu trong reasoning. Có ba lượt cho hai mắt xích và lượt 2 lặp bằng chứng. Đây là phương án dự phòng phù hợp nếu báo cáo muốn minh họa cả mức tăng thông tin lẫn một lượt thừa.
- **Provenance:** Dòng 262, ID 261 trong [No-RAG](../REVIVE_training/result/baselines/musique/no_rag.jsonl#L262), [Vanilla RAG](../REVIVE_training/result/baselines/musique/vanilla_rag_top5.jsonl#L262) và [ART](../art-backup/demo-3/result/musique/rsearch_musique_art_top5_results.jsonl#L262). Đoạn chứng cứ ở `pure_art_passages[0]`, `[5]` và `[10]`; raw reasoning là `pure_art_answer`.

## D. Xếp hạng và khuyến nghị

| Hạng | Mẫu | Bằng chứng và tính thích ứng quan sát được | Điểm cần nêu |
|---:|---|---|---|
| 1 | MuSiQue 348 | Hai lượt, hai đoạn hạng 1; lượt 2 thêm David Hasselhoff và thu được đoạn xác nhận đúng vai. | Có hơn một diễn viên đóng Snapper Foster; câu chứng cứ chỉ rõ Hasselhoff. |
| 2 | HotpotQA 292 | Hai lượt; Vanilla trả thực thể trung gian, ART dùng thực thể ấy để lấy đoạn có người phong tước. | Phải đổi bộ dữ liệu của ví dụ xuyên suốt báo cáo nếu được duyệt. |
| 3 | MuSiQue 261 | Top-5 ban đầu có Campbell ở cả hai hệ thống; truy vấn chứa Campbell đưa vào chứng cứ về county. | Lượt 2 lặp đoạn ở lượt 1; bước có thêm thông tin là lượt 3. |

**Đề nghị ID 348.** Mẫu hiện tại cũng có hai lượt và tài liệu phù hợp, nhưng phải giải thích vì sao tài liệu được model viện dẫn không trực tiếp xác nhận quan hệ. ID 348 tránh bất nhất này: tên hai đoạn trong raw output trùng với tài liệu đã truy hồi và hai mệnh đề đúng theo văn bản, với phần chuyển từ lượt đầu sang lượt sau nhìn thấy trực tiếp trong truy vấn. Trong cả ba mẫu, lợi thế quan sát được là bằng chứng bổ sung xuất hiện ở lượt có thực thể trung gian. Có cơ sở xem đó là một lời giải thích hợp lý cho khác biệt đầu ra; tác động nhân quả cần đối chứng cùng checkpoint, prompt và thiết lập retrieval được kiểm soát.

## E. Thay đổi dự kiến nếu tác giả duyệt

1. **Mục 1.2:** Thay khung ví dụ baseline bằng câu hỏi, đáp án chuẩn, hai raw outputs và đoạn Vanilla hạng 2. Phân tích riêng bên dưới rằng đoạn này xác định diễn viên nhưng thiếu vai diễn đích. Giữ phần ART cho Chương 5.
2. **Mục 5.3.1:** Thay bằng ba raw outputs cùng ID, EM và kết luận ngắn về hai mắt xích. Dẫn Phụ lục A cho trace.
3. **Phụ lục A.1 và A.2:** Cập nhật đầu ra gốc, hai truy vấn, thứ tự top-5 và các đoạn hỗ trợ. Đánh dấu phần lược bỏ. Ghi rõ dữ kiện về nhiều diễn viên và thời kỳ nếu dùng đoạn đầy đủ.
4. **Tái lập:** Cập nhật ID và tiêu chí cắt đoạn trong `report/dacn/generate_case.py`; tái sinh `case-baseline.tex`, `case-outputs.tex`, `case-trace.tex` và kiểm tra PDF, caption, liên kết nội bộ sau khi được duyệt.

Điều tra dừng ở đề xuất. Báo cáo LaTeX, review log, dữ liệu và kết quả đánh giá được giữ nguyên.
