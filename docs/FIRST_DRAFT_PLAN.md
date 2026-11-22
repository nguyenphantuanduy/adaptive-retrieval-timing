# First draft plan for the Specialized Project report

**Quyết định đã duyệt cho kế hoạch.** Version 1 is a full Vietnamese academic report for Đồ án chuyên ngành (DACN), the registered Phase 1. The six chapters follow the supplied HCMUT template. The registered Phase 2 is Đồ án tốt nghiệp (DATN), reserved for future work. Use the exact Vietnamese title on the cover and retain the registered English title in the plan and any required bilingual metadata. Lead with adaptive retrieval timing, H1/H2 and the paired local system benchmark. Use the simplified H2 notebook for the selected operating threshold and all H2 figures. Keep published benchmarks and training workload secondary. The approval gate for report drafting remains in place. Source: [author-approved decisions](FIRST_DRAFT_AUTHOR_DECISIONS_UPDATED.md), §§1–12.

## 1. Source inventory and reading log

Authority follows the question being resolved. The supplied template controls structure and formatting. The registration controls the original scope and titles. Current code, saved metrics and the revised H2 notebook control implementation and outcome claims. The dated presentations document the research evolution.

| Source | Inspection and role | Limits |
|---|---|---|
| [Danh sách các nhóm Đồ Án Chuyên Ngành HK261.xlsx](<../Danh sách các nhóm Đồ Án Chuyên Ngành HK261.xlsx>) | Opened workbook OOXML directly, verified sheet names and sheet relationship, read row 1 headers and row 9 across A:X, checked merge ranges and neighboring IDs. Exact registered text is transcribed below. | No edits, workbook resave or screenshot inference. |
| `slides-template/Specialized-Project-Template-VI/README.txt`, `main.tex`, `hcmut-thesis.cls` | Read driver, class options/layout and compiler instructions. | Actual compilation and any additional department submission rules are outside this review. |
| `slides-template/Specialized-Project-Template-VI/chapters/front/{declaration,acknowledgements,abstract,workload}.tex`, `chapters/main/{01-introduction,02-background,03-methodology,04-implementation,05-evaluation,06-conclusion}.tex`, `chapters/back/{appendix-a,appendix-b}.tex`, `refs/references.bib` | Read front matter, chapter/section scaffolds, example formats, appendix placeholders and sample bibliography. The logo is referenced at `graphics/hcmut.png`. | Example names, references, formulas and section prompts are placeholders. |
| `slides-template/DACN_29_9_Group8/main.tex` and `slides/{0_introduction,1_existing_approaches,2_solution,3_hypothesis,4_teacher_gen,5_hypo1,6_hypo2,7_issue,8_references}.tex` | Traced inclusion order, frame content and references with targeted reads of method, theory and planning text. Inspected `images/overview_model.png` visually. | Probe examples illustrate design. They are not aggregate results. |
| `slides-template/DACN_27_10_Group8/main.tex` and `slides/{0_introduction,1_hypo_1,2_hypo_2,3_analysis,4_demo_model,5_next_week,6_references}.tex`, `diagrams/attention_mixing.tex` | Traced probe pipelines, demo configs and benchmark table. Inspected `images/ART Retrieval Workflow Diagram.png`. | Use the dated slides for research chronology and workflow provenance. |
| `slides-template/DACN_10_11_Group8/main.tex` and `slides/{0_introduction,1_failure_case_analysis,2_dataset_analysis,3_datasetv2_gen,4_demov3,5_problem,6_references}.tex`, `diagrams/attention_mixing.tex` | Traced the failure cases, dataset changes, demo-3 table, rollout estimate and proposed next work. Inspected copied workflow image and verified the October/November raster asset hashes match. The attention-mixing diagram inclusion is commented out in November. | Examples are qualitative and some carried assets were unused. |
| All three `slides-template/DACN_*/thesiscite.bib` and `main-blx.bib` | Empty hand-authored bibliography and generated biblatex control file, respectively. The actual slide references are embedded in the reference frame sources above. Inventoried all `images/` assets, including shared paper illustrations and decorative background PDFs. | Decorative PDF backgrounds needed no rendering. Paper figures were traced to slide references and require attribution before reuse. |
| [BENCHMARK_ANALYSIS.md](BENCHMARK_ANALYSIS.md), [retrieval cost JSON](../REVIVE_training/metrics/retrieval_cost/retrieval_cost_summary.json), [per-sample query CSV](../REVIVE_training/metrics/retrieval_cost/retrieval_cost_per_sample.csv) | Read the entire benchmark report including M01–M23 and query tradeoffs. Parsed the 11 cost summary entries and checked every joined run against its source metric JSON and sample denominator. The CSV is the reproducibility companion. | Ordered digests support within-ART sweep alignment. A direct audit of all nine default/local prediction files verifies cross-method identity and reproduces the scored aggregates. |
| [HYPOTHESIS_VERIFICATION.md](HYPOTHESIS_VERIFICATION.md), H1/H2 source and saved artifacts | Verified the simplified notebook's saved source and outputs and checked the probe driver and pooling. | Use the simplified notebook as the sole source for H2 threshold interpretation and every H2 figure. |
| [METHODOLOGY.md](METHODOLOGY.md), [TRAINING.md](TRAINING.md), [REVIVE_ARCHITECTURE.md](REVIVE_ARCHITECTURE.md), [REVIVE_TRAINING_EVALUATION.md](REVIVE_TRAINING_EVALUATION.md), [EXPERIMENTS.md](EXPERIMENTS.md), [REPORT_PREPARATION.md](REPORT_PREPARATION.md), [WORKFLOWS.md](WORKFLOWS.md), [RAG_BASELINE_IMPLEMENTATION.md](RAG_BASELINE_IMPLEMENTATION.md), [R_SEARCH_COMPARISON.md](R_SEARCH_COMPARISON.md), [R_SEARCH_BENCHMARK_REFERENCE.md](R_SEARCH_BENCHMARK_REFERENCE.md) | Used for method, experiment provenance and baseline comparison. Selected claims checked against `shared/ART_Models/ART_REVIVE/{config.json,model.py,attention_mixing.py,cache_utils.py,retrieval_context_processor.py}` and `REVIVE_training/{training.py,training_v2.py,trainer.py,evaluate_art_pure_generation_top5.py,evaluate_art_pure.py,baseline_generation.py,evaluate_baseline.py,utils.py,analyze_retrieval_cost.py}`. | Current files cannot establish every historical remote revision. External paper and figure metadata should be checked before the bibliography is finalized. |
| [Author decisions](FIRST_DRAFT_AUTHOR_DECISIONS_UPDATED.md), [planning instructions](FIRST_DRAFT_PLANNING_INSTRUCTIONS.md), and this plan's earlier version | Reconciled the approved DACN/DATN boundary, H2 handling, narrative priorities, metadata and per-dataset benchmark selection with the original source audit. | Approval covers planning. |
| [R-Search ACL paper](https://aclanthology.org/2026.findings-acl.1896/), [proceedings PDF](https://aclanthology.org/2026.findings-acl.1896.pdf), [arXiv v1](https://arxiv.org/html/2506.04185v1), [author repository](https://github.com/QingFei1/R-Search) | Checked ACL Table 1, §§3.3–4.3, Appendix A/B, the earlier arXiv table and the public GRPO/PPO launchers and rollout batch logic. | Paper limits, public examples and actual training logs carry different provenance. |

Dates are verified in each main file's date declaration at line 89. The reading sequence was 29 September, 27 October, then 10 November **2026**. No report was written or compiled, and no experiment was run.

## 2. Official project registration

The root workbook contains worksheets `Overall`, `Trang tính1`, **`Trang tính2`** and `Cluster 1` through `Cluster 6`. On `Trang tính2` cell **A9** is numeric **8**, row 9 is the target, and no merged range intersects it. B9 is `KHMT`. A8 is 7 and A10 is 10, so the row was located using its actual identifier.

- **C9, Vietnamese.** Thời điểm truy hồi thích ứng: Học quyết định truy hồi mà không cần giám sát trực tiếp hay học tăng cường
- **D9, English.** Adaptive Retrieval Timing: Learning When to Retrieve without Direct Decision Supervision or Reinforcement Learning
- **M9–P9.** The co-supervisor field is blank. N9 records student ID 2310491 and P9 records Nguyễn Phan Tuấn Duy. O9 contains an administrative Zalo contact, which is unnecessary for the report plan. The sheet header does not provide a primary-supervisor field. All three slide covers identify **PGS. TS. Quản Thành Thơ**, the verified form for report metadata. The author supplies class **MT23KHM2**. Use it in the student metadata where the template permits. Q9–X9 contain no values.
- **Expected outputs.** Row 9 has no standalone deliverables field. The descriptions, objectives and tasks call for a fixed-RAG baseline, feasibility probes, a latent adaptive retriever-timing implementation, comparative measurements and ablations, then a Phase 2 extension and evaluation for adaptive tool use and joint retriever–LLM training. These are intended work products inferred from the task wording. The verbatim source is in the planning appendix below. K9 is blank and L9 lists seven references.

## 3. Template compliance matrix

All paths in the next two tables are relative to `slides-template/Specialized-Project-Template-VI/`. Preserve the driver ordering. “Required” means present in the supplied driver or class, or explicitly instructed by the supplied README. The repository contains no independently verified extra departmental rules.

| Template location | Requirement or example | Planned report placement |
|---|---|---|
| `main.tex`, `hcmut-thesis.cls` | Vietnamese `[vietnamese,final]`, `memoir` 12 pt, A4, oneside, Vietnamese UTF-8/T5, `mathptmx`, 1.5 line spacing. Margins left 3 cm, right/top/bottom 2 cm with header/footer included. Numbering and TOC depth through subsection. | Keep class and preamble. Populate verified metadata and leave the submission date unset pending the official date. |
| `main.tex` cover fields and class `coverpage` | Institutional cover, report type, title, major, supervisor, student name and ID. Council, reviewer, report time and optional department override are optional per README. Sample names/major need confirmation or replacement. | Registered Vietnamese title, Khoa học Máy tính (KHMT), Nguyễn Phan Tuấn Duy, MSSV 2310491, class MT23KHM2 where appropriate, and PGS. TS. Quản Thành Thơ. Use supplied logo. |
| `chapters/front/declaration.tex`, `acknowledgements.tex`, `abstract.tex`, `workload.tex` | Included in this order. Declaration and contribution table are scaffolds awaiting author content. Abstract comment suggests 200–350 words and keywords. | Preserve the four blocks. Adapt declaration and workload to the confirmed roster and actual effort. |
| Front matter in `main.tex` | TOC, list of figures and list of tables after the four prefaces. | Generate from report structure and captions. |
| `chapters/main/01-introduction.tex` through `06-conclusion.tex` | Six included chapters in the supplied order. Section names give the scaffold. README explicitly removes the Introduction formatting example. | Use the chapter map below. Add subsections under the existing headings. |
| `main.tex`, `refs/references.bib` | BibTeX `plain` with superscript, sorted, square numeric `natbib`; bibliography precedes appendices. README removes three TeX example entries. | Curate project references and cite original figures. Use `\cite`, stable labels and `\Cref` across text. |
| Main driver and formatting example | Equation, table and figure numbering reset by chapter. `amsmath`, `booktabs`, and `cleveref` in the preamble. The sample shows figure caption below and table caption above. | Place figure assets in `graphics/`, use descriptive captions with data provenance and keep equation symbols consistent. Treat example float content as a guide. |
| `chapters/back/appendix-a.tex`, `appendix-b.tex` | Both included after bibliography. Headings are supplemental results and source/reproducibility. Repository URL is a placeholder. | Use Appendix A for secondary analysis and Appendix B for an approved URL, artifact index and run manifests or their documented gaps. |
| `README.txt` | pdfLaTeX and BibTeX or `latexmk -pdf main.tex`. English class option exists. | Use the approved Vietnamese option. Compile only after separate authorization. |

## 4. Research narrative and objectives

**Approved report title.** Use C9 exactly on the cover and preserve D9 for bilingual metadata. Introduce **Adaptive Retrieval Timing (ART)** as the research approach and **REVIVE** as the principal implementation evaluated here. Keep the terminology “retrieval timing”, “information value”, “retrieval cost” and “autoregressive generation”, with explicit operational definitions.

**Central question.** In multi-hop QA, can a decoder learn when to query an external retriever from teacher-continuation likelihood and a score penalty, without direct retrieval-decision labels or RL, and what quality/query-workload tradeoffs appear at inference?

- **RQ1, feasibility.** How does a forced attention mixing coefficient affect teacher-target loss under different supporting-to-distractor document ratios? H1 addresses target likelihood and attention sensitivity. Language coherence and free-generation answer accuracy remain open for this probe.
- **RQ2, trainability.** Does the differentiated LM objective provide a useful local signal with supplied documents? H2 measures gradients on a forced coefficient. The observed signal is conditional and does not establish calibration or pre-retrieval policy behavior on its own.
- **RQ3, system outcomes.** How does the REVIVE system compare descriptively with local No-RAG and single-retrieval Vanilla RAG, and how do quality and query counts change across saved thresholds?
- **Defensible contributions.** A differentiable attention-mixing mechanism coupled to a learned decision head, a collect/replay training and autoregressive retrieval implementation, and empirical characterization via H1/H2 probes and a three-dataset benchmark with logged workload. The teacher generation pipeline provides text supervision. The gate is trained indirectly. Hard retrieval and the fixed E5/FAISS lookup remain outside backpropagation.
- **Claim boundaries.** Demonstrating an implemented gate and aggregate quality differences does not show a universal policy, calibrated evidence utility, global threshold optimum, end-to-end latency improvement, or independent novelty over all related work. Reserve Phase 2 for DATN future work. Present missing Phase 1 measurements as specific limitations or prospective checks attached to the claims they affect.

## 5. Chronological slide evolution and corrections

| Milestone | Retain in the narrative | Refine or supersede |
|---|---|---|
| **29/9/2026** | `1_existing_approaches` contrasts parametric LMs, fixed/Vanilla RAG, IRCoT, Self-RAG and R-Search. `2_solution` introduces gated document attention and an information–cost objective. `3_hypothesis`, `5_hypo1`, and `6_hypo2` articulate the hypotheses. `4_teacher_gen` shows the Qwen3-4B teacher approach. `7_issue` records benchmark mismatch and planned verification. | Treat Thomas Paine and Sherlock Holmes examples as teaching illustrations. Use current code and aggregates for confirmed mechanisms and measurements. |
| **27/10/2026** | `1_hypo_1` and `2_hypo_2` turn the hypotheses into a 33-level probe. `3_analysis` records the development of the gradient/cost probe. `4_demo_model` shows collect/replay/inference, demo settings and early MuSiQue results. `5_next_week` proposes failure analysis and query/passage relevance work. | Use the simplified notebook for the H2 interpretation and figures. Demo configurations and external comparators need provenance labels. The scope of local benchmark evidence has since expanded. |
| **10/11/2026** | `1_failure_case_analysis` highlights indirect queries and unsupported reasoning; `2_dataset_analysis` gives a label/evidence concern; `3_datasetv2_gen` introduces four teacher demonstrations, including insufficient evidence, revised rules and Qwen3-8B. `4_demov3` shows matched demo-era training settings, MuSiQue sweep and comparative examples; `5_problem` lists longer runs and cost measurement as pending. | Treat examples as illustrative, use archive-wide benchmarks for headline results and distinguish published reference numbers from local baselines. Keep rollout-count arithmetic out of claims about training cost. |

### Historical corrections and evidence audit

| Historical claim / evidence | Verified or current evidence | Planned treatment | Status |
|---|---|---|---|
| 29/9 `2_solution.tex` and `images/overview_model.png` show an MLP block, local gradient path and segment-based retrieval. | Current REVIVE head is one linear layer plus sigmoid on the last hidden vector. Initial decision runs at prefill, later decisions at completed sentences. The search is discrete. | Redraw and label the hard retrieval boundary, prefill, and train/inference distinction. | Ready to write |
| 29/9 H1/H2 theory and 27/10 `Validation Results` frame labels suggest successful validation. | Forced-coefficient probes show nonmonotonic target loss and conditional gradient directions. H2 gradients describe local score changes with supplied evidence. | Say “supports the narrower likelihood/sensitivity claims” and specify which broader behavioral claims remain unresolved. | Ready to write |
| Simplified H2 notebook | Current gradient and cost panels motivate the selected operational threshold. | Use τ = 0.4297 consistently and use only notebook-derived H2 figures. | Ready to write |
| 27/10 workflow diagram presents exploration and cost simply. | Current collect/replay uses exploration during eligible training steps, no-grad selection and gradient-enabled replay. The loss penalizes accumulated scores, with a terminal edge case. Inference uses a deterministic comparison at candidate steps. | Draw separate collection and replay arrows and label the penalty as a score regularizer. Keep static replay edge cases in limitations. | Ready to write |
| 27/10 demo-1 note says Acc and EM did not change with demonstrations. | Saved matching values show Acc 12.4 to 11.8, F1 18.94 to 19.47 and EM fixed at 11.0. Mapping from the temporary model files to the historical run is not independently pinned. | Report the numerical distinction only if using this historical example, otherwise omit it from the primary comparison. | Needs clarification |
| Slides list both demos at 0.75 epochs. Current `training.py` says 0.1. | Author confirmed both historical runs at 0.75 and the v1 file was edited later. Teacher dataset and validation differ between entry points. | Keep history and current defaults in separate columns. Keep checkpoint effects separate from bundled teacher changes. | Ready to write |
| 10/11 `4_demov3.tex` says “Best configuration” at 0.60 and labels plain rows top-3/default. | Current archive shows dataset-dependent maxima and 3 passages/query for plain MuSiQue. The plain filename does not verify threshold; the slide attests to a configuration but is not an immutable run manifest. Explicit sweep filenames omit top-k, while observed nonzero rows have 5 passages/query. | Use “highest observed for this dataset/sweep”; footnote slide provenance and leave unverified settings unresolved. | Ready to write with caveats |
| October/November MuSiQue comparisons combine local and literature rows. | The saved local No-RAG and Vanilla RAG metrics differ from published R-Search baselines. Saved local rows align exactly on ordered IDs, questions and gold aliases. Shared metric functions score each system's processed output. | Main table uses local comparison only. Put external values in clearly labeled related-work context if approved. | Ready to write |
| 10/11 rollout comparison quotes roughly 6% of R-Search and discusses trainable layer fraction. | Teacher exposure approximation and RL rollout arithmetic represent different work units. Measured training time, FLOPs, compute budget and exact historical training counts are absent. Local layer selection rounds 40% up to 15 of 36 and includes a head. | Keep the short qualitative training-workload discussion in Ch. 5 and document the conditional arithmetic in Appendix B. | Verified as an estimate, actual counts unverified |
| 10/11 demo-v2 and demo-v3 reasoning examples suggest improved handling of indirect queries/insufficient evidence. | Examples motivate data interventions. Saved aggregate results contain no causal attribution to any one v2 prompt change. | Use the verified matched cases below for system illustration. | Ready to write |

### H2 threshold treatment

The [simplified H2 notebook](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/analyze_threshold_and_cost_simplified.ipynb) is the sole authority for threshold interpretation and report-ready H2 figures. Cells 0–8 read and process the referenced gradient CSV and set the operating point. Cells 11 and 13 produce the full gradient and gradient-plus-cost views. Use the probe driver and pooling code only to document how the data were generated.

1. **Probe design.** Force \(d=i/32\) at 33 grid points with question, supplied documents and teacher continuation held fixed within a condition. The recorded gradient is \(g(d)=\partial L_{\rm LM}/\partial d\). A negative gradient gives a local descent direction toward larger \(d\). The pooled 3:0 condition contains 494 examples versus 1,024 in the other pooled conditions because 2-hop 3:0 is unavailable. Interpret gradient signs in light of the ratio and hop strata.
2. **Operating point.** Present \(\tau=0.4297\) as the selected operational hyperparameter. The plotted gradients motivate a neighborhood around \(d=0.40\)–\(0.44\). The notebook illustrates how adding a linear score penalty shifts the local gradient to \(g(d)+\lambda\), for \(\lambda=0,0.05,0.1\). Explain the sign change using the cost panel. Keep \(d\), the forced attention-mixing coefficient in the probe, distinct from \(\tau\), the cutoff applied to the model's predicted decision score during inference. The notebook supplies qualitative motivation only.
3. **Figures and downstream evidence.** Use the notebook's gradient and cost panels and their referenced data for every H2 figure. Label the horizontal axis as forced \(d\) and identify the vertical line as the selected operating threshold shown for reference. Report the saved inference sweep separately as observed quality and logged retrieval workload.

## 6. Template-aligned hierarchical outline and writing plan

Statuses refer to the evidence available for drafting after author approval. Main chapters follow the shipped section structure. Subsections below indicate working hierarchy and depth. P0 covers required first-draft material, P1 supporting detail, P2 optional work.

| Template location and planned hierarchy | Purpose and argument | Evidence and planned depth | Scope / status |
|---|---|---|---|
| Cover and `chapters/front/` | Official cover; declaration; acknowledgements; 200–350-word abstract and keywords; workload. Then TOC and visual lists. | Workbook C9/D9, N9/P9, all slide cover metadata and actual contribution record. Write abstract after results. | P0, **Ready to write** with known metadata. Administrative fields and the factual workload statement can be finalized before submission. |
| **1. Giới thiệu**, `01-introduction.tex` | **1.1 Lý do chọn đề tài** evidence gaps in multi-hop generation. **1.2 Phát biểu bài toán** retrieval timing during a sequence. **1.3 Mục tiêu** three research questions. **1.4 Phạm vi** DACN scope, with DATN reserved for future work. **1.5 Đóng góp** separate mechanism, implementation and descriptive evaluation. **1.6 Cấu trúc báo cáo** chapter guide. | Workbook E9–J9, September introduction/related-work/problem framing and benchmark results. Short motivation and explicit boundary table. | P0, **Ready to write**. DACN/DATN boundary approved. |
| **2. Cơ sở lý thuyết và các nghiên cứu liên quan**, `02-background.tex` | **2.1 Cơ sở lý thuyết** autoregressive LM, causal loss, dense retrieval, multi-hop. **2.2 Các nghiên cứu liên quan** fixed/Vanilla RAG, IRCoT, Self-RAG, R-Search and registered adaptive work relevant to timing. **2.3 Tổng kết** literature gap and testable question. | 29/9 `1_existing_approaches.tex`, workbook L9, `phase_1/stage_0_literature/{ircot,self-rag,r-search}.md`, R-Search comparison. One mechanism/design table. Verify entries before submission. | P0/P1, **Ready to write** on local framing. Verify any remaining source metadata during bibliography preparation. |
| **3. Phương pháp đề xuất**, `03-methodology.tex` | **3.1 Tổng quan hướng tiếp cận** decoder, gate and retriever roles. **3.2 Dữ liệu** H1/H2 validation, final Qwen3-8B teacher data and three held-out test sets with role/split boundaries. **3.3 Phương pháp** decision score, differentiable mixed attention, hypotheses and observable predictions, teacher objective and penalty, collect/replay, exploration and inference. | `METHODOLOGY.md`, `TRAINING.md`, November dataset slides, teacher-prompt code and `ART_REVIVE` sources. Moderate technical depth with about four core equations and an overview diagram. | P0, **Ready to write**. Exact filtered historical counts **Need clarification**. |
| **4. Thiết kế và hiện thực hệ thống**, `04-implementation.tex` | **4.1 Yêu cầu hệ thống** model/retriever runtime and artifacts. **4.2 Kiến trúc hệ thống** data preparation, Qwen, E5/FAISS, cache and evaluation. **4.3 Hiện thực** trace the two training stages and inference candidate/search/insertion loop, then logging and scoring. | `REVIVE_ARCHITECTURE.md`, `REVIVE_TRAINING_EVALUATION.md`, `WORKFLOWS.md`, October workflow asset, `trainer.py:164` and `model.py:1178,1757`. Moderate depth, two pseudocode blocks and one cache/flow diagram. | P0, **Ready to write** for current code. Historical environment details **Need clarification**. |
| **5. Thực nghiệm và đánh giá**, `05-evaluation.tex` | **5.1 Thiết lập thực nghiệm** protocol matrix separating H1/H2, archival demos, local baselines and external literature. **5.2 Kết quả** H1/H2, local No-RAG/Vanilla RAG/REVIVE at default, threshold sensitivity and query tradeoffs, then a compact published EM/F1 context table with dataset-specific post-hoc best-F1 selection. **5.3 Thảo luận** causal limits, optional verified traces and a brief training-workload estimate with measured-cost caveats. | H1 summaries, simplified H2 notebook, saved matched cases, `BENCHMARK_ANALYSIS.md` M01–M23 and query summary. Core of draft. Include comparison, threshold matrix, two probe plots and F1/query figure. | P0, **Ready to write** for verified evidence. Use only the conditional workload estimate. Controlled causal or significance claims **Need more evidence/experiments**. |
| **6. Kết luận và hướng phát triển**, `06-conclusion.tex` | **6.1 Kết luận** one finding per RQ. **6.2 Hạn chế** evidence and implementation gaps. **6.3 Hướng phát triển** prioritize Phase 1 follow-up measurements and describe tool selection and joint retriever training as later DATN work. | Synthesize Ch. 5. Keep outcomes and proposed next research in separate subsections. | P0, **Ready to write** under the approved DACN/DATN framing. |
| Bibliography and **Appendix A – Kết quả bổ sung**, `appendix-a.tex` | Full traceable metric/cost tables, per-hop summaries and optional verified qualitative cases. | Archive IDs, query CSV/JSON, H1/H2 saved summaries and optional verified qualitative trace. | P1, **Ready to write** for verified aggregates. Full demo chronology **Defer beyond first draft**. |
| **Appendix B – Mã nguồn và khả năng tái lập**, `appendix-b.tex` | Repository references, checkpoint ledger, scoring/analysis commands, artifact paths and required environment. | `docs/README.md`, `TRAINING.md`, baseline implementation and retrieval-cost scripts; replace sample URL with approved repository URL. | P1, **Needs clarification** for publication URL and immutable revisions. |

## 7. Alignment with the registered project

| Registered objective or task | Completed / partial / planned evidence | Draft map and approval boundary |
|---|---|---|
| Phase 1. Literature and fixed retrieval baseline; assess retriever | Background notes and actual Qwen No-RAG/Vanilla RAG comparisons. E5/FAISS implemented. No isolated evidence-recall/utility study located in reviewed benchmark. | Ch. 2 and 5. Label generation metrics separately from retriever quality. |
| Phase 1. Verify feasibility with and without documents, utility and gradients | H1 teacher-target loss and H2 local gradient sweeps complete. Masked attention paths and observed sensitivity are supported. Hidden-state change, general language stability and free-generation accuracy as isolated outcomes are incomplete. | Ch. 3 and 5. Reserve broader H1 or universal-gradient claims for future evidence. |
| Phase 1. Implement latent timing, differentiable gating and cost | REVIVE implementation and tuned demos verified or author-confirmed. Collect/replay implements local credit propagation through mixing on a fixed trajectory. No dedicated long-horizon credit-assignment validation located. | Ch. 3 and 4. Clarify if the registered credit-assignment task should be called partial. |
| Phase 1. Exploration, policy formation and collapse avoidance | Heuristic exploration in collection, bounded updates and logs. Sweeps show near-universal at-least-one retrieval and changed counts. Confidence-adaptive exploration and independent anti-collapse tests were not established. | Ch. 4 and 6. Report mechanism as implemented and policy interpretation as limited. |
| Phase 1. Multi-hop evaluation, cost and component ablations | Three datasets, baselines, default and threshold variants plus query-count frontier. No full controlled decision/attention/exploration/penalty/credit ablation, resource-cost measurement or timing-interval study found. | Ch. 5 and App. A. Do not treat threshold tuning or teacher-version comparison as component ablations. |
| Phase 2. Generalized adaptive tool selection | Repository has ART-as-tool evaluation wrappers and slide mentions of R-Search-Tool from related work. These are not evidence of a multi-tool policy. | Ch. 1 scope and Ch. 6 future work. Author-approved DATN future work. |
| Phase 2. Joint retriever–LLM training and generalized evaluation | Current dense lookup is external and fixed in the reviewed training/scoring path. No verified joint-training or multi-tool benchmark artifacts located. | Ch. 6 registered next phase. Author-approved DATN future work. |

## 8. Experimental evidence map

| Evidence group | Model, data, scorer, threshold | Result to carry forward | Caveat and source |
|---|---|---|---|
| H1 and H2 | Qwen2.5-3B attention-mixing probe, annotated MuSiQue 2-/3-/4-hop validation. H1 teacher CE, H2 forced derivative. 33 d levels; no deployed threshold intervention. | H1 supports likelihood sensitivity. H2 supports a local, conditional signal. | Different support-ratio populations, unpinned revisions and no matched free-generation outcomes. `HYPOTHESIS_VERIFICATION.md` and probe CSVs. |
| Local baselines | Base Qwen2.5-3B-Instruct, 500 per dataset, No-RAG or one top-5 retrieval. Current scorer removes three exact Qwen control tokens, then scores the remaining completion. | For MuSiQue/HotpotQA/2Wiki respectively, No-RAG F1 5.61/20.19/27.58 and Vanilla F1 10.25/39.61/37.81. | Current scoring reproduces every saved local baseline metric. Each system uses its own generation and output protocol. `BENCHMARK_ANALYSIS.md` M01–M06. |
| Matched archived MuSiQue demos | Author-confirmed demo-2 `training.py`, demo-3 `training_v2.py`, 0.75 epochs each, top-5 historical default. ART final answer-tag extraction or full stripped fallback. | Demo-2 F1 26.33 to demo-3 31.17, EM 17.60 to 21.60. | Teacher changes are bundled. Exact checkpoint and training dataset revisions are unpinned. M17/M12, `TRAINING.md`. |
| Demo-3 default | Fine-tuned ART on 500 per benchmark, historical τ 0.4297, top-5. Same local metric functions but ART extraction differs from baselines. | MuSiQue Acc/F1/EM 25.80/31.17/21.60, HotpotQA 42.20/48.24/37.20, 2Wiki 48.20/46.30/37.00. REVIVE improves all three metrics over both baselines on every dataset. F1 gains in dataset order are +20.92/+8.63/+8.49 points over Vanilla and +25.56/+28.05/+18.72 over No-RAG. | Paired local system comparison on verified identical ordered instances; fine-tuning and adaptive search are bundled. M12/M08/M07 and M06/M04/M02. |
| Saved threshold sweep and retrieval workload | Demo-3, inference-only changes author-confirmed. Scores from existing ART summaries, query counts from `pure_art_queries`. | Best observed F1 32.08 at 0.60 for MuSiQue (2.588 queries vs 4.792 default), 49.68 at 0.50 for HotpotQA (3.166 vs 4.070), and 46.30 at 0.4297 for 2Wiki (4.012). Frontiers: MuSiQue 0.60/0.70, HotpotQA 0.50/0.60, 2Wiki 0.4297/0.50/0.60. | Ordered identity digests match within ART. Calls are logged before retrieval; no per-call success or latency fields. No 0.70 scores for HotpotQA or 2Wiki. M07–M15/M22/M23 and cost summary. |
| External R-Search benchmark | Published RL policy references, separate model/evaluation provenance. | Use published EM/F1 for context from the Findings of ACL 2026 Table 1. The same core values are in arXiv v1 Table 2 under a different dataset order. | Retrieval corpus/index, exact IDs, prompts, scoring and checkpoint alignment differ or remain unknown. `R_SEARCH_BENCHMARK_REFERENCE.md`, `R_SEARCH_COMPARISON.md`. |

### Evaluation hierarchy and score definitions

**Probe evidence for Ch. 5.** Start with the fixed-within-condition design and state the independent variable \(d=i/32\), document ratio and hop strata. Show H1 loss endpoints for pooled 3:0 (1.2875 to 0.8940) and 0:3 (1.3160 to 1.3943), then show the full curves because endpoints omit interior minima. Present H2 sign contrasts at selected grid points and the observations at both extremes, then give the operating-point explanation from the simplified notebook. These measurements are on supplied evidence under teacher forcing. Follow them with the realized inference search trace and aggregate query-frequency evidence as a distinct empirical layer. [H1 aggregate](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/plots/lm_loss_summary.csv) · [H2 aggregate](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_summary.csv)

1. **Primary** — H1/H2 mechanism evidence, then the local No-RAG/Vanilla RAG/REVIVE comparison at the historical default, then the within-REVIVE threshold and query-workload sweep in Ch. 5. Put observed findings next to the relevant probe or benchmark table. Keep the reason for learning a timing signal separate from headline answer quality.
2. **Secondary** — the contextual published comparison below. Keep the same model-size family where possible and label the local REVIVE threshold selection.
3. **Tertiary** — one short paragraph on training-workload differences, with the conditional arithmetic in an appendix. Place the matched qualitative cases after the quantitative evidence.

Use the actual scorer definitions in [utils.py](../REVIVE_training/utils.py). **Acc** is the percentage of samples where a gold alias occurs as a case-insensitive substring of the prediction. **EM** averages best-alias exact equality after lowercasing and removing ASCII punctuation and English articles and collapsing whitespace. **Token-level F1** averages best-alias whitespace-token multiset F1 after the same normalization. ART extracts the last complete answer-tag block, then strips it, or strips the whole output if no block matches. The current baseline scorer removes only the three exact Qwen text-control tokens before applying those metrics to the remaining completion. Score each method with its implemented output protocol and the same metric definitions. [ART scorer](../REVIVE_training/evaluate_art_pure.py#L115) · [Baseline scorer](../REVIVE_training/evaluate_baseline.py#L11)

### Verified controls for the local system comparison

Use a practically controlled system-level comparison on the same local test instances. The three systems use task-appropriate prompts. REVIVE uses the fine-tuned Qwen2.5-3B-Instruct family checkpoint and its standard answer-tag protocol. The baselines use the original Qwen2.5-3B-Instruct checkpoint and score the completion after removal of tokenizer control tokens. Changes in the full system include fine-tuning and retrieval policy, so the benchmark cannot apportion the observed gains to timing alone.

| Control | Evidence and scope |
|---|---|
| Examples | Direct audit of all nine prediction files. Each method has 500 rows per dataset and matches in order on ID, question and full gold-alias list, with 500 unique matched tuples per dataset. |
| Seed and model family | [Baseline generator](../REVIVE_training/baseline_generation.py#L12) and [REVIVE generator](../REVIVE_training/evaluate_art_pure_generation_top5.py#L27) set seed 44. The code and [training provenance](TRAINING.md) identify the common Qwen2.5-3B-Instruct base family and distinct checkpoints. The archives lack per-run seed and checkpoint revision manifests. |
| Retrieval setting | No-RAG performs no search. Vanilla uses the dataset-specific shared E5/FAISS retriever and a single question query with top-5. REVIVE's checked-in top-5 entry point selects the shared 2Wiki adapter; shared MuSiQue/HotpotQA adapters are available. Both saved retrieval methods return batches of five. In a cross-check where first logged query strings match exactly, all five passage strings agree in 388/388 HotpotQA and 303/314 2Wiki rows. All five titles agree in order on 367/367 MuSiQue, 388/388 HotpotQA and 314/314 2Wiki matching-query rows. For MuSiQue, removing a redundant quoted title header and collapsing whitespace gives 366/367 five-passage text matches. These checks support corpus/retriever comparability while leaving historical index revisions and byte-level parity open. |
| Scoring | Recomputed Acc, F1 and EM in memory from all 4,500 archived predictions using the current method-specific preprocessing and shared scorers. All 27 values agree with the published local table at its displayed precision. |

### Matched qualitative examples for Chapter 5 or Appendix A

All cases below use the demo-3 default \(\tau=0.4297\) top-5 files in the benchmark manifest. The baseline predictions shown are after control-token cleaning and the REVIVE predictions come from valid answer tags. Each row has the same dataset, ID, question and gold aliases across methods. The shared normalized best-alias EM gives REVIVE 1 and both baselines 0 in every row. No-RAG logs zero queries and Vanilla logs one question query and five passages in each case.

| Dataset, ID | Question cue and gold | No-RAG / Vanilla RAG | REVIVE answer and trace |
|---|---|---|---|
| MuSiQue, 27 | Father of Nim's actress in *Return to Nim's Island*. Gold: **Steve Irwin**. | François Civil / John Wayne | **Steve Irwin**. Two logged queries, first about the actress and then about Bindi Irwin's father. The second batch includes *My Daddy, the Crocodile Hunter*. |
| HotpotQA, 73 | Birth date of the Argentine footballer Ronald Koeman replaced at Southampton. Gold: **2 March 1972**. | 1977 / 5 October 1971 | **2 March 1972**. Two logged queries, ending with Mauricio Pochettino's birth date. The first passage in the second batch is titled *Mauricio Pochettino* and contains the gold date. |
| 2WikiMultiHopQA, 0 | Father-in-law of Queen Hyojeong. Gold: **Crown Prince Hyomyeong**. | King Gojong / King Heonjong's father, who is also Queen Hyojeong's father-in-law, is King Injo. | **Crown Prince Hyomyeong**. Three logged queries, ending with King Heonjong's father. The third batch starts with *Heonjong of Joseon* and contains the gold name. |

Artifact lookup by dataset and ID:

- MuSiQue 27, JSONL row 28. [No-RAG](../REVIVE_training/result/baselines/musique/no_rag.jsonl#L28) · [Vanilla](../REVIVE_training/result/baselines/musique/vanilla_rag_top5.jsonl#L28) · [REVIVE](../art-backup/demo-3/result/musique/rsearch_musique_art_top5_results.jsonl#L28)
- HotpotQA 73, JSONL row 74. [No-RAG](../REVIVE_training/result/baselines/hotpotqa/no_rag.jsonl#L74) · [Vanilla](../REVIVE_training/result/baselines/hotpotqa/vanilla_rag_top5.jsonl#L74) · [REVIVE](../art-backup/demo-3/result/hotpotqa/rsearch_hotpotqa_art_top5_results.jsonl#L74)
- 2WikiMultiHopQA 0, JSONL row 1. [No-RAG](../REVIVE_training/result/baselines/2wikimultihopqa/no_rag.jsonl#L1) · [Vanilla](../REVIVE_training/result/baselines/2wikimultihopqa/vanilla_rag_top5.jsonl#L1) · [REVIVE](../art-backup/demo-3/result/2wikimultihopqa/rsearch_2wikimultihopqa_art_top5_results.jsonl#L1)

Use a compact table with one or two cases in the main text and the remaining trace in Appendix A if space permits. Read ordered five-passage batches from the flattened logged fields. These selected successes illustrate query reformulation and exposure to relevant evidence. They establish neither prevalence nor causal attribution.

### Post-hoc best-observed F1 for the contextual table

Select the highest observed F1 independently for each dataset among the known-threshold demo-3 runs. All values in a row come from the same configuration. Query averages were checked against the corresponding totals and sample counts in the [retrieval-cost summary](../REVIVE_training/metrics/retrieval_cost/retrieval_cost_summary.json), with its metric filename matched to the original JSON.

| Dataset | τ | N | Acc | F1 | EM | Avg. queries | Logged queries | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MuSiQue | 0.60 | 500 | 26.00 | 32.08 | 22.80 | 2.588 | 1,294 | [M14](../art-backup/demo-3/metrics/musique/threshold60_metrics.json) |
| HotpotQA | 0.50 | 500 | 43.80 | 49.68 | 39.20 | 3.166 | 1,583 | [M09](../art-backup/demo-3/metrics/hotpotqa/threshold50_metrics.json) |
| 2WikiMultiHopQA | 0.4297 | 500 | 48.20 | 46.30 | 37.00 | 4.012 | 2,006 | [M07](../art-backup/demo-3/metrics/2wikimultihopqa/rsearch_2wikimultihopqa_art_top5_metrics.json) |

This is post-hoc selection on the reported test results. It supplies context about the observed sweep rather than performance at an independently validation-tuned threshold. Keep historical τ = 0.4297 on all three default rows in the primary local table. Keep Acc in the internal results, including the provenance table above. The published table below uses only EM/F1, the measures published together.

| Method and provenance | MuSiQue EM / F1 | HotpotQA EM / F1 | 2WikiMultiHopQA EM / F1 |
|---|---:|---:|---:|
| REVIVE local, per-dataset best observed F1 | 22.80 / 32.08 | 39.20 / 49.68 | 37.00 / 46.30 |
| IRCoT, reported by R-Search (3B table) | 9.2 / 17.6 | 39.0 / 50.4 | 35.8 / 46.0 |
| Adaptive-RAG, reported by R-Search (3B table) | 25.4 / 35.6 | 38.0 / 49.0 | 35.0 / 43.9 |
| Search-R1, reported by R-Search (3B table) | 24.4 / 32.9 | 46.2 / 57.8 | 58.8 / 68.1 |
| R-Search, 3B table | 25.8 / 34.8 | 43.4 / 54.4 | 65.0 / 72.6 |

Cite [Findings of ACL 2026, Table 1](https://aclanthology.org/2026.findings-acl.1896.pdf#page=6) as the source of these external rows. The final paper orders columns as 2Wiki, HotpotQA, MuSiQue. The repository's contextual reference uses [arXiv v1 Table 2](https://arxiv.org/html/2506.04185v1#S4.T2), which puts HotpotQA first. Do not mix table numbers or use the R-Search-Tool row as a same-backbone comparator because it uses GLM-4-Plus for final generation. The external paper trains its RL system on 2Wiki, while the reviewed REVIVE training uses MuSiQue teacher data. R-Search reports using 500 test examples from the named released splits, an E5 retriever over dataset-specific Wikipedia corpora, top-5 evaluation and vLLM. Equal sample counts, similar retriever family and retrieval depth do not establish sample IDs, exact corpus/index identity, prompts, extraction, decoding budgets or version parity. Flag selection asymmetry explicitly. [Paper §§4.1–4.3 and Appendix A](https://aclanthology.org/2026.findings-acl.1896.pdf) · [Local comparability audit](R_SEARCH_COMPARISON.md)

### Training workload and the slide's 6% estimate

- **Author-confirmed REVIVE workflow.** One dataset-level autoregressive generation pass over approximately 20,000 examples with a Qwen3-8B teacher, followed by about 0.75 epoch of 3B student training. The nominal calculation is 0.75 × 20,000 ≈ 15,000 teacher-trajectory presentations in student training. The one teacher pass involves autoregressive decoding steps and is a separate workload. The teacher script caps new output at 2,048 tokens and disables tokenizer truncation in its input path. These are current source settings. They give neither average sequence lengths nor total realized tokens. The v2 dry-run cleaner reports 19,938 inputs and 19,350 valid records. It does not certify the historical training manifest or actual packing, repeats, updates or epoch denominator. [Teacher generator](../shared/MuSiQue/musique_training_teacher_generation_v2.py#L92) · [Training record](TRAINING.md)
- **Trainable capacity.** `training_v2.py` selects the last 40% of decoder layers, rounded up to 15 of 36 (indices 21–35), plus the retrieval decision head, after freezing the model. The frozen prefix, embedding and other frozen modules still participate in forward computation. The source verifies a **layer selection rule**, while the exact fraction of trainable parameter elements for the deployed checkpoint is unverified. Do not apply 40% as a compute multiplier. The current script sets a 0.75-epoch target, microbatch 1 and gradient accumulation 32. Training includes collect/replay work beyond counting teacher-trajectory presentations. [Layer rule](../REVIVE_training/training_v2.py#L489) · [Model config](../shared/ART_Models/ART_REVIVE/config.json) · [Collect/replay](REVIVE_TRAINING_EVALUATION.md)
- **Paper versus released launcher.** The Findings paper §4.3 describes Qwen2.5-3B/7B, GRPO/PPO, 2Wiki training, eight A100 80 GB GPUs, a maximum of 195 steps, a batch size of 256, and five responses per input prompt. Appendix A adds evaluation and ablation context. The public trainer builds batches from `data.train_batch_size` and repeats each batch by `rollout.n_agent` before generation. This substantiates interpreting the corresponding public GRPO settings as prompts per batch and responses per prompt. The released **7B GRPO** launcher uses 256 prompts and five agents but caps steps at **120**. The **7B PPO** launcher uses 256 prompts, **one** agent, and **120** steps. These are launch settings, not proof of the executed training count for the published 3B checkpoint. The paper reports hardware but the inspected paper and supplement did not establish a matching measured GPU-hour, FLOP or wall-clock total. R-Search work includes policy rollout with search, answer/evidence/format scoring including a check model, and optimization with reference-policy terms. PPO also uses a critic. [Paper §§3.3–4.3 and Appendix A](https://aclanthology.org/2026.findings-acl.1896.pdf) · [Public trainer](https://github.com/QingFei1/R-Search/blob/main/src/verl/trainer/ppo/ray_trainer.py) · [GRPO launcher](https://github.com/QingFei1/R-Search/blob/main/scripts/train_grpo_7b.sh) · [PPO launcher](https://github.com/QingFei1/R-Search/blob/main/scripts/train_ppo_7b.sh)
- **Reconstruction and decision for the report.** The 10 November slide computes 195 × 256 × 5 = 249,600 rollout presentations and 15,000 / 249,600 ≈ 6.0%. The calculation assumes full 256-prompt batches, all 195 steps and five generated responses per prompt throughout. It is a paper-config scenario, with exact actual 3B logs and checkpoint ancestry unverified. Student replay presentations and sampled RL rollouts are different units of work. Keep the conditional calculation in Appendix B if needed for audit and use a short qualitative statement in Ch. 5: the verified training procedure avoids on-policy multi-response RL optimization for timing, while the workload includes teacher decoding, retrieval collection and replay. No verified total-cost ratio follows. The public 7B launcher mismatch and the unmeasured teacher and rollout costs rule out presenting 6% as realized compute or 94% as a saving.
- **Future measurement.** If quantitative comparison becomes available, align measured GPU-hours, hardware, peak memory, trainable counts, input/generated/observation tokens, retrieval calls, and the full teacher/collector/replay or RL rollout/scoring/update phases. Count generation as autoregressive forward work once. Use actual profiling and run logs with common units. Avoid an invented FLOP decomposition or a token/parameter conversion without a cited and validated method.

## 9. Figures, equations, tables and citations

| Planned asset | Source and action after approval |
|---|---|
| System overview and attention detail (Ch. 3–4) | Redraw September overview and October TikZ/workflow with current code. Show a single linear score head, document/no-document mask and rotary branches, one value mix, latest masked observation, cached prior context and search boundary. Avoid treating a masked branch as a clean no-document rerun. |
| Training and inference flow (Ch. 4) | Adapt October workflow. A separate no-grad collect path with exploration, replay path with score gradient and inference gate. Label timing at prefill/sentence boundaries. |
| Core equations (Ch. 3) | Causal teacher-target loss with ignored context tokens, \(d_t=\sigma(w^\top h_t+b)\), attention probability mix \(d_tA_++(1-d_t)A_-\), replay loss \(\mathcal L_{\rm LM}+\lambda\sum_{j\in J}d_j\) and threshold condition \(d_t\ge \tau\) subject to query guards. Verify indexing and define \(J\) from code rather than calling the penalty a count of executed retrievals. |
| Probe graphics (Ch. 5 / App. A) | Replot H1 from its saved summary. Reproduce both H2 panels exclusively from the simplified notebook and its referenced data after approval. Use the full d grid, ratio legend and denominators, and distinguish the forced coefficient axis from the τ = 0.4297 reference line. |
| Experiment protocol and benchmark tables (Ch. 5) | New tables from source IDs and receipts in `BENCHMARK_ANALYSIS.md`. Distinguish fixed top-k from decision threshold, base model from fine-tuned model, prediction extraction and historical episode counts. |
| Quality/workload scatter (Ch. 5) | Plot F1 against mean logged queries, panel by dataset, annotate τ and highlight nondominated points. Include no fabricated points, no causal trend or uncertainty band without data. |
| Case study (optional App. A) | Use the verified matched cases and source rows above. Show gold, scored predictions, EM and the logged evidence sequence compactly. |
| Citations | Map L9's seven works, slide references, dataset and model sources, and published R-Search metrics to verified BibTeX. Add IRCoT/MuSiQue/HotpotQA/2Wiki, Qwen, E5/FAISS and relevant software/repository versions where used. October and November use inline citation frames, and the template .bib is sample-only. Use the verified ACL proceedings record for the R-Search Table 1; verify the remaining bibliography metadata, including the registered budget-aware preprint. |

## 10. Remaining factual checks and writing roadmap

The author has resolved language, title, roster, DACN/DATN scope, H2 framing, comparison hierarchy and the per-dataset F1 selection rule. None of the checks below blocks planning.

1. **Administrative and reproducibility details.** Confirm submission date and any applicable council/reviewer entries, workload statement and approved repository URL. Include historical checkpoint, dataset and index revisions or hardware specifics only where independently supported, and label missing run metadata in Appendix B.
2. **Training accounting if quantitative workload claims are desired.** Exact filtered teacher/student counts, realized token lengths and presentations, actual 3B R-Search run settings/logs, and comparable hardware-normalized cost measurements remain unavailable. Keep the conditional scenario and qualitative framing.
3. **Run-level reproducibility.** Archived per-run seed and immutable checkpoint/corpus/index revisions remain unpinned. Use code-level controls and the artifact checks above, and record the boundary in Appendix B.

**Draft order after approval.** Establish the approved cover fields and template. Draft Ch. 3 on the mechanism and hypotheses, then Ch. 4 on the verified data pipeline, collect/replay and inference. Draft Ch. 5 in the approved evidence order. Draft Ch. 2 with verified sources, then Ch. 1 and Ch. 6. Finish appendices, references and the abstract. Stop before drafting or compiling until separately approved.

**Draft checklist.** Keep every reported metric tied to its source and run. Verify the local default table and the post-hoc external table use different selection rules. Validate H1/H2 claims against the forced-probe design. Source every H2 figure exclusively from the simplified notebook and its referenced data. Distinguish forced d from the selected operational τ. Carry matched IDs, verified controls and shared metric definitions into the local table. Label the workload proxy and its assumptions. Check that each chapter follows the approved template and gives demonstrated Phase 1 contributions priority over secondary evidence gaps and DATN follow-up.

## Planning appendix — verbatim registration fields

Workbook `Trang tính2`, row 9. The bilingual title strings are reproduced in Section 2. The following are direct cell transcriptions with line structure retained.

### E9 — Mô tả (Tiếng Việt)

> Giai đoạn 1:
>  Đề tài tập trung nghiên cứu bài toán Retrieval-Augmented Generation (RAG) theo hướng học thời điểm truy hồi thích ứng trong quá trình sinh tự hồi quy, với mục tiêu cho phép mô hình ngôn ngữ lớn (LLM) tự hình thành chiến lược truy hồi dựa trên nhu cầu thông tin của nhiệm vụ mà không cần dữ liệu giám sát trực tiếp cho quyết định truy hồi. Các phương pháp hiện tại thường xác định thời điểm truy hồi bằng chiến lược cố định, heuristic hoặc dữ liệu được gán nhãn bởi con người/teacher model; một số hướng gần đây mô hình hóa truy hồi như một hành động tuần tự và sử dụng học tăng cường để tối ưu quyết định. Những cách tiếp cận này có thể yêu cầu xây dựng retrieval-decision supervision, reward hoặc cơ chế tối ưu riêng, đồng thời việc truy hồi không cần thiết làm gia tăng chi phí tính toán và độ dài ngữ cảnh. Đề tài đề xuất một cơ chế latent adaptive retrieval được tích hợp trực tiếp vào quá trình sinh, trong đó mô hình không được huấn luyện để trực tiếp dự đoán quyết định truy hồi đúng hay sai, mà học cách cân bằng giữa giá trị thông tin mà một lần truy hồi mang lại cho quá trình sinh và chi phí phát sinh từ lần truy hồi đó. Quyết định truy hồi vì vậy được hình thành như một hệ quả của quá trình tối ưu giá trị thông tin–chi phí, thay vì là một mục tiêu được giám sát độc lập. Cơ chế differentiable gating cho phép tín hiệu từ mục tiêu sinh truyền ngược đến quyết định và phản ánh mức độ hữu ích của thông tin truy hồi, trong khi cơ chế exploration tạo điều kiện để mô hình khám phá những hành vi truy hồi khác nhau. Qua đó, mô hình hướng tới việc tự học khi nào thông tin hiện tại không còn đủ để tiếp tục nhiệm vụ và khi nào lợi ích của thông tin bổ sung đủ lớn để biện minh cho chi phí truy hồi, đồng thời hạn chế các lần truy hồi không cần thiết. Nghiên cứu tập trung đánh giá khả năng hình thành chiến lược truy hồi thích ứng và multi-hop retrieval mà không cần retrieval-decision supervision hoặc reinforcement learning, đồng thời phân tích sự đánh đổi giữa chất lượng sinh, giá trị thông tin thu được, số lần truy hồi và chi phí truy hồi.
> 
>  Giai đoạn 2:
>  Đề tài mở rộng cơ chế adaptive retrieval sang bài toán tool-augmented language modeling, trong đó truy hồi được xem như một dạng hành động mà LLM có thể chủ động lựa chọn trong quá trình sinh. Trên cơ sở cơ chế tự học thời điểm truy hồi ở Giai đoạn 1, nghiên cứu mở rộng câu hỏi từ “khi nào cần truy hồi?” sang “khi nào cần tương tác với công cụ bên ngoài và nên thực hiện hành động nào?”, hướng tới khả năng để mô hình tự học chiến lược sử dụng nhiều loại công cụ dựa trên nhu cầu của nhiệm vụ thay vì dựa trên lịch sử hành động được gán nhãn trước. Đồng thời, nghiên cứu xem xét cơ chế huấn luyện phối hợp end-to-end giữa LLM và retriever nhằm cải thiện sự tương thích giữa nhu cầu thông tin của mô hình và khả năng truy hồi. Giai đoạn này hướng tới việc khái quát hóa adaptive retrieval thành một cơ chế adaptive tool-use, trong đó mô hình có thể tự xác định thời điểm tương tác với môi trường bên ngoài, lựa chọn hành động phù hợp và cân bằng giá trị của thông tin hoặc khả năng mà công cụ cung cấp với chi phí tương tác.

### F9 — Description (English)

> Phase 1:
>  The project focuses on Retrieval-Augmented Generation (RAG) from the perspective of learning adaptive retrieval timing during autoregressive generation, with the goal of enabling large language models (LLMs) to autonomously develop retrieval strategies based on the information needs of a task without requiring direct supervision for retrieval decisions. Existing approaches typically determine when to retrieve using fixed strategies, heuristics, or retrieval-decision annotations provided by humans or teacher models; some recent approaches model retrieval as a sequential action and use reinforcement learning to optimize retrieval decisions. Such approaches may require retrieval-decision supervision, reward design, or dedicated optimization procedures, while unnecessary retrieval can increase computational costs and context length. This project proposes a latent adaptive retrieval mechanism integrated directly into the generation process, where the model is not trained to directly predict whether a retrieval decision is correct or incorrect, but instead learns to balance the information value that a retrieval provides to the generation process against the cost incurred by that retrieval. The retrieval decision therefore emerges as a consequence of optimizing the information–cost trade-off, rather than being treated as an independently supervised objective. A differentiable gating mechanism allows signals from the generation objective to propagate back to the retrieval decision and reflect the utility of retrieved information, while an exploration mechanism enables the model to explore different retrieval behaviors. Through this mechanism, the model aims to learn when the available information is insufficient to continue solving the task and when the benefit of additional information is large enough to justify the cost of retrieval, while reducing unnecessary retrieval. The study focuses on evaluating the model's ability to develop adaptive retrieval strategies and perform multi-hop retrieval without retrieval-decision supervision or reinforcement learning, while analyzing the trade-offs among generation quality, information value, retrieval frequency, and retrieval cost.
> 
>  Phase 2:
>  The project extends the adaptive retrieval mechanism to tool-augmented language modeling, where retrieval is treated as a form of action that an LLM can actively choose during generation. Building upon the adaptive retrieval timing mechanism developed in Phase 1, the research extends the question from “when should the model retrieve?” to “when should the model interact with an external tool, and which action should it take?”, with the goal of enabling the model to learn strategies for using multiple types of tools based on task requirements rather than relying on pre-annotated action trajectories. In addition, the project investigates end-to-end joint training of the LLM and retriever to improve the alignment between the model's information needs and the retriever's ability to provide relevant information. This phase aims to generalize adaptive retrieval into a broader framework for adaptive tool use, in which the model can autonomously determine when to interact with external resources, select appropriate actions, and balance the value of the information or capability provided by a tool against the cost of interaction.

### G9 — Mục tiêu (Tiếng Việt)

> Giai đoạn 1:
>  - Nghiên cứu và xây dựng cơ chế truy hồi thích ứng cho RAG, cho phép LLM tự học thời điểm thực hiện truy hồi trong quá trình sinh mà không cần dữ liệu giám sát trực tiếp cho quyết định truy hồi hoặc các cơ chế tối ưu dựa trên reinforcement learning.
>  - Nghiên cứu khả năng hình thành chiến lược truy hồi thông qua sự cân bằng giữa giá trị thông tin và chi phí truy hồi, trong đó quyết định truy hồi được hình thành như một hệ quả của quá trình tối ưu mục tiêu sinh và chi phí, thay vì được học như một mục tiêu riêng biệt.
>  - Đánh giá khả năng của cơ chế đề xuất trong việc thực hiện adaptive và multi-hop retrieval, đồng thời phân tích sự đánh đổi giữa chất lượng sinh, giá trị thông tin thu được, số lần truy hồi và chi phí tính toán.
>  - Phân tích hành vi và chiến lược truy hồi được hình thành trong quá trình học, đặc biệt trên các bài toán yêu cầu suy luận nhiều bước, nhằm đánh giá liệu mô hình có thể tự xác định thời điểm thông tin hiện tại không còn đủ và việc truy hồi bổ sung mang lại lợi ích hay không.
> 
>  Giai đoạn 2:
>  - Mở rộng cơ chế adaptive retrieval theo hướng xem tương tác với nguồn thông tin bên ngoài như một hành động có thể được LLM chủ động lựa chọn trong quá trình sinh.
>  - Nghiên cứu khả năng khái quát hóa cơ chế quyết định từ retrieval sang adaptive tool-use, cho phép mô hình tự xác định thời điểm cần sử dụng công cụ bên ngoài và lựa chọn hành động phù hợp với yêu cầu của nhiệm vụ.
>  - Khảo sát khả năng phối hợp và huấn luyện end-to-end giữa LLM và retriever, nhằm cải thiện sự tương thích giữa nhu cầu thông tin của mô hình, quá trình truy hồi và quá trình sinh.

### H9 — Objectives (English)

> Phase 1:
>  - Research and develop an adaptive retrieval mechanism for RAG, enabling LLMs to learn when to perform retrieval during generation without requiring direct supervision for retrieval decisions or reinforcement-learning-based optimization.
>  - Investigate the formation of retrieval strategies through a trade-off between information value and retrieval cost, where the retrieval decision emerges as a consequence of optimizing the generation objective and retrieval cost, rather than being learned as an independent objective.
>  - Evaluate the ability of the proposed mechanism to perform adaptive and multi-hop retrieval, while analyzing the trade-offs among generation quality, information value, retrieval frequency, and computational cost.
>  - Analyze the retrieval behaviors and strategies that emerge during training, particularly on tasks requiring multi-step reasoning, to determine whether the model can autonomously identify when the available information is insufficient and when additional retrieval provides sufficient benefit to justify its cost.
> 
>  Phase 2:
>  - Extend the adaptive retrieval mechanism by treating interaction with external information sources as an action that can be actively selected by the LLM during generation.
>  - Investigate the generalization of the decision mechanism from retrieval to adaptive tool use, enabling the model to autonomously determine when external tools are needed and select appropriate actions based on task requirements.
>  - Investigate joint and end-to-end training of the LLM and retriever, aiming to improve the alignment among the model’s information needs, the retrieval process, and the generation process.

### I9 — Nhiệm vụ (Tiếng Việt)

> Giai đoạn 1
> 
>  1. Khảo sát và xây dựng baseline
> 
>  - Khảo sát các phương pháp Retrieval-Augmented Generation (RAG) và các cơ chế xác định thời điểm truy hồi, bao gồm chiến lược truy hồi cố định, heuristic, adaptive retrieval và các phương pháp sử dụng retrieval-decision supervision hoặc reinforcement learning.
>  - Xây dựng hệ thống RAG baseline với chiến lược truy hồi cố định làm cơ sở so sánh.
>  - Khảo sát và đánh giá retriever trên tập dữ liệu thực nghiệm, xác định khả năng cung cấp các tài liệu liên quan cho các nhiệm vụ yêu cầu suy luận nhiều bước.
>  - Phân tích ảnh hưởng của thời điểm và tần suất truy hồi đến chất lượng sinh, mức độ sử dụng thông tin truy hồi, độ dài ngữ cảnh và chi phí tính toán.
> 
>  2. Phân tích tính khả thi của cơ chế adaptive retrieval
> 
>  - Xây dựng các thực nghiệm ban đầu nhằm kiểm tra liệu thông tin truy hồi có tạo ra sự khác biệt đủ lớn trong biểu diễn của mô hình giữa trạng thái có và không có tài liệu truy hồi hay không.
>  - Đánh giá sự khác biệt giữa các trạng thái có và không có thông tin truy hồi thông qua attention, hidden representation và mục tiêu sinh ngôn ngữ.
>  - Đo lường mức độ cải thiện của mục tiêu sinh khi bổ sung thông tin truy hồi, qua đó ước lượng giá trị thông tin của retrieval trong các trạng thái khác nhau của nhiệm vụ.
>  - Phân tích gradient truyền từ mục tiêu sinh đến biến quyết định retrieval, đặc biệt là độ lớn và phân bố của gradient đối với các trường hợp retrieval có mức độ hữu ích khác nhau.
>  - Kiểm tra ảnh hưởng của các mức độ gating khác nhau đến khả năng sinh của mô hình nhằm đánh giá tính ổn định của cơ chế trộn thông tin.
>  - Dựa trên kết quả thực nghiệm, lựa chọn và điều chỉnh kiến trúc decision module, cơ chế gating và cách biểu diễn trạng thái retrieval phù hợp.
> 
>  3. Xây dựng cơ chế latent adaptive retrieval
> 
>  - Thiết kế cơ chế quyết định thời điểm truy hồi được tích hợp vào quá trình sinh tự hồi quy, cho phép mô hình học nhu cầu truy hồi từ trạng thái hiện tại của nhiệm vụ.
>  - Xây dựng decision module và differentiable gating để điều tiết mức độ sử dụng thông tin từ tài liệu truy hồi trong quá trình sinh.
>  - Nghiên cứu cơ chế huấn luyện trong đó tín hiệu từ mục tiêu sinh được sử dụng để hình thành retrieval decision mà không yêu cầu nhãn trực tiếp cho thời điểm hoặc quyết định retrieval.
>  - Thiết kế thành phần chi phí retrieval nhằm khuyến khích mô hình cân bằng giữa lợi ích của thông tin bổ sung và chi phí phát sinh từ retrieval.
>  - Nghiên cứu cơ chế credit assignment phù hợp với khoảng thời gian mà thông tin được truy hồi tiếp tục đóng góp cho quá trình suy luận.
>  - Đánh giá và điều chỉnh cơ chế nhằm hạn chế các hành vi suy biến như luôn retrieval hoặc không retrieval.
> 
>  4. Exploration và học retrieval timing
> 
>  - Xây dựng cơ chế exploration cho phép mô hình khám phá các hành vi retrieval khác nhau trong quá trình huấn luyện mà không cần retrieval-decision supervision.
>  - Nghiên cứu cách điều chỉnh mức độ exploration dựa trên trạng thái và độ chắc chắn của decision module nhằm cân bằng giữa exploration và chi phí retrieval.
>  - Phân tích hiện tượng premature convergence về chiến lược retrieval hoặc no-retrieval và xây dựng cơ chế hạn chế nếu cần thiết.
>  - Huấn luyện mô hình bằng mục tiêu sinh kết hợp với các thành phần gating, retrieval cost và exploration phù hợp với kết quả của các thực nghiệm khả thi ban đầu.
>  - Phân tích quá trình hình thành retrieval policy và kiểm tra liệu retrieval timing có thể tự hình thành từ quá trình tối ưu generation objective và retrieval cost hay không.
> 
>  5. Thực nghiệm và đánh giá
> 
>  - Đánh giá khả năng học retrieval timing thích ứng trên các nhiệm vụ yêu cầu thông tin bên ngoài và suy luận nhiều bước.
>  - So sánh với các chiến lược retrieval cố định và các phương pháp adaptive retrieval phù hợp.
>  - Đánh giá chất lượng thực hiện nhiệm vụ, chất lượng thông tin truy hồi, số lần retrieval, độ dài context và chi phí tính toán.
>  - Thực hiện ablation study đối với các thành phần chính như decision module, gating, exploration, retrieval cost và credit assignment.
>  - Phân tích retrieval behavior được hình thành trong quá trình học, bao gồm thời điểm retrieval, số lần retrieval, khoảng cách giữa các retrieval và mối quan hệ giữa retrieval timing với nhu cầu thông tin của nhiệm vụ.
>  - Đánh giá khả năng thực hiện multi-hop retrieval mà không sử dụng retrieval-decision supervision hoặc reinforcement learning.
> 
>  Giai đoạn 2
> 
>  1. Mở rộng sang adaptive tool use
> 
>  - Khảo sát khả năng khái quát hóa cơ chế adaptive retrieval sang bài toán tool-augmented language modeling.
>  - Mở rộng cơ chế quyết định từ retrieval timing sang quyết định thời điểm tương tác với các công cụ bên ngoài và lựa chọn hành động phù hợp.
>  - Nghiên cứu khả năng mở rộng cơ chế exploration và information–cost trade-off trong môi trường có nhiều loại công cụ.
>  - Đánh giá khả năng hình thành adaptive tool-use behavior mà không phụ thuộc hoàn toàn vào các trajectory được gán nhãn trước.
> 
>  2. Joint training giữa Retriever và LLM
> 
>  - Nghiên cứu khả năng mở rộng cơ chế sang joint training giữa retriever và LLM.
>  - Đánh giá ảnh hưởng của sự thay đổi retriever trong quá trình huấn luyện đến chất lượng retrieval và generation.
>  - Nghiên cứu cơ chế cập nhật retriever nhằm duy trì sự tương thích giữa nhu cầu thông tin của LLM và khả năng truy hồi trong quá trình joint training.
> 
>  3. Thực nghiệm và đánh giá
> 
>  - So sánh cơ chế mở rộng với các phương pháp adaptive retrieval và tool use phù hợp.
>  - Đánh giá chất lượng thực hiện nhiệm vụ, hiệu quả retrieval, số lần retrieval/tool call, độ dài context và chi phí tính toán.
>  - Phân tích khả năng khái quát hóa cơ chế adaptive decision từ retrieval sang các loại công cụ và hành động bên ngoài khác.

### J9 — Tasks (English)

> Phase 1
> 
>  1. Literature Review and Baseline Construction
> 
>  - Survey Retrieval-Augmented Generation (RAG) methods and mechanisms for determining retrieval timing, including fixed retrieval strategies, heuristics, adaptive retrieval, and approaches using retrieval-decision supervision or reinforcement learning.
>  - Develop a RAG baseline system with a fixed retrieval strategy as a basis for comparison.
>  - Investigate and evaluate the retriever on the experimental dataset to assess its ability to provide relevant documents for tasks requiring multi-step reasoning.
>  - Analyze the effects of retrieval timing and frequency on generation quality, the utilization of retrieved information, context length, and computational cost.
> 
>  2. Feasibility Analysis of Adaptive Retrieval
> 
>  - Conduct preliminary experiments to investigate whether retrieved information produces sufficiently different model representations between states with and without retrieved documents.
>  - Evaluate the differences between states with and without retrieved information through attention, hidden representations, and the language generation objective.
>  - Measure the improvement in the generation objective when retrieved information is provided, thereby estimating the information value of retrieval across different task states.
>  - Analyze the gradient propagated from the generation objective to the retrieval decision variable, particularly its magnitude and distribution for retrieval cases with different levels of utility.
>  - Examine the effects of different gating values on the model's generation capability to evaluate the stability of the information-mixing mechanism.
>  - Based on the experimental results, select and adjust the decision module architecture, gating mechanism, and representation of the retrieval state as appropriate.
> 
>  3. Development of a Latent Adaptive Retrieval Mechanism
> 
>  - Design a retrieval-timing decision mechanism integrated into the autoregressive generation process, allowing the model to learn its retrieval needs from the current task state.
>  - Develop a decision module and differentiable gating mechanism to regulate the extent to which retrieved document information is utilized during generation.
>  - Investigate a training mechanism in which signals from the generation objective are used to form retrieval decisions without requiring direct labels for retrieval timing or retrieval decisions.
>  - Introduce a retrieval-cost component to encourage the model to balance the benefits of additional information against the cost incurred by retrieval.
>  - Investigate a credit-assignment mechanism appropriate for the interval during which retrieved information continues to contribute to the reasoning process.
>  - Evaluate and adjust the mechanism to mitigate degenerate behaviors such as always retrieving or never retrieving.
> 
>  4. Exploration and Retrieval-Timing Learning
> 
>  - Develop an exploration mechanism that allows the model to explore different retrieval behaviors during training without retrieval-decision supervision.
>  - Investigate how to adjust the level of exploration based on the state and confidence of the decision module to balance exploration and retrieval cost.
>  - Analyze premature convergence toward retrieval or no-retrieval strategies and develop mitigation mechanisms when necessary.
>  - Train the model using the generation objective together with appropriate gating, retrieval-cost, and exploration components based on the results of the preliminary feasibility experiments.
>  - Analyze the formation of the retrieval policy and investigate whether retrieval timing can emerge from optimizing the generation objective and retrieval cost.
> 
>  5. Experiments and Evaluation
> 
>  - Evaluate the ability to learn adaptive retrieval timing on tasks requiring external information and multi-step reasoning.
>  - Compare the proposed approach with fixed retrieval strategies and appropriate adaptive retrieval methods.
>  - Evaluate task performance, the quality of retrieved information, the number of retrievals, context length, and computational cost.
>  - Conduct ablation studies on key components, including the decision module, gating, exploration, retrieval cost, and credit assignment.
>  - Analyze the retrieval behavior formed during training, including retrieval timing, number of retrieval steps, intervals between retrievals, and the relationship between retrieval timing and the information requirements of the task.
>  - Evaluate the ability to perform multi-hop retrieval without retrieval-decision supervision or reinforcement learning.
> 
>  Phase 2
> 
>  1. Extension to Adaptive Tool Use
> 
>  - Investigate the generalization of the adaptive retrieval mechanism to tool-augmented language modeling.
>  - Extend the decision mechanism from retrieval timing to determining when to interact with external tools and selecting appropriate actions.
>  - Investigate the extension of the exploration mechanism and the information–cost trade-off to environments involving multiple types of tools.
>  - Evaluate the ability to develop adaptive tool-use behavior without relying entirely on pre-annotated trajectories.
> 
>  2. Joint Training of the Retriever and LLM
> 
>  - Investigate the extension of the proposed mechanism to joint training of the retriever and LLM.
>  - Evaluate the effects of changes in the retriever during training on retrieval and generation quality.
>  - Investigate mechanisms for updating the retriever to maintain compatibility between the information needs of the LLM and the retrieval process during joint training.
> 
>  3. Experiments and Evaluation
> 
>  - Compare the extended mechanism with appropriate adaptive retrieval and tool-use methods.
>  - Evaluate task performance, retrieval effectiveness, number of retrieval/tool calls, context length, and computational cost.
>  - Analyze the generalizability of the adaptive decision mechanism from retrieval to other types of external tools and actions.

### L9 — References (English)

> [1] P. Qian, S. Wang, C. Peng, J. You, L. Liu, H. Yu, Y. Chen, and X. Jiang, “When Should Active RAG Retrieve? A Budget-Aware Evaluation of Utility, Calibration, and Cost,” arXiv preprint arXiv:2607.24010, 2026.
> 
>  [2] W. Su, Y. Tang, Q. Ai, Z. Wu, and Y. Liu, “DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12991–13013, 2024, doi: 10.18653/v1/2024.acl-long.702.
> 
>  [3] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection,” in International Conference on Learning Representations (ICLR), 2024.
> 
>  [4] Z. Jiang, F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, and G. Neubig, “Active Retrieval Augmented Generation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7969–7992, 2023, doi: 10.18653/v1/2023.emnlp-main.495.
> 
>  [5] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459–9474, 2020.
> 
>  [6] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W.-t. Yih, “REPLUG: Retrieval-Augmented Black-Box Language Models,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pp. 8371–8384, 2024.
> 
>  [7] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, “Corrective Retrieval Augmented Generation,” arXiv preprint arXiv:2401.15884, 2024.
