# Author-approved decisions for revising the DACN first-draft plan

**Status:** Author decisions approved for planning. **Do not write the report yet.**

## Task

Read this file together with:

- `docs/FIRST_DRAFT_PLANNING_INSTRUCTIONS.md`
- `docs/FIRST_DRAFT_PLAN.md`
- `docs/BENCHMARK_ANALYSIS.md`
- `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/analyze_threshold_and_cost_simplified.ipynb`
- Official registration workbook: `Danh sách các nhóm Đồ Án Chuyên Ngành HK261.xlsx`, worksheet `Trang tính2`, project STT 8 (row 9)
- Dated LaTeX slide sources, chronologically: `slides-template/DACN_29_9_Group8/` (29 September), `slides-template/DACN_27_10_Group8/` (27 October), `slides-template/DACN_10_11_Group8/` (10 November)
- Official report template: `slides-template/Specialized-Project-Template-VI/`
- Relevant current code, metrics, and artifacts as needed to verify claims.

Preserve the source inventory and useful evidence audits from the existing plan. Revise **only** `docs/FIRST_DRAFT_PLAN.md`; do not modify source files, template, workbook, slides, experiments, or report LaTeX. Stop for approval after updating the plan.

## 1. Scope: DACN versus DATN

The registration's **Phase 1 is Đồ án chuyên ngành (DACN)**, the subject of this report. **Phase 2 is the later Đồ án tốt nghiệp (DATN)** and belongs only in future work. Do not frame the DACN as an unfinished half of a two-phase report. Explain the boundary naturally, without implying Phase 2 work has already been completed.

## 2. Report maturity and format

This is **version 1 of a full, six-chapter academic report**, not a progress report. Plan the most complete coherent draft possible from verified evidence. Follow the Vietnamese university LaTeX template for mandatory structure, cover, front matter, chapters, references and appendices. Do not add filler or invent results to make it look complete.

## 3. H2 and threshold: authoritative correction

The author confirms `τ = 0.4297` is a **selected operational threshold**, **not a mathematically proven optimum**. Use the current `analyze_threshold_and_cost_simplified.ipynb` as the primary H2 explanation; **do not copy the old threshold-selection narrative from slides**. H2 concerns forced mixing coefficients and conditional gradients; distinguish the forced coefficient `d` from the deployed cutoff `τ`. The historical arithmetic `0.421875` may be mentioned briefly if useful, but should not dominate the report. Missing pooled-contrast/quadratic/bootstrap artifacts are **not blockers**; do not present their unverified numerical claims as established results. Preserve `0.4297` as the actual historical benchmark setting, without silently replacing it. Avoid implying the gradient probe alone proves a calibrated or universally optimal retrieval policy.

## 4. Language and metadata

- Language: **Vietnamese**.
- Official title: **Thời điểm truy hồi thích ứng: Học quyết định truy hồi mà không cần giám sát trực tiếp hay học tăng cường**.
- Student: **Nguyễn Phan Tuấn Duy**.
- MSSV: **2310491**.
- Class: **MT23KHM2** (use where appropriate under template).
- Supervisor: verify spelling and academic title from slide covers.
- Registration workbook is the authoritative source for official registered project information. Use placeholders only for genuinely unknown administrative fields such as submission date/council, not for known metadata.

## 5. Scientific narrative and report priorities

Emphasize the research workflow, not a chronological development diary:

1. Motivation, problem statement and related research.
2. Proposed latent adaptive retrieval timing and differentiable gating.
3. H1/H2 hypotheses, experimental verification, findings and limitations.
4. Final teacher-data preparation, objective, training/collect–replay procedure and architecture.
5. Final REVIVE inference/retrieval workflow.
6. Evaluation, retrieval behavior, quality–workload tradeoffs and discussion.
7. Conclusions and future DATN directions.

Distribute this narrative within the **six mandatory template chapters**. The demo → demo-2 → demo-3 development history can be omitted from the main report; use a historical comparison only if necessary to substantiate a research claim. Prioritize architecture, hypothesis verification, final training and final benchmarks. Figures should show the **current implementation**, not obsolete slide diagrams.

## 6. Evaluation and core research claim

Define **Acc, EM and token-level F1** according to the actual evaluator implementation, including answer-extraction caveats. Present the local **No-RAG, Vanilla RAG and ART/REVIVE** comparisons on the three evaluated multi-hop datasets, with clear model, retrieval and scorer provenance. The core contribution is a mechanism for learning adaptive retrieval timing **without direct retrieval-decision labels or reinforcement-learning-based decision optimization**. Support this through objective, architecture, observed gate/retrieval behavior, threshold sweeps and logged query counts; **do not claim that the benchmark quality difference alone causally proves the timing mechanism is responsible**. A few successful example traces are welcome if their source/run provenance can be verified. Do not claim unseen experiments or unsupported statistical significance.

## 7. Contextual R-Search comparison and training workload

Include a carefully qualified comparison to relevant published approaches, especially R-Search, to contextualize quality and the training procedure. The 10 November slides contain an approximate **6% rollout-related comparison**. Inspect and reconstruct its arithmetic, assumptions and units. The author wants the potential training-efficiency benefit communicated, but **only as a clearly labeled proxy/estimate if that is what the evidence supports**. Do not turn rollout counts or token-exposure estimates into measured GPU-hours, FLOPs, wall-clock time, or a verified 94% compute saving. Do not present literature numbers as matched local benchmark conditions. If the proxy is not comparable or reconstructable, state the issue and propose defensible alternative wording rather than inventing precision.

## 8. Author's final benchmark selection rule (new decision)

The external literature comparison table must use **REVIVE's highest observed F1 threshold independently for each dataset**, NOT the same default threshold everywhere:

| Dataset | Selected threshold | Best observed F1 | Average retrieval queries |
|---|---:|---:|---:|
| MuSiQue | `τ = 0.60` | `32.08` | `2.588` |
| HotpotQA | `τ = 0.50` | `49.68` | `3.166` |
| 2WikiMultiHopQA | `τ = 0.4297` | `46.30` | `4.012` |

These values come from the prior planning discussion; **verify against the original run artifacts before incorporating into the report plan**. Extract **Acc, EM, F1, and average query count from the exact same run/threshold for each dataset**. Never combine EM from the default run with F1 from a best-F1 run. If a corresponding metric is unavailable, mark it as pending rather than filling it from another run.

Label this explicitly as **post-hoc best-observed per-dataset F1 selection**, not independently validation-tuned test performance or a single deployed operating point. The actual historical default `τ = 0.4297` remains valid for the separate main-system/default-threshold analysis. Do not conflate the H2 forced coefficient with the inference cutoff.

For the contextual external comparison, examine **R-Search's own reported baseline table**, including **IRCoT, Search-R1, Adaptive-RAG, and R-Search** where applicable. Use published EM/F1 with source citation, and clearly note different datasets/splits, prompts, corpora, inference budgets and evaluation setups. Never claim matched-condition superiority or treat published scores as locally reproduced. **Acc** belongs in the internal comparison unless published methods supply a directly comparable Acc definition.

**Evaluation hierarchy:** (1) Primary: mechanism verification (H1/H2, retrieval behavior, threshold/query analysis) and local ART vs No-RAG vs Vanilla RAG, with clear Acc/EM/F1 definitions; (2) Secondary: contextual published methods comparison; (3) Tertiary: approximate training-workload efficiency. The core scientific argument is *learnability of adaptive retrieval timing without direct retrieval-decision supervision or RL*, not SOTA dominance or a training-cost record.

## 9. Verified-by-author REVIVE training pipeline and cautious R-Search accounting

The author confirms the following high-level REVIVE pipeline (check code/configs for exact values and token lengths):

- **Teacher data generation:** one generation pass over approximately **20,000 samples** using an **8B teacher**. This is one dataset-level generation pass, **not one Transformer forward operation**; autoregressive generation entails many forward steps.
- **Student training:** **3B model**, approximately **0.75 epoch × 20,000 trajectories = 15,000 trajectory presentations** (approximate; count tokens and packing/repeats if available).
- **Partial unfreezing:** approximately **40% of student parameters trainable**. Verify whether the fraction refers to parameters or layers and which modules are unfrozen.

For R-Search, **inspect the original paper and supplementary materials** for model size, RL algorithm (e.g., PPO/GRPO), prompt batch size, rollout multiplicity, training steps, actual units and any reported compute. Prior discussion mentioned a possible calculation `195 steps × 256 prompts/step × 5 responses/prompt = 249,600 rollout presentations`, but this is **conditional and must NOT be asserted without verifying that the batch-size unit, rollout multiplier and step count are compatible**. Do not silently treat training samples, trajectory presentations and generated tokens as interchangeable.

A crude `15,000 / 249,600 ≈ 6.0%` comparison, if verified, represents **relative trajectory/rollout workload only**; it is NOT a measured FLOPs, GPU-hour, wall-clock or total-cost ratio. In particular, **do not multiply 6% by 40% unfreeze and claim a total compute saving**. The teacher's 8B generation cost and RL's sampling, scoring and training must be acknowledged. Any proposed component-based cost accounting should identify its academic sources, assumptions, consistent units and potential double-counting (generation already uses forward passes). Cite peer-reviewed sources for the decomposition/compute estimation and distinguish paper-established methodology from the project's own derived proxy. If reliable quantities are missing, keep the discussion qualitative or bounded rather than manufacturing a precise percent.

Keep this section **short and secondary** in the eventual report. Its role is to illustrate a potential training-efficiency advantage, not prove the principal contribution.

## 10. Fair treatment of unfinished Phase 1 objectives

Evaluate the evidence for retriever quality, credit assignment, exploration, ablations, context length, latency and other registered tasks individually. Distinguish:

- demonstrated core contributions;
- secondary measurements not yet performed;
- limitations that constrain causal interpretation;
- future enhancements.

Be academically fair and appropriately defend the work: do not bury its demonstrated contributions under a long list of missing optional tests, but do not claim unperformed ablations or measurements. DATN-specific Phase 2 objectives are future work, not Phase 1 failures.

## 11. Visuals, cases and reproducibility

Prioritize system architecture, decision/attention mixing, H1/H2 verification, collect/replay training, final inference, metric tables and retrieval quality–query-count tradeoffs. Include a small number of successful/failure cases only when they clarify a verified mechanism or observation; the full demo history is unnecessary. Include model/data/index/hyperparameter/hardware/run provenance where available and material. Do not block first-draft planning on optional immutable manifests or unknown administrative details.

## 12. Required update to `docs/FIRST_DRAFT_PLAN.md`

Revise the executive planning decision, research questions and contributions, DACN/DATN alignment, H2 treatment, six-chapter outline, evidence and figure priorities, benchmark narrative (including best-F1 per-dataset external table), verified teacher/student training workload, cautious R-Search workload comparison, limitations and writing roadmap. Remove author questions already answered here. Retain a **short list of genuinely unresolved factual checks**, such as the 6% calculation, trace provenance or cover administrative fields. Clearly label what is verified, inferred or still awaiting evidence.

**Approval gate:** Update the plan only, summarize changes and unresolved factual checks, then stop. **Do not draft or compile the report.**
