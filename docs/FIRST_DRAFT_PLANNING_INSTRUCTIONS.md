# First Draft Report — Planning Instructions for Codex

> **Status:** PLANNING ONLY — author approval required before report drafting.
> **Project:** Adaptive Retrieval Timing (ART / REVIVE), specialized project report.
> **Expected planning output:** `docs/FIRST_DRAFT_PLAN.md`.

## 0. Mission and boundaries

Inspect the original university registration, the three dated LaTeX presentations, the university report template, and current research evidence. Produce a **source-grounded, template-compliant writing plan** for the first draft. **Do not write the actual report, chapters, or report LaTeX yet.** Do not modify any source materials, existing code, experimental predictions, or benchmark metrics. Do not run training, inference, or new retrieval experiments.

The report should reflect the **registered project scope**, follow the **university's template**, and accurately describe the **work actually completed**. Historical slides are evidence of research evolution, not automatically current ground truth.

## 1. Exact source locations and reading order

All paths are relative to the repository root unless otherwise specified.

### 1.1 Official university registration (Excel)

- Workbook filename: `Danh sách các nhóm Đồ Án Chuyên Ngành HK261.xlsx`.
- Locate this exact filename in the repository; **do not assume a directory** if it is not at the root.
- Target worksheet: `Trang tính2` (verify actual workbook sheet names rather than relying on the UI tab label).
- Target project: **STT = 8**, expected in **cell A9 / Excel row 9**. Verify the identifier before extracting the row.
- Read the **entire logical project record**, including all relevant columns to the right, even if the text is multiline, cells are merged, or the screenshot showed only part of the record.
- Extract the Vietnamese and English titles, descriptions, objectives, phases, expected outputs, supervisor and other relevant fields if present. Preserve original wording; identify missing/uncertain fields instead of guessing.
- Prefer direct workbook reading (e.g. `openpyxl`) to screenshot interpretation. Do not edit or resave the workbook.

**Authority:** This is the official *historical project registration* and anchors the proposed scope, objectives, and deliverables. It is **not** proof that proposed work has been completed.

### 1.2 Historical slides (LaTeX sources)

Read in **chronological order**, as indicated by the dates embedded in the directory names:

1. `slides-template/DACN_29_9_Group8/` — **29 September** (29/9), earliest.
2. `slides-template/DACN_27_10_Group8/` — **27 October** (27/10), intermediate.
3. `slides-template/DACN_10_11_Group8/` — **10 November** (10/11), latest of the three.

The names encode **day/month**, not month/day. Do not read in lexicographic order. Do not infer a calendar year from the folder names alone; verify from source metadata or content if year matters.

Read the primary `.tex` files, included `.tex` files, `.bib` files, diagrams, figures, tables, notes, and other supporting assets. Prefer LaTeX source to rendered PDF for technical content; consult rendered PDFs only when needed to understand visual layout, diagrams, or figures. Identify which statements, figures, experiments, and explanations were retained, refined, contradicted, or superseded across the three milestones.

### 1.3 University report template

- `slides-template/Specialized-Project-Template-VI/`

Inspect its structure, LaTeX class/style, main file, chapter layout, mandatory front matter, bibliography/citation conventions, figure/table/equation formatting, and any university-specific rules. Distinguish **mandatory requirements** from **example/placeholder content**. Do not alter the template. The plan must map the report outline to the actual template structure, not invent a competing format.

### 1.4 Current project evidence

Prioritize:

- `docs/BENCHMARK_ANALYSIS.md` (current benchmark and retrieval-cost analysis).
- Relevant project docs (methodology, hypotheses, training provenance, architecture, evaluation protocol, experiments), including `docs/TRAINING.md` or other similarly named files **if present**.
- Phase 1 / Stage 2 H2 threshold-analysis notebooks, scripts, plots, and notes. Search relevant paths such as `phase_1/stage_2_hypothesis/` **if present**, rather than assuming exact names.
- Implementation code **selectively**, to verify material claims, especially retrieval decisions, query logging, model training, inference, and scoring.

Record the exact inspected file paths. Do not claim to have verified any source that could not be opened.

## 2. Source authority and conflict-resolution policy

Use distinct authority domains rather than treating one document as universally authoritative:

1. **University report template:** mandatory formatting and structural requirements.
2. **Official Excel registration:** original approved topic, intended scope, objectives, phases, and expected outputs.
3. **Verified latest research artifacts:** what was actually implemented, measured, and concluded, including corrected H2 analysis and current benchmark data.
4. **Historical dated slides:** development of the research story, terminology, motivation, earlier hypotheses, and illustrations, subject to verification.

If the official registration and current work diverge, **do not silently rewrite either**. Document the divergence, recommend a truthful report framing consistent with the registration where possible, and flag substantive scope changes for author approval.

If historical slides conflict with later verified results, identify the original claim, its date/source, newer evidence, and the proposed correction. Never silently combine contradictory claims. Mark unresolved matters as `Needs author confirmation`.

## 3. Reconstruct the research narrative

Establish, with source references:

- Registered Vietnamese and English project titles and scope.
- Motivation, problem statement, research questions, and objectives.
- Related approaches and limitations, as actually discussed in the sources.
- Proposed ART/REVIVE approach and its relation to retrieval timing, information value, retrieval cost, and autoregressive generation.
- Hypotheses and their evidence, with particular scrutiny of H2.
- Model architecture, training setup, data, evaluation protocol, and relevant implementation details.
- Current experimental outcomes and retrieval efficiency.
- Supported contributions versus hypotheses, preliminary observations, future work, and limitations.

Do not invent novelty claims, equations, citations, completed experiments, or causal explanations. Preserve the project's established technical vocabulary unless a correction is justified and documented.

## 4. CRITICAL — H2 threshold-selection correction

Historical slides may contain an outdated or incorrect account of how the retrieval decision threshold was selected. **Audit every claim involving H2, threshold selection, `0.4297`, optimality, gradient contrast, or empirical validation.**

Context from prior author-reviewed analysis, **to be independently checked against actual repository evidence**:

- `τ = 0.4297` was the **historical deployed/default benchmark threshold**, not a proven mathematical optimum.
- The historical four-extrema computation gave approximately **0.421875**.
- The **pooled gradient-contrast** estimator peaked at **0.40625 on the observed grid**.
- Local quadratic interpolation estimated a peak near **0.42165**.
- `0.4297` was close to the pooled peak and can be discussed as a near-peak operating point, but its original numerical entry was identified as an **author-confirmed mistake**.
- A paired bootstrap did not resolve a difference between `0.4297` and `0.421875`; **failure to resolve a difference does not prove equivalence**.
- Some individual reasoning conditions, including the 4-hop condition, may differ from pooled behavior. Do not present a universal threshold optimum.

**Required actions:** Locate the latest H2 notebook/analysis, read estimator definitions, assumptions, plots, and numerical results, and reconcile the slide narrative with the verified analysis. If the relevant source cannot be found, preserve these numbers as *unverified context* and ask the author to provide the missing artifact before treating them as report-ready facts.

**Do not globally replace `0.4297` with `0.42165`.** Distinguish:

- **H2 selection analysis:** estimates and rationale for a decision threshold.
- **Historical ART benchmark runs:** actually executed with default `τ = 0.4297`.
- **Downstream threshold sensitivity:** observed quality and retrieval counts at multiple `τ` values on MuSiQue, HotpotQA, and 2WikiMultiHopQA.

These are different analyses; downstream benchmark maxima do not retrospectively validate the historical H2 derivation.

## 5. Benchmark and methodology cautions

Read `docs/BENCHMARK_ANALYSIS.md` in full. Treat its saved values and qualifications as current documented evidence, while verifying claims from source artifacts when necessary.

Important distinctions:

- Original pretrained Qwen baseline versus fine-tuned ART/REVIVE model.
- No-RAG, single-retrieval Vanilla RAG, and iterative ART have different retrieval/prompt protocols.
- Baseline and ART response extraction/scoring may differ; the same nominal metric functions do not ensure a fully matched protocol.
- All runs containing 500 samples does not by itself prove matched IDs, aliases, order, or dataset versions.
- `top5` is retrieval **top-k**, not decision threshold; historical default is `τ = 0.4297`.
- Threshold sweeps are descriptive, not evidence of universal optimality or statistical significance.
- `pure_art_queries`-based counts are logged search-call events / retrieval-workload proxies, not independently verified successful retrievals or end-to-end latency.
- A quality improvement between ART and baselines cannot be attributed **solely** to retrieval timing without appropriate same-checkpoint controlled experiments.
- Do not extrapolate unmeasured latency, compute savings, or significance from query counts alone.

Use the existing results to plan relevant tables and figures. Do not rerun experiments or alter the metrics.

## 6. Analyze the template and map the report

Extract the actual required front matter, chapters, appendices, references, and formatting constraints. For every proposed section, specify:

- Template location / corresponding file or section.
- Section purpose and key argument.
- Precise supporting sources (file paths and, where possible, sections/labels/figures).
- Intended depth and any necessary equations, figures, or tables.
- What is verified, what needs clarification, what requires further experiments, and what can be deferred.
- Whether the proposed content matches the official Excel registration.

Propose a coherent narrative, but **do not decide substantive scope changes or unsupported contribution claims on the author's behalf**.

## 7. Required planning deliverable

Create **only** the planning document:

`docs/FIRST_DRAFT_PLAN.md`

It must contain at least:

1. **Source inventory and reading log:** exact paths inspected, roles, gaps, and access failures.
2. **Official registration extraction:** original bilingual titles, descriptions, objectives, phases, expected outputs; clearly cite workbook, sheet, and STT.
3. **Template compliance matrix:** mandatory sections/format and corresponding planned report locations.
4. **Research narrative:** candidate title (consistent with registered title), central problem, research questions, objectives, and defensible contributions.
5. **Chronological slide evolution:** 29/9 → 27/10 → 10/11; what changed and which statements are superseded.
6. **Complete hierarchical outline:** chapters and sections conforming to the template.
7. **Detailed section-by-section plan:** purpose, arguments, sources, proposed figures/tables/equations, expected depth, status, and priority.
8. **Alignment with the Official Project Registration:** registered objective → completed/partial/planned evidence → proposed report coverage → discrepancies requiring approval.
9. **Historical Corrections and Evidence Audit:** original claim, slide/date/source, verified evidence, required correction, recommended report wording, verification status. Give H2 special attention.
10. **H2 threshold-selection treatment:** precise separation of original mistake, corrected analysis, historical deployment, and benchmark threshold sensitivity.
11. **Experimental evidence map:** dataset, model/checkpoint, baseline, scorer, threshold, result, caveat, source; avoid unsupported causal claims.
12. **Figures, tables, equations, and citations plan:** reuse versus create; provenance; bibliography gaps; no invented references.
13. **Open questions / decisions requiring author approval:** prioritized by impact on first-draft readiness.
14. **Writing roadmap and checklist:** recommended drafting order, dependencies, and a status for each section: `Ready to write`, `Needs clarification`, `Needs more evidence/experiments`, or `Defer beyond first draft`.

If the planning file already exists, read it and **update it conservatively** to incorporate these instructions, preserving valid earlier findings rather than blindly overwriting them.

## 8. Working procedure

1. Inventory all required sources and verify the Excel workbook, worksheet, and STT.
2. Read the registration and report template carefully.
3. Read the slides in chronological order (29/9, 27/10, 10/11).
4. Inspect current research docs and verify the H2 correction and benchmark caveats.
5. Build a claim-to-evidence mapping and identify contradictions.
6. Draft `docs/FIRST_DRAFT_PLAN.md` only.
7. Self-check against the template, official registration, historical correction, and all constraints.
8. Stop and await author review.

If a critical source is missing, do not fabricate it. Complete the verifiable portions of the plan, mark blocked sections, and explicitly ask for the missing source.

## 9. Final Codex response

Briefly report:

1. Files and directories inspected (and any missing files).
2. Verified official title/scope and template structure.
3. Proposed chapter outline and central research narrative.
4. Major chronological changes across the three slide decks.
5. H2 correction status and any unresolved numerical/source issues.
6. Key mismatches between official registration, slides, and current experiments.
7. Highest-priority author decisions.
8. Exact path of `docs/FIRST_DRAFT_PLAN.md`.

**Stop. Do not write or compile the research report until explicitly approved.**
