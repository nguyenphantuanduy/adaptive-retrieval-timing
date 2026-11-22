# Codex mission: Autonomous DACN report generation and validation

## Mission and autonomy
Produce a complete, academically rigorous Vietnamese Đồ án chuyên ngành (DACN, Phase 1) report for the **adaptive-retrieval-timing** project, using the approved plan and HCMUT LaTeX template. Write all six chapters, front matter, bibliography, and appropriate appendices; compile a final PDF and validate its scientific content and visual presentation.

You are authorized to plan, edit report source, build, inspect, debug, and iterate without chapter-by-chapter approval. **Do not stop at a first draft or merely successful compilation.** Stop when acceptance criteria are evidenced, or when a genuine blocker needs the author. Never claim checks were performed if they were not.

## Authority and boundaries
1. `docs/FIRST_DRAFT_PLAN.md`: approved scope, structure, research narrative, evidence hierarchy, numerical claims, and Phase 1 / Phase 2 boundary.
2. `slides-template/Specialized-Project-Template-VI/`: official HCMUT layout, chapter organization, and LaTeX driver/class.
3. `docs/LATEX_SETUP.md`: verified local MiKTeX, `.venv`, build and PDF inspection setup on drive D.
4. Actual code, notebooks, saved results, and source artifacts referenced by the plan: technical and experimental truth.
5. Original academic publications: prior-work facts and citations. Historical slides: provenance/context, not higher authority than current verified sources.

If the plan and evidence disagree, investigate, log the discrepancy, and use only defensible claims. Do not modify the approved plan, research/training/evaluation code, datasets, model checkpoints, or saved experimental artifacts. Avoid changes to the official HCMUT class; if a template bug truly requires changes, document it and request approval before changing the class. Keep report sources and generated outputs cleanly separated, and build artifacts ignored by Git.

## Phase 0: Self-defined goals and durable progress
Before drafting, create and maintain:
- `docs/report_execution/GOALS.md`: explicit, measurable acceptance criteria and pass/fail conditions.
- `docs/report_execution/PROGRESS.md`: phases, completed work, next actions, checkpoints for session resumption.
- `docs/report_execution/VALIDATION.md`: actual commands, results, evidence, checked pages, remaining warnings.
- `docs/report_execution/ISSUES.md`: defects, severity, attempts, resolution or escalation.

Define measurable goals for template compliance, approved-plan coverage, technical fidelity, experiment accuracy, figures/tables/equations, Vietnamese writing, citations, reproducibility, successful compilation, and page-by-page visual quality. Map each planned chapter/major claim/figure/table to authoritative source evidence. Do not mark a goal passed on intuition alone. Re-read these files when resuming after context compaction or interruption.

## Phase 1: Evidence audit
Read the plan and template completely. Verify the mechanism against implementation: autoregressive decision timing, gate and attention mixing, teacher-target LM objective, cost regularization, training data, exploration, no-grad collect / gradient-enabled replay, external E5/FAISS retrieval, cache, inference and scoring. Check H1/H2 probe design and datasets. Verify every reported metric and query count against saved artifacts. Audit local No-RAG / Vanilla RAG / REVIVE controls and matched qualitative examples. Check the provenance and caveats for published comparator numbers and training workload.

**H2:** Use `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/analyze_threshold_and_cost_simplified.ipynb` and its referenced data as the sole source for H2 interpretation and report figures. Present `τ = 0.4297` as the selected operational threshold, not a proven optimum. Distinguish forced attention mixing coefficient `d` from inference cutoff `τ`. Do not mention obsolete threshold arithmetic, old numerical mistakes, or use superseded H2 figures.

## Phase 2: Write the full report
Use the official six-chapter template and the exact registered Vietnamese title and verified student/supervisor metadata. Write fluent, formal Vietnamese; maintain a coherent scientific argument rather than a chronological development diary or pasted plan. Cover: motivation and RQs; foundations and related work; mechanism and mathematical definitions; teacher data and training; implementation; H1/H2; local three-dataset benchmark; threshold/query-count tradeoffs; verified qualitative retrieval cases; contextual published comparisons; limitations and future work.

Make the genuine contributions of ART/REVIVE and positive verified results clear. Do not understate the project through repetitive or excessive caveats, but do not claim that a system-level baseline difference isolates the timing mechanism. Method-specific prompts are legitimate in the practically controlled local comparison. Explicitly distinguish the original base checkpoint used by baselines from the fine-tuned REVIVE checkpoint. Keep Phase 2 / DATN extensions in future work. Never invent ablations, significance, latency, GPU-hours, training cost, sample outcomes, implementation details, or citations. Mark any indispensable unresolved fact in ISSUES.md instead of fabricating it.

## Phase 3: Accurate visuals and citations
Create readable, professional diagrams, plots, tables, equations, captions, and references. Source plots from verified data; reproduce H2 figures exclusively from the simplified notebook. Confirm axes, labels, units, sample denominators, thresholds, and table values. Depict the implemented mechanism faithfully, including discrete retrieval and the train/inference distinction. Prefer vector graphics where feasible; preserve aspect ratios and readable typography. Cite original publications and figure sources. Keep source-to-claim traceability in the execution documentation.

## Phase 4: Build and inspect
Use the existing scripts and environment in `docs/LATEX_SETUP.md`; prefer pdfLaTeX + BibTeX + latexmk as specified by the template. Compile the complete report. Inspect build exit status and logs for missing packages/assets, undefined references/citations, overfull boxes, and meaningful warnings. Extract PDF text to check Vietnamese diacritics, missing glyphs, unexpected placeholders, and blank pages. Render **every page** to images and **actually visually inspect every page**, including all figures, tables, formulas, cover, front matter, bibliography, and appendices. A clean build log does not substitute for visual review. If image inspection capabilities are unavailable, record the limitation and do not falsely mark visual QA passed.

## Phase 5: Iterative correction
Loop until all goals pass:
1. Build PDF and collect diagnostics.
2. Run automated content, log, structural, and numeric checks.
3. Render and visually inspect pages.
4. Compare critical formulas, claims, graphs, and tables against sources.
5. Record defects in ISSUES.md and fix the underlying cause.
6. Rebuild; inspect affected pages and run whole-document regression checks.
7. Update PROGRESS.md and VALIDATION.md with evidence.

Do not hide warnings, silently remove substantive scientific content to make layout easier, weaken accurate claims, or declare success without evidence. Keep iteration focused on real defects; avoid endless cosmetic churn after criteria are met.

## Escalation policy
Work autonomously on ordinary writing, citations, tooling, LaTeX, and layout. Ask the author **only** if an essential fact cannot be verified, authoritative evidence materially conflicts, privileged/system-level action needs permission, a repeated problem remains unresolved after documented reasonable attempts, or the only path forward would fabricate evidence. When asking, provide the exact blocker, attempts, impact, smallest needed decision, and recommended resolution. Otherwise continue without waiting for approval.

## Deliverables and completion report
Provide complete `.tex` sources, verified bibliography, plot/diagram sources, final compiled PDF, and the four execution Markdown files. Report exact file paths and build/inspection commands. Summarize each acceptance criterion with its evidence, any remaining limitations, and a chapter coverage/source-to-claim matrix. Never report all goals passed if any mandatory criterion remains unverified.

**Begin immediately with Phase 0, then proceed through full report generation and iterative validation without asking for approval between chapters.**
