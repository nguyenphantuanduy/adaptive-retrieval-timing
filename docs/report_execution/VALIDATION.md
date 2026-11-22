# DACN validation record

## Reproduction

Run from the repository root with the existing environment in [LATEX_SETUP.md](../LATEX_SETUP.md).

~~~powershell
.\.venv\Scripts\python.exe -B report/dacn/generate_assets.py
.\scripts\build-hcmut-template.ps1 -Report
.\.venv\Scripts\python.exe -B scripts/inspect_pdf.py build/latex/dacn/main.pdf --output-dir build/latex/dacn/inspection
~~~

The original template remains the build script's default target. The report sources and compile-ready graphics are under [report/dacn](../../report/dacn/). The PDF, BibTeX log and LaTeX log are under [build/latex/dacn](../../build/latex/dacn/). No model inference, retrieval, or training was invoked.

## Chapter and claim matrix

| Report coverage | Evidence and check performed |
|---|---|
| Cover/front matter, chapter sequence | Approved plan §2, workbook STT 8 and slide cover metadata transcribed there. Report class and logo verified byte-identical to the HCMUT template; checked source-to-build copies. Cover, prefaces and lists inspected at pages 1–9. |
| Ch. 1: objective and Phase 1 / Phase 2 boundary | Approved plan §§2, 4, 7 and registration transcript. Pages 10–11. |
| Ch. 2 and related-work table | Original IRCoT, Adaptive-RAG, FLARE, DRAGIN, Self-RAG, R-Search and budget-aware paper sources. Bibliography points to the publication/preprint records. Pages 12–13. |
| Ch. 3: gate, attention, target loss and penalty | [model.py](../../shared/ART_Models/ART_REVIVE/model.py), [attention_mixing.py](../../shared/ART_Models/ART_REVIVE/attention_mixing.py), [training_v2.py](../../REVIVE_training/training_v2.py) and [REVIVE architecture audit](../REVIVE_ARCHITECTURE.md). Equation and diagram review on pages 14–15. |
| Ch. 4: teacher and training path, retrieval timing, cache | [trainer.py](../../REVIVE_training/trainer.py), model collection/replay/generate branches, [cache_utils.py](../../shared/ART_Models/ART_REVIVE/cache_utils.py), retrieval adapters and splitter, [training provenance](../TRAINING.md). Pages 16–18. |
| Ch. 5: H1 | H1 validation driver and saved LM-loss CSV; rechecked all 33 grid points per ratio, pooled denominators and printed endpoints. Fig. 5.1 regenerated with no training, page 20. |
| Ch. 5: H2 | [Simplified H2 notebook](../../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/analyze_threshold_and_cost_simplified.ipynb), referenced gradient CSV and cell 8. Ran only cells 2, 4, 8, 10, 11, 13 to regenerate figures 5.2–5.3. The report generator repairs a literal tab in the legend string and staggers nearby numeric labels for presentation. Data, gradients, threshold, points and costs are unchanged. Page 21. |
| Ch. 5: local tables and sweep | Independently rescored 4,500 saved predictions using method-specific preprocessing and the shared functions in [utils.py](../../REVIVE_training/utils.py). All 27 metrics exactly equal their JSON values. Each dataset's 500 ordered triples of ID, question and full gold aliases match across all three systems. Asset generation reconciles all 10 sweep rows with their metric files and per-sample query counts, totals and rates. Highest observed F1 rows are MuSiQue 0.60 (32.08), HotpotQA 0.50 (49.68) and 2Wiki 0.4297 (46.30). [Benchmark manifest](../BENCHMARK_ANALYSIS.md). Pages 22–23. |
| Ch. 5: published comparison | Verified [R-Search Table 1](https://aclanthology.org/2026.findings-acl.1896.pdf#page=6), including dataset order and 3B block. Kept external EM/F1 separate from local results and documented post-hoc selection. Page 24. |
| Ch. 6: synthesis/limitations | Checked against H1/H2 qualified conclusions, locally reproduced scores and the scope in the registration. Page 25. |
| Bibliography, Appendices A/B | BibTeX generated 17 entries; checked bibliography pages 26–27. Three matched EM examples on pages 23 and 28 rechecked at MuSiQue ID 27, HotpotQA ID 73, 2Wiki ID 0, and linked from the plan. Reproducibility and conditional workload proxy on page 29. R-Search §4.3 provides 195 maximum steps, batch 256 and five responses per prompt; the ratio is a scenario and carries no measured compute claim. |

## Iterations

1. First invocation exposed Windows PowerShell native stderr handling. Direct invocation exposed the memoir legacy font-command error. Replaced the legacy math command.
2. First full PDF rendered at 29 pages. Reviewed all pages and corrected path overflow, a stranded workflow diagram, bibliography spacing and the H2 sign narrative.
3. Subsequent full rebuild returned 0. Rendered and visually checked all 29 pages in numbered sheets and opened the equations, H2, tables and scatter at native scale. Fixed the split sweep introduction and regenerated H2 as vector graphics. The full-size review exposed a malformed threshold glyph in the notebook legend and adjacent crossing labels, addressed in the asset generator.

4. Final regression build exited 0 with latexmk 4.89, pdfTeX 1.40.28 and BibTeX 0.99e. Rendered all 29 A4 pages at 144 dpi. Compared PNG hashes against the previously visually inspected 29-page build: 25 pages were pixel-identical, and pages 21, 22, 23 and 29 were re-opened at native size. Both H2 legends render and extract the Greek tau correctly. The final review covered pages 1 through 29 and all six figures. No suspiciously empty or textless pages.
5. Final logs: no missing assets/packages, undefined references/citations, duplicate destinations, missing glyphs or overfull boxes. One benign Appendix B underfull box (badness 1087) and the existing memoir/Babel/MiKTeX warnings remain. Readable on the inspected page. PDF extraction found no sample placeholder, unresolved reference marker or replacement characters. PDF contains 118 link annotations.
6. Final PDF SHA-256: EB6711C95A1F8D3DF2CFBF51B161FCCA7A60A597FE32AEF2B8B66EED8967EC41.

## Acceptance result

All six gates in GOALS.md pass. The visual evidence is in the numbered page PNGs and contact sheets under build/latex/dacn/inspection. Remaining historical provenance and cover metadata limits are listed in ISSUES.md.

## ART naming revision

Ran the three reproduction commands above. The asset generator reconciled all 10 threshold configurations again. A before/after source comparison verified identical ordered numeric tokens and inline formulas in every TeX file, including all generated metric tables. The four vector plots had identical extracted text and rendered pixels after regeneration, so their original bytes were retained. No experiment was rerun.

The abstract defines Adaptive Retrieval Timing (ART). Public labels now read No-RAG, Vanilla RAG and ART. The source and extracted PDF audit found no public-facing occurrences of the internal name. The only PDF occurrences (page 29, Appendix B) are the literal directories shared/ART_Models/ART_REVIVE and REVIVE_training. The generator also uses REVIVE_training as a literal path in three places to locate existing evidence.

The rebuilt PDF has 29 pages. All were rendered, with no blank or textless pages flagged. Pages 4, 5, 9, 10, 12, 13, 14, 19, 20, 22, 23, 24, 25 and 28 changed and were visually inspected in the numbered naming-review sheets. The other 15 pages matched the previous render byte for byte. No new overflow, clipped labels or missing glyphs. Build exit code 0. Final PDF SHA-256: 7F77AF165957C0788BBEE3E4503E963FDECE61879029689D70A80EA90F5FF4C4.

Modified report source files (relative to report/dacn):

- generate_assets.py
- chapters/front/abstract.tex and workload.tex
- chapters/main/01-introduction.tex, 02-background.tex, 03-methodology.tex, 05-evaluation.tex and 06-conclusion.tex
- chapters/back/appendix-a.tex
- graphics/local-table.tex, sweep-table.tex and external-table.tex

Updated all four execution records and rebuilt the PDF and inspection artifacts under build/latex/dacn.

## 2026-11-22 author review

The current PDF supersedes the builds above. See REVIEW_2026-11-22.md for the final checksum, decision-by-decision audit, and visual evidence.
