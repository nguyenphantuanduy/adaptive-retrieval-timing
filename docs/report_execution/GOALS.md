# Acceptance gates

A gate passes only with evidence in [VALIDATION.md](VALIDATION.md). Unresolved defects and residual limitations belong in [ISSUES.md](ISSUES.md).

| Gate | Pass condition |
|---|---|
| G1 — Template | Exact Vietnamese registration title and verified student/adviser metadata, SHA-256 identical HCMUT class and logo, four prefaces, three lists, six chapters, bibliography and two appendices. |
| G2 — Technical fidelity | Source matrix maps each chapter and major mechanism to code or primary sources. Timing, search, masking, cache, teacher inputs, and gradient paths are covered. Phase 2 stays in future work. |
| G3 — Empirical accuracy | All 27 local scores reproduced, ordered IDs/questions/aliases match, 10 sweep configurations reconciled with metric and query-count files, three EM cases checked, and published scores separated from local results. H1 and H2 figures have traceable probe sources and H2 uses only the simplified notebook. |
| G4 — Language, citations and assets | Formal Vietnamese throughout, no template sample text, correct tables/units and resolved references, citations and links. Every figure has a caption and a reproducible source. |
| G5 — Build | Asset script runs without inference, PDF pipeline completes with exit 0, and final logs contain no missing files, undefined references/citations, missing glyphs or overfull boxes. Explain any residual environment warnings. |
| G6 — Visual and iterative QA | Render all pages, inspect them individually or in numbered page sheets, check critical figures at native scale, and record layouts/issues, corrections and a final regression. |

## Naming consistency
The report uses Adaptive Retrieval Timing (ART) at first mention and ART thereafter for the method, trained model and evaluated system. Comparison labels are No-RAG, Vanilla RAG and ART. Literal technical paths are exempt. Verify the source, generated assets and final PDF.
