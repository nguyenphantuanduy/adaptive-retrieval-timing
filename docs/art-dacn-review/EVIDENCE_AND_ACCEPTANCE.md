# ART DACN — Evidence requirements and acceptance checklist

## Evidence discovery (must precede source-grounded edits)

- [ ] Identify official HCMUT LaTeX template and exact declaration text, mandatory structure and formatting.
- [ ] Locate the report TeX root, chapter files, BibTeX, images and build command; compare to current PDF.
- [ ] Locate actual ART Decision Head, Attention Mixing, regularization, Collect–Replay, teacher/trajectory and inference implementations; inspect relevant configs.
- [ ] Verify H1/H2 probe setup, `d` grid, conditions, loss and gradient plots from notebooks/artifacts; distinguish simplified probe from full ART.
- [ ] Verify `tau=0.4297`, fixed-threshold and sweep data from raw results; preserve every verified number.
- [ ] Verify No-RAG, Vanilla and ART checkpoints, prompts, answer extraction, retrieval configs and exact comparison protocol.
- [ ] Verify metric functions Acc/EM/F1 in ART evaluation scripts and R-Search paper/code before attributing provenance.
- [ ] Search actual repository for demo-v3 results without assuming a specific path. Validate a **single same-ID MuSiQue case** with No-RAG fail, Vanilla fail, ART pass, raw outputs and evidence/trajectory; otherwise document blocker.
- [ ] Verify official metadata for IRCoT, Self-RAG, R-Search, models and datasets; ensure every citation is justified.
- [ ] Verify actual status/scope of GitHub/HF artifacts where possible; never claim public access without evidence and author approval.

## Scientific consistency

- [ ] Central thesis explicitly says retrieval timing is studied as a consequence of LM Loss vs retrieval regularization trade-off, not direct decision labels or RL.
- [ ] H1/H2 = foundational-assumption verification; ART benchmark = end-to-end system evaluation.
- [ ] `d` (forced mixing) and `tau` (inference threshold) are distinct; no unproven optimality claim.
- [ ] Fixed `tau=.4297` is primary; per-benchmark sweep best is exploratory/test-selection caveat.
- [ ] F1 gains are system-level; checkpoint/prompt/extraction differences are transparently stated without excessive disclaimers.
- [ ] Retrieval count is not presented as measured latency or total computational cost.
- [ ] No assertion that gradients traverse discrete retrieval/FAISS or that each local decision is optimal.
- [ ] No new experiments or unverified results added.

## Document and presentation

- [ ] HCMUT template requirements and verbatim declaration preserved.
- [ ] Academic Vietnamese + natural English technical terminology consistently used.
- [ ] Abstract has problem → method/H1/H2 → results and matches verified benchmark numbers.
- [ ] Chapters 1–6 match the brief; Ch4 is implementation, Ch5 is experimental evidence.
- [ ] Ch1 MuSiQue baseline-only failure example is consistent with Ch5/Appendix A ART case.
- [ ] Every quoted model answer, question, ground truth and retrieved passage matches raw artifacts; omissions clearly marked.
- [ ] Diagrams are readable, correct, reproducible from separate source, captioned/referenced, vector when feasible.
- [ ] Related Work is mechanism-focused; no unsupported leaderboard/SOTA comparison; Adaptive-RAG removed from designated comparison.
- [ ] Bibliography compiles, citation keys resolve, metadata is verified, references relevant.
- [ ] Appendix B does not promise access to private resources or equate counts with compute.
- [ ] R-007 does **not** introduce a new task/experiment.
- [ ] Conclusion reflects R-008: strong but bounded claims.

## Build and final reporting

- [ ] Build command and toolchain recorded; PDF generated successfully.
- [ ] No missing figures, broken references, unresolved citations, overfull boxes or unreadable diagrams in inspected PDF.
- [ ] Report all changed source files and explain main changes by U/R ID.
- [ ] Provide paths to PDF and reproducible diagram sources.
- [ ] Report missing evidence as `BLOCKED` with exact missing path/artifact and proposed next action; do not invent a substitute.
- [ ] Preserve original experimental artifacts and code unless a separate, explicit authorization is given.
