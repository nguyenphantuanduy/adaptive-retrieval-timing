# ART DACN — Final implementation brief

## 0. Authority and scope

The author completed a collaborative review and approved U-001–U-015, R-001–R-006, and R-008. R-007 is **closed without additional changes** because the existing MuSiQue case study requirement already covers the intended illustration. This brief authorizes **editing the LaTeX report**; it does **not** authorize new training, evaluations, fabricated examples, or changing benchmark numbers. The full decision record is `ART_DACN_REVIEW_LOG.md` and prevails for nuances omitted here.

**Priority:** official HCMUT template > academic-paper standards where template is silent > author decisions. The method's public name is **ART (Adaptive Retrieval Timing)**. Legacy names may remain in literal paths/checkpoint IDs only for reproducibility, not as a v1/v2 method narrative.

## 1. Central scientific thesis

ART investigates whether **retrieval timing can be learned as an emergent consequence of optimizing a trade-off between LM Loss and retrieval regularization**, without direct labels for each retrieval decision or reinforcement learning. ART is **methodology-oriented research**, not a claim to SOTA F1 or globally optimal timing.

In a simplified one-score illustration, `L(d) = L_LM(d) + lambda*d`, so `dL/dd = dL_LM/dd + lambda`. If the local benefit of retrieved evidence outweighs the penalty (`dL_LM/dd < -lambda`), gradient descent locally favors a higher mixing/decision score; otherwise the regularizer favors lowering it. This is an **intuition**, not a guarantee for every score or a direct scalar update: actual gradients update Decision Head parameters through the differentiable Attention Mixing path during replay. Verify the actual objective and regularizer implementation (including termination positions) before writing formal equations.

**Separate evidence levels:** H1/H2 test foundational assumptions about LM Loss and Attention Mixing / gradient signals under experimental conditions; they do not establish end-to-end ART performance. ART benchmarks evaluate the trained system and its retrieval behavior; they do not isolate causal impact of timing from fine-tuning or prompting differences.

**Do not conflate:** forced probe mixing coefficient `d`, inference retrieval threshold `tau`, retrieval-score regularizer, discrete retrieval count, and total computational cost.

## 2. Global editorial rules (U-001, U-002, U-015)

- Academic Vietnamese with natural English AI/ML terms (Attention, KV Cache, Decoder, Embedding, Dense Retriever, Attention Mixing, Decision Head, Collect–Replay, etc.); do not mechanically translate established terms. Explain first occurrences as useful; be consistent.
- Restore the **verbatim official HCMUT declaration** from the source template, changing only required individual pronouns/personal fields (student Nguyễn Phan Tuấn Duy; supervisor PGS. TS. Quản Thành Thơ) and confirmed date. Never reconstruct declaration wording from memory.
- Tight, coherent reasoning. Do not pad pages, repeat disclaimers, invent mechanisms, infer numerical compute costs, or embellish raw model answers.
- When helpful, produce **standalone TikZ/LaTeX diagram sources**, compile vector PDF, include with `\includegraphics`, and retain reproducible sources. Figures must be readable, source-faithful, captioned and referenced; avoid decorative figures.
- Respect all template-mandated components, headings, front matter, appendices, citation formatting.

## 3. Front matter and Chapter 1 (U-003–U-007)

**Abstract:** three concise paragraphs (~200–250 Vietnamese words as a guideline): problem/objective; ART mechanism and H1/H2; benchmark results. See the **approved reference draft in U-003 of the full log**. Use accurate data and no unjustified causal attribution.

**Chapter 1:** a connected research narrative, template permitting: 1.1 motivation/gap (including qualitative latency/compute implications of possible RAG steps such as rewriting, embedding, vector search, reranking; do not assert ART implements/measures all); 1.2 problem with **actual MuSiQue baseline-only** example (No-RAG and Vanilla RAG fail; ART success revealed in later chapters); 1.3 objectives/research questions (H1/H2, ART method, benchmark evaluation); 1.4 actual scope/contributions, including empirical H1/H2 investigation, method, implementation and evaluation. Avoid duplication between 1.3 and 1.4. Template wins on titles.

**Single MuSiQue case:** find real demo-v3 artifacts by searching repository; **do not assume `result/demo-v3` exists**. Prefer same question with No-RAG incorrect, Vanilla RAG incorrect, ART correct and meaningful retrieval. Match IDs, ground truth, exact raw generations, retrieved passages and trajectory. Reproduce **verbatim**; if abbreviated, mark omission explicitly. Chapter 1 presents baseline failures only; Chapter 3/4 may explain mechanism; Chapter 5.3.1 and Appendix A show ART and comparative evidence. If no valid case, **stop that part and report blocker**.

## 4. Chapter 2 — Foundations and Related Work (U-008, U-013)

Selective, mechanism-focused comparison: Vanilla RAG, IRCoT, Self-RAG, R-Search; explain learning signal, supervision, retrieval decisions and optimization rather than compare cross-paper F1 as a leaderboard. Clarify research gap for indirect LM-Loss-driven retrieval timing without direct decision labels or RL; do not claim first-ever approach, universal inability of prior work, or superiority without controlled evidence. Use accurate official citations and necessary foundational model/dataset references. Remove Adaptive-RAG from comparative discussion per U-011 unless it has a separately justified essential citation.

## 5. Chapter 3 — Methodology (U-009, R-003, R-004)

Explain final ART method only: hypotheses as foundations; Decision Head; differentiable Attention Mixing; gradient path; LM Loss + retrieval regularization trade-off; Collect–Replay and fixed discrete retrieval actions during replay; inference threshold and retrieval workflow. Check notation and mathematics against implementation. Do not imply gradients pass through discrete retrieval/FAISS or that local H2 gradient proves an optimal threshold. Verify teacher Qwen3-8B/four few-shot examples, actual dataset construction, loss/config, model variants and conditions against code. Omit irrelevant development-version history; retain differences affecting reproducibility.

## 6. Chapter 4 — System design and implementation (U-010)

Show how ART is actually built and run: architecture, training Collect–Replay, Attention Mixing branches and gradient flow, Decision Head, autoregressive inference and KV Cache if verified. Prefer a few useful source-faithful vector diagrams; no mandatory count. Separate Ch3 methodology, Ch4 implementation, Ch5 hypothesis verification/results. Do not move H1/H2 experiment findings into Ch4 or invent implementation details.

## 7. Chapter 5 — Experiments and discussion (U-011, R-001–R-006)

**5.1:** concise purpose → setup → measurement → limitations. H1/H2 use a simplified ART/Attention Mixing probe **only to the extent code confirms**; describe differences from full ART. Keep reproducibility details in appendix if lengthy. Explain evaluation protocol and Acc/EM/F1 **by checking ART evaluation scripts against R-Search paper/code**; preserve current metric definitions and numbers if verified, clarify that Acc may be substring matching rather than EM; cite R-Search only after verifying.

**5.2:** H1/H2 each as hypothesis → observation → interpretation → warranted conclusion. Keep `d` separate from `tau`. Do not claim H1/H2 alone prove the trained system or global optimum.

**5.3:** preserve measured benchmark and retrieval-count results; describe retrieval count as usage behavior, **not** latency/GPU-hours/FLOPs/total compute. The **primary** ART comparison uses a common `tau=0.4297` on all three benchmarks. Threshold sweep and per-benchmark best-observed F1 are **exploratory**, not an independent held-out test of tuned thresholds. Do not call `tau=0.4297` a proven optimum. In 5.3.1 use the verified single MuSiQue example with exact No-RAG/Vanilla/ART raw outputs and relevant passages/trajectory, concise analysis only.

**Recorded values to verify, never silently change:** 500 matched questions per benchmark. At common `tau=0.4297`: F1 MuSiQue No-RAG 5.61 / Vanilla 10.25 / ART 31.17; HotpotQA 20.19 / 39.61 / 48.24; 2WikiMultiHopQA 27.58 / 37.81 / 46.30. Exploratory sweep maxima: MuSiQue `tau=.60`, F1 32.08; HotpotQA `tau=.50`, F1 49.68; 2Wiki `tau=.4297`, F1 46.30. Use source artifacts for all additional numbers, counts and table columns. If conflict, flag it rather than overwrite.

**5.4:** mechanism-oriented, fair comparison; exclude Adaptive-RAG from comparison. Do not compare unlike checkpoints/prompts as controlled evidence. ART fine-tuned checkpoint vs No-RAG/Vanilla base checkpoints, and possible prompt/extraction/retrieval differences: state **system-level** F1 gains, not all gains caused by timing. Controlled ablation is a possible future task, **not required now**. Discuss computational implications only where documented; no unsupported training-cost rankings.

## 8. Chapter 6 — Conclusion (U-012, R-008)

Confidently summarize gap, conditional H1/H2 evidence, ART learning mechanism, implementation, three-benchmark system outcomes and retrieval behavior. Keep claims aligned with evidence: no global timing optimum, no full causal attribution of F1 to timing, no unmeasured total-cost savings. Limitations should be specific, short and fair. Future directions: exploratory tool selection and joint retriever–generator training; neither is an established improvement or commitment. Keep Abstract, Results and Conclusion consistent.

## 9. References and appendices (U-013, U-014)

References: only sources actually relevant and cited; verify metadata, official papers and BibTeX; include appropriate Self-RAG, IRCoT, R-Search and foundational model/dataset references. No citation padding.

Appendix A: verified verbatim MuSiQue question, GT, raw outputs, passages and trajectory where useful, with clearly marked omissions and short analysis. Appendix B.1: relevant artifacts/reproducibility, accurately reflecting access state; links known to author: `https://github.com/nguyenphantuanduy/adaptive-retrieval-timing.git`, `https://huggingface.co/adaptive-retrieval-timing`, `https://huggingface.co/adaptive-retrieval-timing/ART-REVIVE`. **Author stated these are not yet public**; do not claim public availability. `ART-REVIVE` may remain as literal checkpoint ID. Appendix B.2: explain what training-load count includes/excludes; do not equate trajectories/steps/tokens with total compute.

## 10. Explicit non-goals

- No new training, benchmark runs, threshold tuning or ablations unless author separately authorizes.
- No independent R-007 timing analysis: **CLOSED / NO CHANGE**; rely on existing MuSiQue case requirement.
- No fabricated or reconstructed model outputs, retrieved passages, results, citations, code paths, diagram arrows, training costs or repository-publication status.
- No method history narrative (v1/v2), no broad SOTA race, no changes to raw result files.

## 11. Implementation sequencing

1. Inspect HCMUT template, report TeX project and actual code/results; inventory verified evidence and blockers.
2. Make an edit plan mapped to U/R IDs; preserve all report files outside intended scope.
3. Edit front matter and chapters in a consistent narrative, then appendices and bibliography; create source-faithful diagrams.
4. Build PDF using the repository's actual build workflow; fix TeX errors and check figure placement, references, equations, citations and readability.
5. Produce change/verification report with U/R mapping, source paths and unresolved blockers. Never mark a requirement verified without inspecting its evidence.
