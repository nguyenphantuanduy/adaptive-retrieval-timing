# Issue log

| Severity | Finding | Resolution/status |
|---|---|---|
| Build blocker | Redirected native stderr in Windows PowerShell converted MiKTeX's update notice to a terminating PowerShell error. | Ran build script directly, preserving the subprocess exit status. |
| Build blocker | Report equations used legacy `\\rm`, which memoir rejects. | Replaced with `\\mathrm{LM}` in report chapters. |
| Scientific | First draft described opposite H2 signs throughout the entire neighborhood; the all-distractor series crosses zero near the reference line. | Described observed extrema, regime-dependent signs and positive terminal gradients from the current notebook/CSV. |
| Layout | Appendix B reproduced a 55.44397 pt overflow from an unbreakable path; bibliography had conspicuous spacing. | Used breakable paths, shorter sentence boundaries and ragged-right bibliography. |
| Layout | Workflow diagram floated alone after the chapter text; the external table split a discussion paragraph. | Moved workflow source next to collect/replay with top placement and started the separate external context on a new page. |
| Layout | The stock template baseline had an overfull work table and duplicate page.1 destination. | Used a single-student flexible-width workload table and disabled the page anchor for the cover in the report driver. Official class unchanged. |
| Residual | MiKTeX update notice, memoir compatibility/deprecation, and Babel Vietnamese hyphenation fallback. | Existing environment/class diagnostics. Visual QA validates Vietnamese glyphs. Do not change the official class to suppress notices. |
| Residual | Official submission date and optional council/reviewer metadata are unavailable. | Uses template's build-month default, with optional fields omitted. |
| Residual | Historical immutable checkpoint/index revisions and measured training/inference costs are unavailable. | Report identifies the evidence boundary and uses workload counts as proxies. |

The final full-size H2 review confirmed the report-only legend repair and staggered crossing labels. The official notebook and CSV were preserved. Appendix B retains one benign underfull line (badness 1087); no clipping or overflow in the final inspection.

## Naming revision
Resolved public-facing internal naming and abbreviated Vanilla labels. Kept literal reproducibility paths. Visual inspection found no new layout defects. Existing MiKTeX, memoir and Babel warnings and the Appendix B underfull line remain.
