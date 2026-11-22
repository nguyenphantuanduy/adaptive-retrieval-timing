# Phase 1 hypothesis verification

This chapter distinguishes the implemented probe, the saved aggregate measurements, and conclusions that those measurements can support. The probes intervene on an externally supplied attention coefficient. They do not exercise REVIVE's trained retrieval head or its search loop.

## Design shared by H1 and H2

The current drivers load the Qwen2.5-3B-Instruct tokenizer and the Hugging Face `ART-Attention-Mixing-Probe` model with custom code and BF16. The local [probe forward](../shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/model.py#L285) accepts the document mask and decision directly and computes causal-LM loss when labels are supplied. The data loader takes the 2-, 3-, and 4-hop validation splits of `teacher-generated-hypothesis-validation`. Current code identifies expected inputs, while the saved CSVs do not establish immutable model or dataset revisions. [H1 driver](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/run_lm_loss_evaluation.py#L1039) · [H2 driver](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/run_gradient_evaluation.py#L1426) · [Dataset loader](../shared/MuSiQue/load_hypothesis_validation.py#L19)

1. Partition paragraphs by the dataset's `is_supporting` flag. A module-local RNG seeded at 42 samples up to three from each partition once per question. For each feasible positive:negative ratio (3:0, 2:1, 1:2, 0:3), take prefixes of those sampled pools and place positives before negatives. Each condition has three documents. A question lacking sufficient positives or negatives skips that ratio. The RNG sequence depends on traversal order. [build_lm_loss_samples.py:14–28, 104–317](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/build_lm_loss_samples.py#L104)
2. Render the chat question and append a single observation made from titled passages. Tokenize prefix and observation separately and concatenate IDs. Mark the whole observation span with additive negative infinity in the document-key mask. Duplicate the input across the 33 settings \(d=i/32\), \(i=0,\ldots,32\). Append newline, the teacher-generated continuation and EOS, and mask prefix plus observation labels with `-100`. [model_input.py:92–413, 416–759](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/model_input.py#L92)
3. Within one question and document ratio, only \(d\) varies. Across ratios, passage identity, relevance, ordering and lengths can vary. Observations are supplied from dataset annotations and distractors, so there is no retrieval quality or scheduling intervention.

The summary scripts pooled available sample-level observations within each condition, and their Total rows pool across applicable hops. The diagnostic `check_nan.py` scripts inspect NaNs. The saved aggregates each have 528 grid rows and 495 populated rows. The 33 missing rows are exactly the 2-hop 3:0 condition. Across grid points, each available condition has a constant reported count within hop: 530 for 2-hop, 322 for 3-hop and 172 for 4-hop. Total 3:0 uses 494 examples, while other Total ratios use 1,024. Cross-ratio comparisons with 3:0 therefore have a different question mix. [H1 plotting and pooling](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/plot_lm_loss_vs_decision_score.py#L74) · [H2 plotting and pooling](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/plot_gradient_vs_decision_score.py#L79)

## H1 — Attention modulation

The [original H1 specification](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/README.md) proposes coherent-language preservation, ordered control of document information, and downstream answer improvement when more useful evidence is accessible. A motivation for the probe is that interpolation gives direct control over document-key attention mass at fixed projections. Through multiple decoder layers, the effect on outputs can depend on evidence quality and hidden-state feedback.

The H1 driver calls `model.eval()`, feeds four coefficients per forward under inference mode, retries an out-of-memory batch one coefficient at a time, and marks persistent failures NaN. The dependent variable is per-sample mean token cross-entropy over the newline, teacher reasoning/answer and EOS. The helper compares logits at \(t-1\) with label \(t\) and averages valid positions. It does not generate answers, score exact match, or rate coherence. [H1 driver:160–279, 417–530, 600–693](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/run_lm_loss_evaluation.py#L160) · [Loss helper](../shared/utils/compute_per_sample_lm_loss.py#L5)

**Observed.** Saved pooled means are shown below. The best \(d\) is the grid point with lowest saved mean. Grid labels in the CSV are rounded.

| Positive:negative | Loss at 0 | At 0.5 | At 1 | Lowest mean (at \(d\)) |
|---|---:|---:|---:|---:|
| 3:0 | 1.2875 | 0.9832 | 0.8940 | 0.8830 (0.7500) |
| 2:1 | 1.3164 | 1.0344 | 0.9735 | 0.9495 (0.7188) |
| 1:2 | 1.3159 | 1.1582 | 1.1798 | 1.1295 (0.6562) |
| 0:3 | 1.3160 | 1.2952 | 1.3943 | 1.2753 (0.2500) |

Source: [lm_loss_summary.csv](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/plots/lm_loss_summary.csv).

**Interpretation.** The evidence supports controllability of teacher-target likelihood and a dependence on evidence composition. Supporting-document conditions improve from \(d=0\) to \(d=1\) in the pooled curves. Every curve has an interior optimum and increases at some adjacent steps, so strictly monotonic target-loss improvement is contradicted. The all-distractor condition can help at low settings and hurts at maximum mixing relative to zero. H1.1 coherence preservation and H1.3 free-generation answer accuracy remain inconclusive from this CSV. H1.2 is supported at the level of fixed-state direct attention and observed likelihood sensitivity. The end-to-end amount of accessible information was not directly measured.

## H2 — Gradient direction

The operational H2 question, reconstructed from the driver and cost-analysis notebooks, is whether the language-model loss gives a locally useful signal for increasing the document coefficient with supporting evidence and reducing it with distractors. For a directly optimized coefficient, the sign of \(g(d)=\partial L_{\rm LM}/\partial d\) sets the direction of a sufficiently small gradient step. A negative sign favors increasing \(d\); a positive sign favors decreasing it. This motivates learning the gate from LM gradients, but a head that predicts from pre-retrieval state must generalize beyond a gradient measured with known inserted documents.

H2 builds a 33-row input dataset for each question and ratio. A temporary forward wrapper enables gradients on the exact decision tensor supplied by Trainer, maps its actual numeric value to the grid index and registers a hook. The hook records the detached mean of the gradient tensor as a scalar. Trainer uses batch one, accumulation one, learning rate zero, non-reentrant checkpointing, and no saving or evaluation. The CLI freezes model parameters. Trainer's training loop supplies backward and can set training mode despite the preceding `model.eval()`; it is therefore a different execution path from the H1 inference CE helper. If a CUDA OOM interrupts a sweep, missing indices are retried individually and missing results become NaN. [H2 driver:175–271, 392–600, 663–824, 1477–1504](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/run_gradient_evaluation.py#L175)

**Observed.** These are pooled gradients from the saved summary.

| Positive:negative | Gradient at 0 | At 0.5 | At 1 | Characteristic extremum |
|---|---:|---:|---:|---|
| 3:0 | -0.2864 | -1.1295 | +0.0430 | Minimum -2.4784 at 0.4375 |
| 2:1 | -0.3914 | -1.0241 | +0.0694 | Minimum -2.1233 at 0.4375 |
| 1:2 | -0.1739 | -0.5211 | +0.1266 | Minimum -1.2356 at 0.4375 |
| 0:3 | -0.3636 | +0.0017 | +0.1763 | Maximum +0.4016 at 0.3750 |

Source: [gradient_summary.csv](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_summary.csv).

**Interpretation.** H2 is supported conditionally. At \(d=0.5\), the pooled supporting conditions strongly favor increasing the coefficient, while the all-distractor gradient is near zero and positive. The all-distractor condition is negative at zero, and every condition is positive at one. Per-hop all-distractor signs also differ at 0.5. A universal relevance classifier based on the gradient sign is contradicted by these aggregates. The summaries do not report the fraction of questions with the expected sign, uncertainty intervals, or gradient magnitude matched to H1's measured numerical slope. Distinct forward modes, loss implementations and unpinned run revisions limit treating the two aggregates as measurements of exactly the same numerical function.

## Heuristic retrieval threshold and cost

The saved H2 curves place the all-distractor maximum at \(d=0.375\) and each of the three supporting-condition minima at \(d=0.4375\). These features motivate examining the transition region around \(d=0.4\)?\(0.44\). The [simplified notebook](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/analyze_threshold_and_cost_simplified.ipynb) uses **\(\tau=0.4297\) as a fixed heuristic hyperparameter** within this region. Here \(d\) is the attention-mixing coefficient varied in H2, while \(\tau\) is the cutoff applied to REVIVE's predicted score when deciding whether to retrieve. The H2 curves do not establish an optimal retrieval cutoff.

The earlier notebook averaged the four characteristic extrema and obtained \(0.421875\). The project author confirmed that historical REVIVE used \(0.4297\) because of a numerical entry mistake. The revised notebook retains the historical setting and uses the observed gradient region as qualitative motivation.

For a linear retrieval penalty \(\lambda d\), the derivative becomes \(g(d)+\lambda\). The notebook displays \(-\lambda\) for \(\lambda\in\{0,0.05,0.1\}\) and marks crossings by linear interpolation between adjacent observed points at or above \(\tau\). These crossings describe how the penalty changes the local gradient direction for supplied documents. They do not measure the downstream effect of changing the hard retrieval gate.

## Limits on report claims

- Dataset support labels supply relevance categories. Distractors may still contain related information, and three-document truncation of the supporting set can omit a necessary hop.
- Supporting passages appear before distractors. The design does not isolate relevance from order and length or control comparisons across different ratio populations.
- Teacher-forced likelihood emphasizes the whole continuation, including reasoning and formatting. It can improve independently of decoded answer accuracy and has no direct coherence measure.
- End-to-end gate learning, evidence recall, calibration and retrieval efficiency require additional measurements. Read the saved ART inference aggregates separately in [EXPERIMENTS.md](EXPERIMENTS.md).
