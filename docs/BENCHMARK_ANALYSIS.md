# ART benchmark analysis

## Scope and reading conventions

This snapshot contains **21 benchmark metric files** in `REVIVE_training/metrics/` and `art-backup/`, plus two supplemental summaries in `temp_model/`. The live metrics tree contains all six baseline summaries, including HotpotQA Vanilla RAG. The ART archive has four demo-2 and eleven demo-3 metric files. All individual summaries are JSON objects in `.json` files. The nine JSON members in [demo-3 metrics.zip](../art-backup/demo-3/metrics.zip) match their extracted counterparts and are counted once. The two new 2WikiMultiHopQA summaries were scored from saved predictions with the existing ART evaluator. TensorBoard training events are outside this benchmark inventory.

For corresponding archived ART filenames ending in `_art_top5_metrics.json`, **τ = 0.4297** is the historical default supplied by the author. The `top5` suffix specifies retrieval top-k, independently of τ. Explicit `threshold50`, `threshold60`, and `threshold70` files denote τ = 0.50, 0.60, and 0.70. These files do not encode top-k, so their top-k is left unstated in the table. The author previously confirmed that demo-3 threshold runs used the same checkpoint and changed only inference threshold. The archive does not independently record the complete run configuration. Plain, demonstration and temporary-model filenames do not independently establish their threshold. This analysis uses demo-3 top-5 as the primary default on all three datasets. Demo-2 provides a separate earlier MuSiQue checkpoint.

## Complete benchmark

Scores are percentages. Source IDs resolve to the exact files below.

| Dataset | Method | τ | Retrieval k | N | Acc | F1 | EM | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MuSiQue | No-RAG | — | — | 500 | 0.80 | 5.61 | 0.60 | M05 |
| MuSiQue | Vanilla RAG | — | 5 | 500 | 6.20 | 10.25 | 4.80 | M06 |
| MuSiQue | ART demo-3 | 0.4297 | 5 | 500 | 25.80 | 31.17 | 21.60 | M12 |
| MuSiQue | ART demo-3 | 0.50 | Unstated | 500 | 24.00 | 30.27 | 20.80 | M13 |
| MuSiQue | ART demo-3 | 0.60 | Unstated | 500 | 26.00 | 32.08 | 22.80 | M14 |
| MuSiQue | ART demo-3 | 0.70 | Unstated | 500 | 22.60 | 29.21 | 19.80 | M15 |
| HotpotQA | No-RAG | — | — | 500 | 14.00 | 20.19 | 14.20 | M03 |
| HotpotQA | Vanilla RAG | — | 5 | 500 | 33.20 | 39.61 | 30.20 | M04 |
| HotpotQA | ART demo-3 | 0.4297 | 5 | 500 | 42.20 | 48.24 | 37.20 | M08 |
| HotpotQA | ART demo-3 | 0.50 | Unstated | 500 | 43.80 | 49.68 | 39.20 | M09 |
| HotpotQA | ART demo-3 | 0.60 | Unstated | 500 | 40.00 | 46.87 | 36.40 | M10 |
| 2WikiMultiHopQA | No-RAG | — | — | 500 | 24.80 | 27.58 | 24.40 | M01 |
| 2WikiMultiHopQA | Vanilla RAG | — | 5 | 500 | 34.80 | 37.81 | 32.20 | M02 |
| 2WikiMultiHopQA | ART demo-3 | 0.4297 | 5 | 500 | 48.20 | 46.30 | 37.00 | M07 |
| 2WikiMultiHopQA | ART demo-3 | 0.50 | Unstated | 500 | 46.40 | 44.75 | 35.60 | M22 |
| 2WikiMultiHopQA | ART demo-3 | 0.60 | Unstated | 500 | 44.40 | 44.34 | 36.00 | M23 |

### Additional archived configurations

| Dataset | Method | τ | Retrieval k | N | Acc | F1 | EM | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MuSiQue | ART demo-3 | Unverified | Unstated | 500 | 22.80 | 28.46 | 19.60 | M11 |
| MuSiQue | ART demo-2 | Unverified | Unstated | 500 | 18.00 | 25.04 | 16.80 | M16 |
| MuSiQue | ART demo-2 | 0.4297 | 5 | 500 | 20.00 | 26.33 | 17.60 | M17 |
| MuSiQue | ART demo-2, demonstrations | Unverified | Unstated | 500 | 15.20 | 22.73 | 13.20 | M18 |
| MuSiQue | ART demo-2, demonstrations | Unverified | 5 | 500 | 11.40 | 17.13 | 9.20 | M19 |
| MuSiQue | Temporary model | Unverified | Unstated | 500 | 12.40 | 18.94 | 11.00 | M20 |
| MuSiQue | Temporary model, demonstrations | Unverified | Unstated | 500 | 11.80 | 19.47 | 11.00 | M21 |

## Baseline comparisons

Differences are percentage points, computed directly from displayed file precision. For each metric key, subtract the second listed source's `metrics[key]` from the first. Positive means the first run scored higher. No normalization or rescaling of the stored values was applied.

The baseline comparisons still use the historical default ART threshold for all three datasets. The newly scored variants appear in the threshold comparisons below.

| Dataset | Comparison | Δ Acc | Δ F1 | Δ EM | Sources |
|---|---|---:|---:|---:|---|
| MuSiQue | Vanilla RAG − No-RAG | +5.40 | +4.64 | +4.20 | M06 − M05 |
| MuSiQue | ART default − Vanilla RAG | +19.60 | +20.92 | +16.80 | M12 − M06 |
| MuSiQue | ART default − No-RAG | +25.00 | +25.56 | +21.00 | M12 − M05 |
| HotpotQA | Vanilla RAG − No-RAG | +19.20 | +19.42 | +16.00 | M04 − M03 |
| HotpotQA | ART default − Vanilla RAG | +9.00 | +8.63 | +7.00 | M08 − M04 |
| HotpotQA | ART default − No-RAG | +28.20 | +28.05 | +23.00 | M08 − M03 |
| 2WikiMultiHopQA | Vanilla RAG − No-RAG | +10.00 | +10.23 | +7.80 | M02 − M01 |
| 2WikiMultiHopQA | ART default − Vanilla RAG | +13.40 | +8.49 | +4.80 | M07 − M02 |
| 2WikiMultiHopQA | ART default − No-RAG | +23.40 | +18.72 | +12.60 | M07 − M01 |

## Demo-3 threshold comparisons

| Dataset | τ vs historical default | Δ Acc | Δ F1 | Δ EM | Sources |
|---|---:|---:|---:|---:|---|
| MuSiQue | 0.50 | -1.80 | -0.90 | -0.80 | M13 − M12 |
| MuSiQue | 0.60 | +0.20 | +0.91 | +1.20 | M14 − M12 |
| MuSiQue | 0.70 | -3.20 | -1.96 | -1.80 | M15 − M12 |
| HotpotQA | 0.50 | +1.60 | +1.44 | +2.00 | M09 − M08 |
| HotpotQA | 0.60 | -2.20 | -1.37 | -0.80 | M10 − M08 |
| 2WikiMultiHopQA | 0.50 | -1.80 | -1.55 | -1.40 | M22 − M07 |
| 2WikiMultiHopQA | 0.60 | -3.80 | -1.96 | -1.00 | M23 − M07 |

- **MuSiQue:** τ = 0.60 has the highest observed demo-3 Acc/F1/EM at **26.00 / 32.08 / 22.80**. It exceeds the default by 0.20 / 0.91 / 1.20 points.
- **HotpotQA:** τ = 0.50 has the highest observed demo-3 Acc/F1/EM at **43.80 / 49.68 / 39.20**. It exceeds the default by 1.60 / 1.44 / 2.00 points. No τ = 0.70 metric was found.
- **2WikiMultiHopQA:** The historical default τ = 0.4297 leads on all three metrics at **48.20 / 46.30 / 37.00**. Among the two higher thresholds, τ = 0.50 has higher Acc and F1, while τ = 0.60 has higher EM. Valid answer-tag counts are 488 at the default, 486 at τ = 0.50 and 492 at τ = 0.60. No τ = 0.70 summary was found.

These are observed maxima over available files. The aggregates provide no statistical significance test or evidence of a globally optimal threshold.

## Compatibility and interpretation

- **Observed.** Vanilla RAG scores above No-RAG on all three metrics for every dataset. ART default scores above both baselines on all three datasets. Against Vanilla RAG, the largest F1 and EM increases are on MuSiQue (+20.92 and +16.80 points). The largest Acc increase is also on MuSiQue (+19.60 points).
- **Metric scale.** All 23 distinct summaries have 500 examples, numeric Acc/F1/EM in [0,100], and Acc and EM consistent with count increments of 0.20 percentage points. There are no sample-count mismatches. Equal counts do not verify paired IDs, ordering, alias sets, or identical dataset revisions.
- **Scoring definitions.** The two local scorers share [Acc, F1, and EM functions](../REVIVE_training/utils.py). Acc tests case-insensitive raw substring containment. EM and F1 normalize punctuation, articles and whitespace and choose the best score across gold aliases before averaging. HotpotQA No-RAG reports EM 14.20 above Acc 14.00. These definitions allow that ordering, for example when punctuation differs, and the aggregate does not identify the responsible rows.
- **Prediction protocol.** The [ART scorer](../REVIVE_training/evaluate_art_pure.py#L115) selects the last complete answer-tag pair or falls back to the stripped response. The [current baseline scorer](../REVIVE_training/evaluate_baseline.py#L11) removes three exact Qwen text-control tokens and scores the entire remaining completion. The archive records tagged-answer counts for ART runs. Baseline JSON metrics do not record scorer revision, so the exact cleaning code used to produce each saved baseline is unverified. Formatting differences can affect F1 and EM even with common metric functions.
- **Model and search.** The [baseline generator](../REVIVE_training/baseline_generation.py#L12) loads original Qwen2.5-3B-Instruct. The [ART top-5 generator](../REVIVE_training/evaluate_art_pure_generation_top5.py#L27) names ART-REVIVE, and the author confirmed that the archive's two demo checkpoints came from their respective fine-tuning scripts. Exact historical checkpoint and index revisions are unverified. No-RAG lacks retrieval, Vanilla RAG retrieves once with the question, and ART can retrieve iteratively during generation. The prompts and evidence insertion differ. [Baseline protocol](RAG_BASELINE_IMPLEMENTATION.md) · [ART training provenance](TRAINING.md)
- **Hypotheses.** Better evidence availability, tuned generation behavior, answer formatting, and adaptive retrieval may all contribute to the observed differences. The aggregate comparisons do not isolate these factors. A same-checkpoint fixed-retrieval control and an aligned scoring protocol would help test the contribution of adaptive timing. The published R-Search benchmark is separate from these local artifacts.

## Retrieval Cost and Quality-Efficiency Tradeoffs

### Counting method and reproducibility

The [analysis script](../REVIVE_training/analyze_retrieval_cost.py) pairs each demo-3 prediction with the matching metric by dataset directory and the evaluator's filename transformation. Run `python -m REVIVE_training.analyze_retrieval_cost` from the repository root. It produces the [summary JSON](../REVIVE_training/metrics/retrieval_cost/retrieval_cost_summary.json) and [per-sample CSV](../REVIVE_training/metrics/retrieval_cost/retrieval_cost_per_sample.csv). These include 11 configurations and 5,500 rows. Source paths and benchmark metrics are included in the JSON. The CSV preserves prediction order within each file and leaves missing IDs blank. The JSON records the full count distribution, population standard deviation and linearly interpolated P90/P95.

The inspected prediction rows store an ordered string list in `pure_art_queries`. The generation wrapper resets `model.query` per sample and copies it into the prediction after generation returns. The local inference implementation appends a query immediately before calling the retriever, after the decision threshold and forbidden-marker checks. Count each list entry, including repeated identical queries. It represents a logged search-call event on that code path. Because the append precedes the call, the files carry no per-call success/failure status and the historical remote model revisions are unpinned, so the count is a proxy for executed retrievals in completed predictions rather than independent execution verification. Exceptions that abort a sample and skipped decisions are not recoverable as retrieval records. A returned empty hit list could still represent a search. [Generator logging](../REVIVE_training/evaluate_art_pure_generation_top5.py#L253) · [Inference gate and search](../shared/ART_Models/ART_REVIVE/model.py#L1335)

The analysis ignores generated search-like markup in `pure_art_answer`. It also avoids counting `pure_art_retrieval_scores`, which records decision candidates before the checks, and `pure_art_passages`, which is a flat list of returned hits. The wrapper can emit an empty list if the loaded model lacks the query attribute. Across this archive, every query list passed validation, no query count exceeded its score-list length, and no zero-query row had passages. For nonzero-query rows, the aggregate passage-to-query ratio was exactly five in every known-threshold configuration and three in the plain MuSiQue run. That consistency supports the log interpretation but cannot verify individual call outcomes. Ordered ID, question and gold-answer digests match within each dataset across these ART prediction files. All counts match the metric denominators. Missing or malformed query fields, empty prediction files and sample-count mismatches are reported and excluded from the joined table.

### Observed quality and retrieval cost

Retrieval rate is the percentage of samples with at least one logged query. Pareto status is computed within each dataset's known-threshold sweep. A configuration is dominated if another reaches at least its F1 with no more average queries and improves at least one of the two. The plain MuSiQue configuration is included for completeness and is also dominated when all available same-dataset runs are considered. ART quality scores below are the saved values and were not recomputed.

| Dataset | Threshold τ | Acc | F1 | EM | Avg queries | Retrieval rate | Total queries | Pareto-efficient |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MuSiQue | 0.4297 | 25.80 | 31.17 | 21.60 | 4.792 | 100.0% | 2,396 | No |
| MuSiQue | 0.50 | 24.00 | 30.27 | 20.80 | 3.594 | 100.0% | 1,797 | No |
| MuSiQue | 0.60 | 26.00 | 32.08 | 22.80 | 2.588 | 100.0% | 1,294 | Yes |
| MuSiQue | 0.70 | 22.60 | 29.21 | 19.80 | 2.126 | 99.2% | 1,063 | Yes |
| MuSiQue | Unverified (plain) | 22.80 | 28.46 | 19.60 | 4.438 | 100.0% | 2,219 | — |
| HotpotQA | 0.4297 | 42.20 | 48.24 | 37.20 | 4.070 | 100.0% | 2,035 | No |
| HotpotQA | 0.50 | 43.80 | 49.68 | 39.20 | 3.166 | 99.8% | 1,583 | Yes |
| HotpotQA | 0.60 | 40.00 | 46.87 | 36.40 | 2.278 | 99.2% | 1,139 | Yes |
| 2WikiMultiHopQA | 0.4297 | 48.20 | 46.30 | 37.00 | 4.012 | 100.0% | 2,006 | Yes |
| 2WikiMultiHopQA | 0.50 | 46.40 | 44.75 | 35.60 | 2.930 | 100.0% | 1,465 | Yes |
| 2WikiMultiHopQA | 0.60 | 44.40 | 44.34 | 36.00 | 2.126 | 99.4% | 1,063 | Yes |

| Dataset | Threshold τ | N | Median | Population SD | Min–max | P90 / P95 | Zero queries | ≥1 query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MuSiQue | 0.4297 | 500 | 5 | 1.653 | 1–11 | 7.00 / 8.00 | 0 (0.0%) | 500 |
| MuSiQue | 0.50 | 500 | 3 | 1.345 | 1–9 | 5.00 / 6.00 | 0 (0.0%) | 500 |
| MuSiQue | 0.60 | 500 | 2 | 0.935 | 1–7 | 4.00 / 4.00 | 0 (0.0%) | 500 |
| MuSiQue | 0.70 | 500 | 2 | 0.700 | 0–5 | 3.00 / 3.00 | 4 (0.8%) | 496 |
| MuSiQue | Unverified (plain) | 500 | 4 | 1.591 | 1–18 | 6.00 / 7.00 | 0 (0.0%) | 500 |
| HotpotQA | 0.4297 | 500 | 4 | 1.393 | 2–13 | 6.00 / 6.00 | 0 (0.0%) | 500 |
| HotpotQA | 0.50 | 500 | 3 | 0.958 | 0–8 | 4.00 / 5.00 | 1 (0.2%) | 499 |
| HotpotQA | 0.60 | 500 | 2 | 0.688 | 0–8 | 3.00 / 3.00 | 4 (0.8%) | 496 |
| 2WikiMultiHopQA | 0.4297 | 500 | 4 | 1.367 | 1–10 | 6.00 / 6.00 | 0 (0.0%) | 500 |
| 2WikiMultiHopQA | 0.50 | 500 | 3 | 1.083 | 1–9 | 4.00 / 5.00 | 0 (0.0%) | 500 |
| 2WikiMultiHopQA | 0.60 | 500 | 2 | 0.674 | 0–5 | 3.00 / 3.00 | 3 (0.6%) | 497 |

### Threshold sensitivity and tradeoffs

- **MuSiQue.** Average queries fell from 4.792 at the default to 3.594, 2.588 and 2.126 at higher thresholds. At τ = 0.60, the archive shows 1,102 fewer queries than default (46.0% fewer), while F1 rises by 0.91 points and EM by 1.20 points. At τ = 0.70, cost falls further and F1 drops by 2.87 points relative to 0.60. **0.60 and 0.70** form the observed frontier.
- **HotpotQA.** Average queries fell from 4.070 to 3.166 to 2.278. At τ = 0.50, there are 452 fewer queries than default (22.2% fewer), with F1 up 1.44 points and EM up 2.00 points. At τ = 0.60, cost is lower and F1 is 2.81 points below 0.50. **0.50 and 0.60** form the observed frontier.
- **2WikiMultiHopQA.** Average queries fell from 4.012 to 2.930 to 2.126. Against default, τ = 0.50 saves 541 queries (27.0%) for a 1.55-point F1 decrease, and τ = 0.60 saves 943 (47.0%) for a 1.96-point decrease. EM at 0.60 is higher than at 0.50 while both fall below the default. **0.4297, 0.50 and 0.60** form the observed frontier.

Mean query counts decrease with higher thresholds at every observed step in these datasets. Retrieval rates remain at or above 99.2%, so the change mainly reflects fewer repeated retrieval opportunities per sample rather than a large increase in samples skipping retrieval entirely. Quality is nonmonotonic for several metrics. These are descriptive comparisons across the saved runs. They do not establish causal effects of query count, universal monotonicity, or statistical significance.

Query counts provide a workload proxy. Per-call latency, query length, retrieval cache hits, delivered passage tokens, generation length and processing costs were not measured here. Different early retrieval decisions can change later generations and decision opportunities. Unpinned revisions, possible differences in decoding or infrastructure, and the finite set of thresholds limit generalization. No τ = 0.70 metric exists for HotpotQA or 2WikiMultiHopQA in the analyzed archive.

## Source file inventory

| ID | Source | ART responses with an answer tag |
|---|---|---:|
| M01 | [`REVIVE_training/metrics/baselines/2wikimultihopqa/no_rag_metrics.json`](../REVIVE_training/metrics/baselines/2wikimultihopqa/no_rag_metrics.json) | — |
| M02 | [`REVIVE_training/metrics/baselines/2wikimultihopqa/vanilla_rag_top5_metrics.json`](../REVIVE_training/metrics/baselines/2wikimultihopqa/vanilla_rag_top5_metrics.json) | — |
| M03 | [`REVIVE_training/metrics/baselines/hotpotqa/no_rag_metrics.json`](../REVIVE_training/metrics/baselines/hotpotqa/no_rag_metrics.json) | — |
| M04 | [`REVIVE_training/metrics/baselines/hotpotqa/vanilla_rag_top5_metrics.json`](../REVIVE_training/metrics/baselines/hotpotqa/vanilla_rag_top5_metrics.json) | — |
| M05 | [`REVIVE_training/metrics/baselines/musique/no_rag_metrics.json`](../REVIVE_training/metrics/baselines/musique/no_rag_metrics.json) | — |
| M06 | [`REVIVE_training/metrics/baselines/musique/vanilla_rag_top5_metrics.json`](../REVIVE_training/metrics/baselines/musique/vanilla_rag_top5_metrics.json) | — |
| M07 | [`art-backup/demo-3/metrics/2wikimultihopqa/rsearch_2wikimultihopqa_art_top5_metrics.json`](../art-backup/demo-3/metrics/2wikimultihopqa/rsearch_2wikimultihopqa_art_top5_metrics.json) | 488 |
| M08 | [`art-backup/demo-3/metrics/hotpotqa/rsearch_hotpotqa_art_top5_metrics.json`](../art-backup/demo-3/metrics/hotpotqa/rsearch_hotpotqa_art_top5_metrics.json) | 483 |
| M09 | [`art-backup/demo-3/metrics/hotpotqa/threshold50_metrics.json`](../art-backup/demo-3/metrics/hotpotqa/threshold50_metrics.json) | 492 |
| M10 | [`art-backup/demo-3/metrics/hotpotqa/threshold60_metrics.json`](../art-backup/demo-3/metrics/hotpotqa/threshold60_metrics.json) | 488 |
| M11 | [`art-backup/demo-3/metrics/musique/rsearch_musique_art_metrics.json`](../art-backup/demo-3/metrics/musique/rsearch_musique_art_metrics.json) | 479 |
| M12 | [`art-backup/demo-3/metrics/musique/rsearch_musique_art_top5_metrics.json`](../art-backup/demo-3/metrics/musique/rsearch_musique_art_top5_metrics.json) | 462 |
| M13 | [`art-backup/demo-3/metrics/musique/threshold50_metrics.json`](../art-backup/demo-3/metrics/musique/threshold50_metrics.json) | 475 |
| M14 | [`art-backup/demo-3/metrics/musique/threshold60_metrics.json`](../art-backup/demo-3/metrics/musique/threshold60_metrics.json) | 486 |
| M15 | [`art-backup/demo-3/metrics/musique/threshold70_metrics.json`](../art-backup/demo-3/metrics/musique/threshold70_metrics.json) | 495 |
| M16 | [`art-backup/demo-2/metrics/rsearch_musique_art_metrics.json`](../art-backup/demo-2/metrics/rsearch_musique_art_metrics.json) | 493 |
| M17 | [`art-backup/demo-2/metrics/rsearch_musique_art_top5_metrics.json`](../art-backup/demo-2/metrics/rsearch_musique_art_top5_metrics.json) | 486 |
| M18 | [`art-backup/demo-2/metrics/rsearch_musique_art_with_demonstrations_metrics.json`](../art-backup/demo-2/metrics/rsearch_musique_art_with_demonstrations_metrics.json) | 481 |
| M19 | [`art-backup/demo-2/metrics/rsearch_musique_art_with_demonstrations_top5_metrics.json`](../art-backup/demo-2/metrics/rsearch_musique_art_with_demonstrations_top5_metrics.json) | 493 |
| M20 | [`temp_model/rsearch_musique_art_metrics.json`](../temp_model/rsearch_musique_art_metrics.json) | 494 |
| M21 | [`temp_model/rsearch_musique_art_with_demonstrations_metrics.json`](../temp_model/rsearch_musique_art_with_demonstrations_metrics.json) | 490 |
| M22 | [`art-backup/demo-3/metrics/2wikimultihopqa/threshold50_metrics.json`](../art-backup/demo-3/metrics/2wikimultihopqa/threshold50_metrics.json) | 486 |
| M23 | [`art-backup/demo-3/metrics/2wikimultihopqa/threshold60_metrics.json`](../art-backup/demo-3/metrics/2wikimultihopqa/threshold60_metrics.json) | 492 |

M22 and M23 were generated by running `REVIVE_training.evaluate_art_pure` from the repository root on [threshold50.jsonl](../art-backup/demo-3/result/2wikimultihopqa/threshold50.jsonl) and [threshold60.jsonl](../art-backup/demo-3/result/2wikimultihopqa/threshold60.jsonl), respectively. The runs used UTF-8 stdout to handle the evaluator's debug output on Windows.

To reproduce the tables, read each linked JSON object's `num_samples` and `metrics` mapping, compute differences by key as described above, and take the maximum by key over the corresponding demo-3 default and available threshold rows. Exclude ZIP duplicates and configurations without a saved metric object.
