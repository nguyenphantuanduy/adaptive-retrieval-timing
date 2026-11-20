# ART experiments and provenance

**Author.** Demo-2 comes from `training.py`, demo-3 from `training_v2.py`. Both trained for 0.75 epochs and were uploaded to Hugging Face. For corresponding demo runs, test examples, retrieval settings, prompts and evaluation procedure were the same. This supports comparing the resulting checkpoints as outputs of different training pipelines. The current `0.1` in `training.py` was introduced after the experiment.

**Artifacts.** Every row below has 500 samples. Score columns are percentages. Tagged is the number of generations with a recognized answer tag pair. File paths are relative to [`art-backup/`](../art-backup/).

| Demo | Dataset | Variant | Acc | F1 | EM | Tagged | Metrics file |
|---|---|---|---:|---:|---:|---:|---|
| 2 | MuSiQue | Plain | 18.0 | 25.04 | 16.8 | 493 | `demo-2/metrics/rsearch_musique_art_metrics.json` |
| 2 | MuSiQue | Top-5 | 20.0 | 26.33 | 17.6 | 486 | `demo-2/metrics/rsearch_musique_art_top5_metrics.json` |
| 2 | MuSiQue | Demonstrations | 15.2 | 22.73 | 13.2 | 481 | `demo-2/metrics/rsearch_musique_art_with_demonstrations_metrics.json` |
| 2 | MuSiQue | Demonstrations, top-5 | 11.4 | 17.13 | 9.2 | 493 | `demo-2/metrics/rsearch_musique_art_with_demonstrations_top5_metrics.json` |
| 3 | MuSiQue | Plain | 22.8 | 28.46 | 19.6 | 479 | `demo-3/metrics/musique/rsearch_musique_art_metrics.json` |
| 3 | MuSiQue | Top-5 | 25.8 | 31.17 | 21.6 | 462 | `demo-3/metrics/musique/rsearch_musique_art_top5_metrics.json` |
| 3 | HotpotQA | Top-5 | 42.2 | 48.24 | 37.2 | 483 | `demo-3/metrics/hotpotqa/rsearch_hotpotqa_art_top5_metrics.json` |
| 3 | 2WikiMultiHopQA | Top-5 | 48.2 | 46.30 | 37.0 | 488 | `demo-3/metrics/2wikimultihopqa/rsearch_2wikimultihopqa_art_top5_metrics.json` |

Demo-3 minus demo-2 on MuSiQue yields **+4.8 Acc, +3.42 F1 and +2.8 EM** points for plain, and **+5.8 Acc, +4.84 F1 and +4.0 EM** for top-5. Tagged generations fell by 14 and 24 respectively. The archive supplies no matching demo-3 demonstrations row or demo-2 HotpotQA/2Wiki row. Variant labels come from filenames. Comparisons between variants within a demo require accounting for their own configuration changes.

## Inference-only threshold sweep

**Author.** These runs use the same demo-3 checkpoint, changing only the inference threshold. No retraining occurred. An explicit threshold filename establishes the value with that provenance. The threshold of a plain or top-5 file without a threshold label is unverified.

| Dataset | Threshold | Acc | F1 | EM | Tagged | File under `demo-3/metrics/` |
|---|---:|---:|---:|---:|---:|---|
| MuSiQue | 0.50 | 24.0 | 30.27 | 20.8 | 475 | `musique/threshold50_metrics.json` |
| MuSiQue | 0.60 | 26.0 | 32.08 | 22.8 | 486 | `musique/threshold60_metrics.json` |
| MuSiQue | 0.70 | 22.6 | 29.21 | 19.8 | 495 | `musique/threshold70_metrics.json` |
| HotpotQA | 0.50 | 43.8 | 49.68 | 39.2 | 492 | `hotpotqa/threshold50_metrics.json` |
| HotpotQA | 0.60 | 40.0 | 46.87 | 36.4 | 488 | `hotpotqa/threshold60_metrics.json` |

Among saved points, MuSiQue peaks at 0.60 on the three answer metrics and HotpotQA at 0.50. HotpotQA has no 0.70 aggregate and 2Wiki has no threshold-labeled aggregate in the inspected archive. The sweep supplies no retrieval-count, latency, or evidence-recall aggregates. A higher threshold is stricter at the same decision score, while the actual trajectory and later scores can change when an earlier retrieval is skipped.

The two [`temp_model/`](../temp_model/) MuSiQue summaries also have 500 samples. Plain is 12.4 Acc / 18.94 F1 / 11.0 EM with 494 tagged. Demonstrations is 11.8 / 19.47 / 11.0 with 490 tagged. Their immutable checkpoint/configuration and relationship to the demo runs remain unverified.

## Outstanding evidence

Exact model, corpus and dataset revisions, archived commands and environment, step counts, training dataset counts after filtering, assigned sample IDs, actual decoding settings and absolute retrieval settings for each archived run are not independently linked by the saved aggregates. The author confirms equality for matched demo evaluation settings. Aggregate scores do not establish per-question gains, confidence intervals, robustness across seeds, calibration, retrieval efficiency, or a causal decomposition of the training changes. Historical code served via `trust_remote_code` may differ from the current local source.
