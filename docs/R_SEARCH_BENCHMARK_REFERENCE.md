# R-Search Benchmark Reference for ART

> External published benchmark, **not** an ART result. Source: Zhao et al., *R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning*, arXiv:2506.04185v1, Table 2 (June 2025).
>
> Paper: https://arxiv.org/html/2506.04185  
> Code: https://github.com/QingFei1/R-Search

## 1. Main benchmark: Qwen2.5-3B-Instruct

The following values are percentages. EM = Exact Match; F1 = answer token F1. Transcribed from the **Qwen-2.5-3B-Instruct** section of Table 2 of the cited paper. Dataset columns are ordered as in the paper.

| Method | HotpotQA EM | HotpotQA F1 | 2WikiMultiHopQA EM | 2WikiMultiHopQA F1 | MuSiQue EM | MuSiQue F1 | Bamboogle EM | Bamboogle F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLM w/o Search | 15.0 | 20.6 | 24.4 | 27.8 | 1.4 | 7.2 | 2.4 | 9.5 |
| Vanilla RAG | 34.0 | 43.2 | 33.6 | 38.0 | 5.6 | 11.7 | 9.6 | 19.6 |
| SuRe | 29.2 | 37.8 | 26.8 | 32.4 | 4.2 | 8.9 | 7.2 | 15.6 |
| Iter-Retgen | 34.4 | 43.4 | 33.2 | 38.2 | 8.2 | 14.8 | 12.0 | 20.2 |
| IRCoT | 39.0 | 50.4 | 35.8 | 46.0 | 9.2 | 17.6 | 23.2 | 33.3 |
| FLARE | 14.0 | 20.4 | 24.2 | 27.5 | 1.0 | 6.5 | 3.2 | 9.0 |
| Adaptive-RAG | 38.0 | 49.0 | 35.0 | 43.9 | 25.4 | 35.6 | 24.0 | 32.8 |
| Search-R1 | 46.2 | 57.8 | 58.8 | 68.1 | 24.4 | 32.9 | 41.6 | 53.9 |
| **R-Search (GRPO)** | **43.4** | **54.4** | **65.0** | **72.6** | **25.8** | **34.8** | **37.6** | **49.8** |

Note: R-Search is not the top-scoring method in every column; these are reported results, not rankings.

## 2. R-Search 7B results (reference only)

| Backbone | HotpotQA EM | HotpotQA F1 | 2Wiki EM | 2Wiki F1 | MuSiQue EM | MuSiQue F1 | Bamboogle EM | Bamboogle F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct / R-Search (GRPO) | 52.2 | 64.4 | 69.8 | 77.7 | 31.4 | 41.6 | 42.4 | 57.6 |

## 3. R-Search experimental setup reported in the paper

| Parameter | R-Search paper |
|---|---|
| Policy backbones | Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct |
| RL algorithms | GRPO and PPO |
| RL training dataset | 2WikiMultiHopQA training set |
| Dense retriever | E5 (official repository specifies `e5-base-v2`) |
| Evaluation retrieval top-k | 5 |
| Training retrieval top-k | 3 |
| Multi-hop evaluation | HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle |
| Single-hop evaluation | NQ, TriviaQA, PopQA |
| Retrieval corpus | Dataset-specific Wikipedia corpora for HotpotQA, 2WikiMultiHopQA and MuSiQue; Wiki-2018 for the remaining datasets |
| Training hardware | 8 × NVIDIA A100 80 GB |
| Reported maximum training steps | 195 |
| Reported training batch size | 256 |
| Learning rate | 1e-6 |
| KL coefficient | 0.001 |

Source: paper §4.1–4.3 and official GitHub README. These are **R-Search settings**, not verified ART settings.

## 4. Comparability checklist for ART

See [ART and R-Search experimental comparability](R_SEARCH_COMPARISON.md) for the source-grounded comparison and remaining evidence gaps.

Before using the published values as a head-to-head benchmark, verify:

- [ ] Same backbone model and checkpoint family (3B versus 7B, base versus instruct).
- [ ] Same evaluation dataset version, sample IDs, split, and sample count.
- [ ] Same answer normalization, EM and F1 implementation.
- [ ] Same retrieval corpus and index (including preprocessing and embedding model).
- [ ] Same retrieval top-k (the paper uses 5 for evaluation).
- [ ] Same prompting, answer extraction, generation limits, and decoding settings.
- [ ] Same policy for failed searches and malformed answers.

**Interpretation rule:** Even if ART uses a similar evaluation environment, the paper's R-Search values remain *published reference numbers*, not a controlled reproduction. If any setting is unverified or differs, label comparisons **indicative / non-controlled**, not evidence that ART outperforms R-Search under identical conditions.

## 5. Guidance for Codex and report writing

1. Use this file as an external baseline reference when drafting the ART experimental analysis.
2. Keep R-Search numbers in a clearly labeled separate table until comparability is checked.
3. Never mix R-Search's published metrics with ART's local demo-2/demo-3 metrics without identifying their origins.
4. ART's demo-2 and demo-3 both used 0.75 training epochs according to author clarification; the `0.1` value in the current `training.py` is a later accidental edit. Do not silently overwrite that provenance.
5. The demo-3 threshold50/60/70 variants are inference-time thresholds 0.50/0.60/0.70 on the same demo-3 checkpoint, according to the author.
6. Do not read large result JSONL files; use the corresponding metrics files.
7. When writing the report, cite the original paper (Table 2) for every imported benchmark table.

## 6. Citation

Zhao, Q., Wang, R., Xu, D., Zha, D., & Liu, L. (2025). *R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning*. arXiv:2506.04185. https://arxiv.org/abs/2506.04185

Checked against arXiv HTML version 1, Table 2: https://arxiv.org/html/2506.04185#S4.T2
