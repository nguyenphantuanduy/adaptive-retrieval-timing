# ART ablation · pure SFT without retrieval

This experiment measures the incremental effect of the ART system over conventional teacher-forced fine-tuning. ART versus SFT changes retrieval, Attention Mixing, the Decision Head, the prompt's retrieval instruction and the training procedure together. Original No-RAG versus SFT No-RAG uses a shared prompt and scoring protocol. Neither comparison isolates Attention Mixing by itself.

## Configuration

| Setting | ART reference (`training_v2.py`) | Pure SFT |
| --- | --- | --- |
| Initialization | Loads `adaptive-retrieval-timing/ART-REVIVE`; initializer derives its Qwen weights from the original backbone. The historical checkpoint revision and its full prior training history are unavailable. | Standard `Qwen/Qwen2.5-3B-Instruct`, optional `--backbone-revision` |
| Tokenizer | Original Qwen tokenizer | Same repo and standard chat template |
| Training data | `adaptive-retrieval-timing/MuSiQue-ART-Training-v2`, `train` in `2-hop`, `3-hop`, `4-hop`, concatenated in that order, shuffle seed 42. Source revision unspecified. | Same; optionally pin `--dataset-revision` |
| Validation and target | Rejects non-string and blank teacher output. The whole `teacher_generation_output` plus EOS is tokenized separately from the prompt. | Same. Targets remain intact. No query, document or answer field is inserted. |
| Prompt and mask | System + user + assistant generation prefix. Mask prompt and retrieval observation tokens with -100. | Same shared instructions and answer tags, omitting the retrieval-specific sentence. Mask prompt and right padding with -100. |
| Length | No explicit tokenizer truncation; ART builds trajectories from variable-length sequences. | No truncation or packing. Fails on sequences beyond the model context. Dynamic right padding, Qwen pad token. |
| Frozen and trainable weights | Last ceil(0.4 × 36) = 15 decoder layers, plus Decision Head. Embedding, LM head, final norm and other decoder blocks frozen. | Same decoder layers (0-based 21–35); embedding, final norm and LM head stay frozen. |
| Objective | LM Loss on target tokens plus retrieval regularization using Collect–Replay. | Only standard causal LM cross-entropy on target tokens. Native Transformers Trainer. |
| Optimizer | Transformers TrainingArguments default, resolved locally to fused PyTorch AdamW (betas 0.9/0.999, epsilon 1e-8). Exact historical library version unavailable. | Same library default; resolved arguments saved. Linear schedule, default zero warmup. |
| Training | LR 1e-5, weight decay 0, grad norm 1, microbatch 1, accumulation 32, 0.75 epoch, seed/data seed 42. | Same on a single CUDA device. |
| Precision and memory | BF16, eager attention, gradient checkpointing. ART uses cache during Collect, no cache during Replay. | BF16, eager attention, gradient checkpointing, cache off during training and restored for generation. |
| Logs and checkpoint | TensorBoard every step, save every 100 updates, no validation schedule. | Same intervals. Separate output directory, resumable Trainer checkpoints and a standalone final checkpoint. |

The default teacher revision is `aaf4b709de390e665047492016b9e8401b2def2d`. The audited dataset has 19,938 rows (14,376 / 4,387 / 1,175 by hop count). There were zero matches for the marker/tool-call scan and 19,794 matches for evidence-reference phrases. The scan is lexical and cannot rule out implicit dependence on the teacher's supporting documents. Each original target is preserved, including those references. Run the audit for any different revision before proceeding. Explicit marker matches require review and opt-in.

Expected workload from the planned sample count and batch settings is about 14,954 sample presentations and about 468 optimizer updates. Actual Trainer rounding and resume affect these counts. The final manifest records length statistics, trainable and frozen parameters, actual total optimizer steps, and this invocation's sample/token counts, elapsed training time and peak CUDA memory. For resumed training the invocation counters exclude earlier processes. Historical ART receipts do not establish matched optimizer steps, processed tokens, wall time or peak GPU memory.

## Commands

Run from the repository root with the project training environment, one visible BF16-capable CUDA GPU and sufficient storage. The current laptop's 6 GiB GPU is below the expected requirement. Roughly 6 GB of BF16 weights, about 2 GB of trainable gradients and about 4 GB of BF16 Adam moments already approach 12 GB before activations and allocator overhead. A 24 GiB GPU is a starting point, subject to the full-size smoke check and longest sequence. Allow at least 50 GB for model/cache and several checkpoints. Time should be estimated from observed update throughput on that GPU.

```text
python -m REVIVE_training.training_sft_baseline --audit-only --dataset-revision aaf4b709de390e665047492016b9e8401b2def2d
python -m REVIVE_training.training_sft_baseline --smoke-test --output-dir sft-norag-smoke --dataset-revision aaf4b709de390e665047492016b9e8401b2def2d
python -m REVIVE_training.evaluate_sft_baseline --dataset musique --model sft-norag-smoke/model --limit 2
python -m REVIVE_training.evaluate_sft_baseline --dataset hotpotqa --model sft-norag-smoke/model --limit 2
python -m REVIVE_training.evaluate_sft_baseline --dataset 2wikimultihopqa --model sft-norag-smoke/model --limit 2
python -m REVIVE_training.training_sft_baseline --dataset-revision aaf4b709de390e665047492016b9e8401b2def2d
python -m REVIVE_training.training_sft_baseline --resume latest --dataset-revision aaf4b709de390e665047492016b9e8401b2def2d
```

Confirm full-size forward/backward, finite loss, gradients only in permitted parameters, save/reload and the three smoke evaluations before launching full training. Supply the same revisions and output directory when resuming. An explicit checkpoint path can replace `latest`.

Local validation used the real Qwen tokenizer and audited teacher dataset. A decoded sample from each hop split reproduced the original target plus EOS, and a mixed-length batch masked the prompt and padding. A tiny standard Qwen2 model on CPU passed checkpointed forward/backward, gradient isolation, checkpoint saving and reload. The CLI generated and scored one archived sample per benchmark using that tiny checkpoint and `--cpu-smoke --limit 1`. These smoke scores are not benchmark results. The full 3B BF16 sanity run has to be verified on the training server.

## Evaluation and publication

The default DACN comparison uses the archived zero-shot ART top-5 runs at the historical threshold 0.4297. The evaluator takes only the ID, question and gold values from each archived JSONL, preserving all 500 rows and order. This pins the sample set independently of the upstream dataset revision. The local overlap check found zero exact train/test question matches on all three benchmarks after case and whitespace normalization and removing a trailing question mark. This check does not assess paraphrases. Generation uses seed 44, BF16, 1024 new tokens, temperature 0.1, top-p 0.9, top-k 0, one beam, and standard Qwen EOS handling. ART's custom cache and retrieval path may still affect effective generation. The shared No-RAG prompt omits the retrieval instruction, while preserving the reasoning and answer-tag format. The evaluator imports ART answer extraction and Acc/F1/EM directly.

```text
python -m REVIVE_training.evaluate_sft_baseline --dataset musique
python -m REVIVE_training.evaluate_sft_baseline --dataset hotpotqa
python -m REVIVE_training.evaluate_sft_baseline --dataset 2wikimultihopqa
python -m REVIVE_training.evaluate_sft_baseline --dataset musique --model Qwen/Qwen2.5-3B-Instruct
python -m REVIVE_training.evaluate_sft_baseline --dataset hotpotqa --model Qwen/Qwen2.5-3B-Instruct
python -m REVIVE_training.evaluate_sft_baseline --dataset 2wikimultihopqa --model Qwen/Qwen2.5-3B-Instruct
python -m REVIVE_training.evaluate_sft_baseline --summary
python -m REVIVE_training.training_sft_baseline --upload
```

Generation and scoring run together. The default batch size is 32 with left padding and an attention mask. Set `--batch-size` to a smaller value if memory is insufficient. Sampled generations may change with batch size even at the same seed. Outputs are `sft-norag-evaluation/<arm>/<dataset>/results.jsonl` and `metrics.json`. Each rerun stages both files and replaces that run's previous files after generation and scoring complete. Other datasets and arms are untouched. Smoke tests use separate subdirectories. A JSONL row contains `id`, `question`, `golden_answers`, `pure_art_answer` (raw), `extracted_answer`, `has_answer_tag`, and empty compatibility query/passage arrays. Metrics contain the sample count, extraction counts, scores, batch size and generation provenance. `--summary` produces `sft-norag-evaluation/summary.json`.

Authenticate through the existing Hub login or `HF_TOKEN`. Run `python -m REVIVE_training.training_sft_baseline --push-to-hub` to upload automatically after successful training and final saves. For an already saved model, use `--upload`. Both commands create the separate private repository `adaptive-retrieval-timing/Qwen2.5-3B-Instruct-SFT-NoRAG`. Repository creation enforces write permission and fails if the repo already exists. Upload includes weights, configuration, tokenizer, generation configuration, the training manifest and a model card. Benchmark results are optional and can be added to the card later.

| Model | MuSiQue Acc/F1/EM | HotpotQA Acc/F1/EM | 2WikiMultiHopQA Acc/F1/EM |
| --- | --- | --- | --- |
| Qwen2.5-3B-Instruct, Original No-RAG | Pending matched prompt/scoring | Pending | Pending |
| Qwen2.5-3B-Instruct, SFT No-RAG | Pending training | Pending | Pending |
| ART archived default | 25.80 / 31.17 / 21.60 | 42.20 / 48.24 / 37.20 | 48.20 / 46.30 / 37.00 |

### Training and evaluation on separate machines

On machine A, train with `--push-to-hub`. Before upload, the script checks that the final model directory contains the model config, generation config, tokenizer and chat template, and all weight shards. Evaluation scores are optional.

On machine B, clone the repository including the tracked `art-backup/demo-3/result/*/*_art_top5_results.jsonl` files. These supply the exact evaluation samples. No training data, Trainer checkpoint or files from machine A are needed. Authenticate using `hf auth login` or set `HF_TOKEN` with read access to the private model. Run from the repository root with the normal BF16 inference environment.

```bash
python -m REVIVE_training.evaluate_sft_baseline --dataset musique --model adaptive-retrieval-timing/Qwen2.5-3B-Instruct-SFT-NoRAG
hf download adaptive-retrieval-timing/Qwen2.5-3B-Instruct-SFT-NoRAG --local-dir checkpoints/sft-norag
python -m REVIVE_training.evaluate_sft_baseline --dataset musique --model checkpoints/sft-norag --local-files-only --output-dir sft-norag-evaluation-local
```

Use either loading route. Substitute `hotpotqa` or `2wikimultihopqa` for the other benchmarks. To reproduce the same Hub snapshot, pass its commit SHA via `--revision` on the evaluator or on `hf download`. The direct route uses the resolved model revision for the tokenizer too and records it in the metrics.

The older No-RAG results use a different prompt and raw-response scoring. They are context for the DACN system benchmark. Use the matched Original No-RAG run above to estimate the SFT effect.
