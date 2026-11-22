# ART codebase reading coverage

## Subsequent semantic review

The inventory and counts below describe the original snapshot. The following research paths received additional function-body and data-flow review. This update records the reviewed behavior rather than assigning full-file coverage.

| Research path | Review completed | Remaining limits |
|---|---|---|
| REVIVE decoder and inference | Forward dispatch, rotary positions and additive masks, attention and KV updates, generation/insertion order, streaming boundaries and all three retriever return contracts. Initialization and config were inspected for model transfer and export. [Architecture](REVIVE_ARCHITECTURE.md) | No dynamic validation of cache compatibility, unusual input shapes or historical remote revisions. |
| REVIVE training and evaluation | Compared training entry points, traced collation, collect/replay, penalty and terminal behavior, top-5 generation, answer extraction and all three metrics. [Implementation](REVIVE_TRAINING_EVALUATION.md) | Alternate trainers remain at their earlier depth. Edge cases and historical revisions were not runtime-verified. |
| H1 and H2 | Traced document pools, mask and supervision construction, loss and gradient loops, pooling, notebook analysis source, and computed descriptive findings from both aggregate CSVs. [Hypothesis verification](HYPOTHESIS_VERIFICATION.md) | Individual records, stored notebook outputs and hypothesis reruns were outside this review. |

## Scope and assessment

The snapshot found **52,897 paths** outside Git internals. This includes **51,718 virtual-environment files**, **56 project bytecode files** and **828 compressed corpus shards**. Those groups are recorded by pattern below. The remaining **295 paths** are listed individually. The snapshot precedes creation of this audit page. An empty `phase_2/` directory was present. The Git index listed 257 paths. The workspace inventory includes ignored and untracked files, since both can contain project artifacts.

The previous documentation traced selected REVIVE paths and read the demo metrics. Its source citations did not imply full-file review. In this audit, every project Python file outside the generated Hugging Face modules was submitted to a static parser and inspected through its structural summary, with targeted body reads where recorded. The parser flagged `set_model.py`. Notebook coverage focuses on source cells. Full semantic and runtime coverage remains incomplete. No project Python module, notebook, training script or evaluator was executed.

| Status | Paths | Meaning |
|---|---:|---|
| Excluded | 81 | Filename only. Reason listed below |
| Read | 48 | Contents read or previously authored and reviewed |
| Surveyed | 94 | Structural inspection, with targeted excerpts or selected configuration fields |
| Surveyed (some prior excerpts) | 61 | Structural inspection in this audit, with prior excerpts in parts of the group |
| Traced | 11 | Relevant behavior traced through selected function bodies across the two passes |

## Findings that extend the earlier docs

- `ART_HOPE` splits the shared decoder at layer 22 by default and runs 14 separate decision layers. It has a separate decision cache and an exponential term in its retrieval cost. `HOPE_training` explicitly unfreezes the decision branch, decision normalization and decision head. These paths extend the model map and should not inherit the REVIVE objective.
- `full_training.py` enables full fine-tuning. `training_70.py` selects the last 70% of layers. `training_with_2_gpu.py` uses model placement via `device_map`. `train_fsdp2.py` defines a sampler that repeats indices so Accelerate's intended sharding gives each rank the same example. That is source intent. Distributed behavior was not tested.
- `temp_model/model.py` has a local cost coefficient of 0.1, while current REVIVE uses 0.05. The local coefficient does not establish the setting of the archived temporary-model metrics.
- `debug_server.py` and `review_removed_samples_v2.py` are Streamlit analysis tools. The first inspects generation, retrieval and original MuSiQue evidence. The second filters and reviews rejected teacher samples. The default result path in `debug_server.py` omits a dataset subdirectory, and no file at that exact path was in the snapshot.
- The saved MuSiQue comparison CSV records byte-for-byte equivalence of six original versus R-Search-distributed raw split files. This extends corpus provenance, while identity of the 500-example evaluation set with the published sample remains unverified.
- The saved teacher-cleaning summaries report different modes. V1 records 19,919 valid of 19,938 in a clean-and-upload run and 10,082 cleaned question prefixes. V2 records 19,350 valid of 19,938 in a dry run, with 588 removals and 454 insufficient-evidence answers flagged. These counts describe cleaner outputs. They do not identify rows used in either archived checkpoint or independently measure teacher quality.
- H1 and H2 each have a saved aggregate CSV with 528 rows, by hop, retrieval ratio and decision score. H2 notebooks analyze a gradient-informed threshold and candidate retrieval costs. These artifacts extend the research map. Numerical reproduction and linkage to historical demo configurations remain open.

## Recursive exclusions recorded by path pattern

| Scope | Exclusion |
|---|---|
| `.git/**` | Git internals. Used index and status only. Outside the workspace file count. |
| `.venv/**` | 51,718 installed third-party files and environment artifacts. |
| `**/__pycache__/**` outside `.venv/` | 56 project bytecode entries. Source paths are listed individually. |
| `**/*.bz2` outside `.venv/` | 828 compressed corpus shards. Used ingestion scripts and aggregate evidence. |

The individual exclusions below cover cached modules, credential material, archives, raw JSONL, TensorBoard streams, plots and per-sample CSV/JSON. No local weights or generated FAISS index files were identified in the non-environment inventory snapshot. Remote model and index artifacts were outside scope.

## Remaining uncertainties and next reading priorities

1. Complete line-by-line semantic review of the surveyed files, especially alternate trainers, initialization and upload checks, distributed execution, branch-specific decoding and cache behavior, and edge cases in the three index builders. The symbols below are navigation evidence, not proof that each implementation branch was checked.
2. Trace all profiling and H2 notebook cells in detail. Verify any reported H1/H2 claims against published aggregates and the exact probe/checkpoint configuration before reusing them as results.
3. Resolve immutable model, dataset and index revisions and the exact configurations for historical runs. The [experiment evidence ledger](EXPERIMENTS.md#outstanding-evidence) records the existing gaps.
4. Verify when the local debug scripts, copied model files and standalone generation utilities were used. Filename similarity and local presence cannot establish demo provenance.

## Individual file inventory

Each row records the actual review depth and a purpose or inspected entry point. Paths are repository-relative.

### Excluded

| File | Purpose or evidence |
|---|---|
| `.cache/huggingface/.gitignore` | Generated Hugging Face cache or downloaded module copy |
| `.cache/huggingface/CACHEDIR.TAG` | Generated Hugging Face cache or downloaded module copy |
| `.cache/huggingface/download/attention_mixing.py.metadata` | Generated Hugging Face cache or downloaded module copy |
| `.cache/huggingface/download/model.py.metadata` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/__init__.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/__init__.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/Qwen2_dot_5_3B/__init__.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/Qwen2_dot_5_3B/2821da4538c26c38/__init__.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/Qwen2_dot_5_3B/2821da4538c26c38/attention_mixing.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/Qwen2_dot_5_3B/2821da4538c26c38/cache_utils.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/Qwen2_dot_5_3B/2821da4538c26c38/decoder_layer.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/Qwen2_dot_5_3B/2821da4538c26c38/model.py` | Generated Hugging Face cache or downloaded module copy |
| `.hf_modules/transformers_modules/Qwen2_dot_5_3B/2821da4538c26c38/utils.py` | Generated Hugging Face cache or downloaded module copy |
| `ART_SECRET_TOKEN.md` | Credential file, path only |
| `art-backup/demo-2/art-training-output/runs/Oct14_11-30-25_bubbly-gecko/events.out.tfevents.1791952225.bubbly-gecko.12450.0` | Binary TensorBoard run event stream |
| `art-backup/demo-2/art-training-output/runs/Oct14_12-07-15_bubbly-gecko/events.out.tfevents.1791954435.bubbly-gecko.12689.0` | Binary TensorBoard run event stream |
| `art-backup/demo-2/result/rsearch_musique_art_as_tool_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-2/result/rsearch_musique_art_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-2/result/rsearch_musique_art_top5_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-2/result/rsearch_musique_art_with_demonstrations_as_tool_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-2/result/rsearch_musique_art_with_demonstrations_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-2/result/rsearch_musique_art_with_demonstrations_top5_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-2/result/teacher_hypothesis_demonstrations.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/art-training-output/runs/Nov02_15-11-26_O-2389147/events.out.tfevents.1793607086.O-2389147.9814.0` | Binary TensorBoard run event stream |
| `art-backup/demo-3/art-training-output/runs/Nov02_15-26-54_O-2389147/events.out.tfevents.1793608014.O-2389147.12201.0` | Binary TensorBoard run event stream |
| `art-backup/demo-3/art-training-output/runs/Nov02_15-33-39_O-2389147/events.out.tfevents.1793608419.O-2389147.13843.0` | Binary TensorBoard run event stream |
| `art-backup/demo-3/result/2wikimultihopqa/rsearch_2wikimultihopqa_art_top5_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/hotpotqa/rsearch_hotpotqa_art_top5_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/hotpotqa/threshold50.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/hotpotqa/threshold60.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/musique/rsearch_musique_art_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/musique/rsearch_musique_art_top5_results.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/musique/threshold50.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/musique/threshold60.jsonl` | Raw generation or experimental input records |
| `art-backup/demo-3/result/musique/threshold70.jsonl` | Raw generation or experimental input records |
| `assets/logo/hf-logo.png` | Image asset or generated plot |
| `docs.zip` | Archive |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/lm_loss_2_hop.csv` | Per-sample probe output. Used aggregate summary file |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/lm_loss_3_hop.csv` | Per-sample probe output. Used aggregate summary file |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/lm_loss_4_hop.csv` | Per-sample probe output. Used aggregate summary file |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/plots/lm_loss_vs_decision_score_2hop.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/plots/lm_loss_vs_decision_score_3hop.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/plots/lm_loss_vs_decision_score_4hop.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/plots/lm_loss_vs_decision_score_total.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/gradient_2_hop.csv` | Per-sample probe output. Used aggregate summary file |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/gradient_3_hop.csv` | Per-sample probe output. Used aggregate summary file |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/gradient_4_hop.csv` | Per-sample probe output. Used aggregate summary file |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_vs_decision_score_2hop.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_vs_decision_score_3hop.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_vs_decision_score_4hop.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_vs_decision_score_total.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_vs_decision_score_total_nonlinear_boundary.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_vs_decision_score_total_with_threshold.png` | Image asset or generated plot |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_vs_decision_score_total_with_threshold_and_costs.png` | Image asset or generated plot |
| `REVIVE_training/art-training-output/runs/Nov07_17-30-59_O-2414799/events.out.tfevents.1794047459.O-2414799.8679.0` | Binary TensorBoard run event stream |
| `REVIVE_training/art-training-output/runs/Nov07_17-38-25_O-2414799/events.out.tfevents.1794047905.O-2414799.10724.0` | Binary TensorBoard run event stream |
| `REVIVE_training/art-training-output/runs/Nov07_17-43-38_O-2414799/events.out.tfevents.1794048218.O-2414799.12106.0` | Binary TensorBoard run event stream |
| `REVIVE_training/art-training-output/runs/Nov07_18-03-24_O-2414799/events.out.tfevents.1794049404.O-2414799.16760.0` | Binary TensorBoard run event stream |
| `REVIVE_training/art-training-output/runs/Nov07_18-32-07_O-2414799/events.out.tfevents.1794051127.O-2414799.23479.0` | Binary TensorBoard run event stream |
| `REVIVE_training/art-training-output/runs/Nov07_19-02-20_O-2414799/events.out.tfevents.1794052940.O-2414799.30551.0` | Binary TensorBoard run event stream |
| `shared/ART_Models/ART_HOPE/cache_utils.py` | Generated per-sample cleaning details |
| `shared/HotpotQA_Distractor/data/corpus/hotpotqa/hotpotqa_corpus.jsonl` | Raw dataset or corpus |
| `shared/HotpotQA_Distractor/data/hotpotqa_rsearch/rsearch_hotpotqa_test.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_ans_v1.0_dev.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_ans_v1.0_test.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_ans_v1.0_train.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_corpus.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_full_v1.0_dev.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_full_v1.0_test.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_full_v1.0_train.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique/musique_v1.0.zip` | Archive |
| `shared/MuSiQue/data/musique_rsearch/musique_ans_v1.0_dev.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique_rsearch/musique_ans_v1.0_test.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique_rsearch/musique_ans_v1.0_train.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique_rsearch/musique_full_v1.0_dev.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique_rsearch/musique_full_v1.0_test.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique_rsearch/musique_full_v1.0_train.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/musique_rsearch/rsearch_musique_test.jsonl` | Raw dataset or corpus |
| `shared/MuSiQue/data/teacher_training_removed_samples_v2.json` | Generated per-sample cleaning details |
| `temp_model/rsearch_musique_art_results.jsonl` | Raw generation or experimental input records |
| `temp_model/rsearch_musique_art_with_demonstrations_results.jsonl` | Raw generation or experimental input records |

### Read

| File | Purpose or evidence |
|---|---|
| `.gitignore` | Version-control ignore rules |
| `LICENSE` | MIT license |
| `shared/MuSiQue/load_hypothesis_validation.py` | Loads the validation split for each hop from the hypothesis-probe dataset repository |
| `shared/MuSiQue/load_training_dataset.py` | Loads the v1 teacher dataset's training split per hop. REVIVE entry points have their own dataset selectors |
| `ART_CODEBASE_ANALYSIS_TASK.md` | Project, hypothesis or teacher documentation |
| `art-backup/demo-2/metrics/rsearch_musique_art_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-2/metrics/rsearch_musique_art_top5_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-2/metrics/rsearch_musique_art_with_demonstrations_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-2/metrics/rsearch_musique_art_with_demonstrations_top5_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/2wikimultihopqa/rsearch_2wikimultihopqa_art_top5_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/hotpotqa/rsearch_hotpotqa_art_top5_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/hotpotqa/threshold50_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/hotpotqa/threshold60_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/musique/rsearch_musique_art_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/musique/rsearch_musique_art_top5_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/musique/threshold50_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/musique/threshold60_metrics.json` | Archived aggregate answer metrics and tag counts |
| `art-backup/demo-3/metrics/musique/threshold70_metrics.json` | Archived aggregate answer metrics and tag counts |
| `demo_training/__init__.py` | Empty package marker |
| `docs/EXPERIMENTS.md` | Documentation from the earlier deliverable |
| `docs/METHODOLOGY.md` | Documentation from the earlier deliverable |
| `docs/R_SEARCH_BENCHMARK_REFERENCE.md` | Documentation from the earlier deliverable |
| `docs/R_SEARCH_COMPARISON.md` | Documentation from the earlier deliverable |
| `docs/README.md` | Documentation from the earlier deliverable |
| `docs/REPORT_PREPARATION.md` | Documentation from the earlier deliverable |
| `docs/TRAINING.md` | Documentation from the earlier deliverable |
| `docs/WORKFLOWS.md` | Documentation from the earlier deliverable |
| `HOPE_training/__init__.py` | Empty package marker |
| `phase_1/stage_2_hypothesis_verification/__init__.py` | Empty package marker |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/__init__.py` | Empty package marker |
| `pyproject.toml` | Empty project metadata file |
| `README.md` | Project, hypothesis or teacher documentation |
| `requirements.txt` | Runtime dependency declarations |
| `REVIVE_training/__init__.py` | Empty package marker |
| `set_model.py` | Generated prose stored with a Python extension. Python parser reports unmatched closing parenthesis on line 1 |
| `shared/ART_Models/ART_demo_version/__init__.py` | Empty package marker |
| `shared/ART_Models/ART_HOPE/__init__.py` | Empty package marker |
| `shared/ART_Models/ART_MU/__init__.py` | Empty package marker |
| `shared/ART_Models/ART_REVIVE/__init__.py` | Empty package marker |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/__init__.py` | Empty package marker |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/__init__.py` | Empty package marker |
| `shared/MuSiQue/__init__.py` | Empty package marker |
| `shared/MuSiQue/data/musique_comparison.csv` | Dataset equivalence or teacher-cleaning aggregate |
| `shared/MuSiQue/data/teacher_training_cleaning_statistics_v2.csv` | Dataset equivalence or teacher-cleaning aggregate |
| `shared/MuSiQue/teacher_training_cleaning_statistics.csv` | Dataset equivalence or teacher-cleaning aggregate |
| `temp_model/rsearch_musique_art_metrics.json` | Archived aggregate answer metrics and tag counts |
| `temp_model/rsearch_musique_art_with_demonstrations_metrics.json` | Archived aggregate answer metrics and tag counts |
| `test/__init__.py` | Empty package marker |

### Surveyed

| File | Purpose or evidence |
|---|---|
| `cleanup_credentials.sh` | Credential cleanup operations. Script not executed |
| `debug_server.py` | `extract_answer`, `normalize_question`, `safe_json`, `as_list`, `get_passage_text` |
| `demo_training/test_model_after_training.py` | `main` |
| `demo_training/trainer.py` | `ARTTrainer` |
| `demo_training/training.py` | `set_seed`, `load_teacher_generated_training`, `build_combined_dataset`, `tokenize_training_sample`, `preprocess_dataset` |
| `demo_training/upload_to_hf.py` | `upload_model_to_hf` |
| `GPU_SETUP.md` | Linux GPU setup commands |
| `HOPE_training/test_model_after_training.py` | `main` |
| `HOPE_training/trainer.py` | `ARTTrainer` |
| `HOPE_training/training.py` | `set_seed`, `load_teacher_generated_training`, `build_combined_dataset`, `tokenize_training_sample`, `preprocess_dataset` |
| `HOPE_training/upload_to_hf.py` | `upload_model_to_hf` |
| `main.py` | `main` |
| `model.py` | `flip_coin`, `increase_probability`, `decrease_probability`, `format_search_query`, `compute_retrieval_cost` |
| `phase_1/stage_0_literature/ircot.md` | Phase 1 related-work notes |
| `phase_1/stage_0_literature/r-search.md` | Phase 1 related-work notes |
| `phase_1/stage_0_literature/self-rag.md` | Phase 1 related-work notes |
| `phase_1/stage_1_system_profiling/art-system-profiling-stage-0.ipynb` | Source cell headings and declarations inspected. Embedded output not analyzed |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/lm_loss_results/plots/lm_loss_summary.csv` | Aggregate H1 loss or H2 gradient curves by hop, ratio and score |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/README.md` | Project, hypothesis or teacher documentation |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/analyze_threshold_and_cost.ipynb` | Source cell headings and declarations inspected. Embedded output not analyzed |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/analyze_threshold_and_cost_simplified.ipynb` | Source cell headings and declarations inspected. Embedded output not analyzed |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/gradient_results/plots/gradient_summary.csv` | Aggregate H1 loss or H2 gradient curves by hop, ratio and score |
| `shared/ART_Models/ART_demo_version/attention_mixing.py` | `attention_mixing_forward`, `AttentionMixingProbeAttention` |
| `shared/ART_Models/ART_demo_version/cache_utils.py` | `ARTCache`, `copy_dynamic_cache` |
| `shared/ART_Models/ART_demo_version/config.json` | Model type, architecture, decoder count and AutoModel mapping |
| `shared/ART_Models/ART_demo_version/decoder_layer.py` | `AttentionMixingProbeDecoderLayer` |
| `shared/ART_Models/ART_demo_version/initialize_art_demo_qwen2_5_3b.py` | `print_names`, `verify_qwen_compatibility`, `verify_transferred_values`, `initialize_retrieval_decision_head`, `verify_retrieval_decision_head` |
| `shared/ART_Models/ART_demo_version/model.py` | `flip_coin`, `increase_probability`, `decrease_probability`, `AttentionMixingProbePreTrainedModel`, `AttentionMixingProbeModel` |
| `shared/ART_Models/ART_demo_version/musique_retrieval_db.py` | `truncate_passages`, `MuSiQueRetrievalDB` |
| `shared/ART_Models/ART_demo_version/retrieval_context_processor.py` | `SentenceContextWindow`, `load_spacy_model`, `RetrievalContextProcessor` |
| `shared/ART_Models/ART_demo_version/utils.py` | `ensure_document_attention_mask`, `create_position_ids_without_doc`, `align_document_attention_mask`, `create_document_attention_mask`, `update_attention_mask` |
| `shared/ART_Models/ART_HOPE/attention_mixing.py` | `attention_mixing_forward`, `AttentionMixingProbeAttention` |
| `shared/ART_Models/ART_HOPE/config.json` | Model type, architecture, decoder count and AutoModel mapping |
| `shared/ART_Models/ART_HOPE/configuration_art.py` | `ARTConfig` |
| `shared/ART_Models/ART_HOPE/decoder_layer.py` | `AttentionMixingProbeDecoderLayer` |
| `shared/ART_Models/ART_HOPE/initialize_art_hope_qwen2_5_3b.py` | `print_names`, `build_qwen_transfer_mapping`, `verify_transfer_mapping`, `verify_transferred_values`, `verify_decision_branch_copy` |
| `shared/ART_Models/ART_HOPE/model.py` | `flip_coin`, `increase_probability`, `decrease_probability`, `ARTModelOutputWithPast`, `AttentionMixingProbePreTrainedModel` |
| `shared/ART_Models/ART_HOPE/musique_retrieval_db.py` | `truncate_passages`, `MuSiQueRetrievalDB`, `append_retrieval_observation` |
| `shared/ART_Models/ART_HOPE/retrieval_context_processor.py` | `SentenceContextWindow`, `RetrievalContextProcessor` |
| `shared/ART_Models/ART_HOPE/utils.py` | `ensure_document_attention_mask`, `create_position_ids_without_doc`, `align_document_attention_mask`, `create_document_attention_mask`, `update_attention_mask` |
| `shared/ART_Models/ART_MU/attention_mixing.py` | `attention_mixing_forward`, `AttentionMixingProbeAttention` |
| `shared/ART_Models/ART_MU/cache_utils.py` | `ARTCache`, `copy_dynamic_cache` |
| `shared/ART_Models/ART_MU/config.json` | Model type, architecture, decoder count and AutoModel mapping |
| `shared/ART_Models/ART_MU/decoder_layer.py` | `AttentionMixingProbeDecoderLayer` |
| `shared/ART_Models/ART_MU/hotpotqa_retrieval_db.py` | `truncate_passages`, `append_retrieval_observation`, `HotpotQARetrievalDB` |
| `shared/ART_Models/ART_MU/initialize_art_revive.py` | `print_names`, `verify_qwen_compatibility`, `verify_transferred_values`, `initialize_retrieval_decision_head`, `verify_retrieval_decision_head` |
| `shared/ART_Models/ART_MU/initialize_art_revive_with_seed.py` | `set_seed`, `print_names`, `verify_qwen_compatibility`, `verify_transferred_values`, `initialize_retrieval_decision_head` |
| `shared/ART_Models/ART_MU/model.py` | `flip_coin`, `increase_probability`, `decrease_probability`, `compute_retrieval_cost`, `AttentionMixingProbePreTrainedModel` |
| `shared/ART_Models/ART_MU/musique_retrieval_db.py` | `truncate_passages`, `MuSiQueRetrievalDB`, `append_retrieval_observation` |
| `shared/ART_Models/ART_MU/retrieval_context_processor.py` | `SentenceContextWindow`, `RetrievalContextProcessor` |
| `shared/ART_Models/ART_MU/utils.py` | `ensure_document_attention_mask`, `create_position_ids_without_doc`, `align_document_attention_mask`, `create_document_attention_mask`, `update_attention_mask` |
| `shared/ART_Models/ART_REVIVE/config.json` | Model type, architecture, decoder count and AutoModel mapping |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/attention_mixing.py` | `attention_mixing_forward`, `AttentionMixingProbeAttention` |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/cache_utils.py` | `ARTCache` |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/config.json` | Model type, architecture, decoder count and AutoModel mapping |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/decoder_layer.py` | `AttentionMixingProbeDecoderLayer` |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/initialize_art_probe_qwen2_5_3b.py` | `print_names`, `verify_state_dicts`, `verify_transferred_values`, `copy_custom_code`, `main` |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/model.py` | `AttentionMixingProbePreTrainedModel`, `AttentionMixingProbeModel`, `AttentionMixingProbeForCausalLM` |
| `shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/utils.py` | `ensure_document_attention_mask`, `create_position_ids_without_doc`, `align_document_attention_mask`, `make_document_attention_mask` |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/attention_mixing.py` | `attention_mixing_forward`, `AttentionMixingProbeAttention` |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/cache_utils.py` | `ARTCache` |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/config.json` | Model type, architecture, decoder count and AutoModel mapping |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/decoder_layer.py` | `AttentionMixingProbeDecoderLayer` |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/masking.py` | `prepare_document_attention_mask` |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/model.py` | `AttentionMixingProbePreTrainedModel`, `AttentionMixingProbeModel`, `AttentionMixingProbeForCausalLM` |
| `shared/ART_Models/attention_mixing_probe/Qwen3_4B/utils.py` | `ensure_document_attention_mask`, `create_position_ids_without_doc`, `align_document_attention_mask`, `make_document_attention_mask` |
| `shared/MuSiQue/art-musique-teacher-gen.ipynb` | Source cell headings and declarations inspected. Embedded output not analyzed |
| `shared/MuSiQue/teacher_demonstrations.md` | Project, hypothesis or teacher documentation |
| `shared/MuSiQue/teacher_reasoning_demonstrations.json` | Four teacher prompt demonstrations |
| `shared/utils/compute_per_sample_lm_loss.py` | `compute_per_sample_lm_loss` |
| `shared/utils/musique_retrieval_db.py` | `MuSiQueRetrievalDB` |
| `shared/utils/retrieval_context_processor.py` | `SentenceContextWindow`, `load_spacy_model`, `RetrievalContextProcessor` |
| `temp_model/model.py` | `flip_coin`, `increase_probability`, `decrease_probability`, `compute_retrieval_cost`, `AttentionMixingProbePreTrainedModel` |
| `temp_model/musique_retrieval_db.py` | `truncate_passages`, `MuSiQueRetrievalDB`, `append_retrieval_observation` |
| `test/test_art_attention_mixing_probe_generation.py` | `get_input_device`, `main` |
| `test/test_art_attention_mixing_probe_generation_real_sample.py` | `get_input_device`, `select_documents`, `main` |
| `test/test_art_gradient_probe.py` | `cleanup_cuda`, `GradientCapture`, `get_decision_index`, `build_dataset`, `install_gradient_hook` |
| `test/test_chat_template.py` | `main` |
| `test/test_config.py` | Top-level runtime or configuration check |
| `test/test_h1_attention_modulation_full_text_ratios.py` | `main` |
| `test/test_index.py` | `mean_pooling`, `load_hf_artifact`, `load_e5`, `encode_sentence`, `search` |
| `test/test_load_hotpotqa.py` | `test_load_hotpotqa` |
| `test/test_online_sentence_boundary.py` | `main` |
| `test/test_online_sentence_boundary_hard.py` | `main` |
| `test/test_teacher_dataset.py` | `main` |
| `test/test_teacher_dataset_correct.py` | `normalize_answer`, `extract_final_answer`, `check_sample`, `check_hop_dataset`, `main` |
| `test/test_teacher_generated_training_dataset.py` | `normalize_answer`, `extract_final_answer`, `check_sample`, `check_hop_dataset`, `main` |
| `test/test_teacher_output_question_prefix.py` | `output_starts_with_question`, `main` |
| `test/test_token_stream.py` | `analyze`, `main` |
| `test/test_VRAM.py` | Top-level VRAM training diagnostic |
| `test/test_VRAM_dummy_input.py` | Top-level VRAM training diagnostic |
| `test_hotpot.py` | Top-level HotpotQA generation diagnostic |
| `test_train.py` | `initialize_decision_head`, `freeze_backbone`, `print_parameter_summary`, `build_sample`, `ARTDataCollator` |
| `utils.py` | `GenerateDecoderOnlyOutput`, `GenerateEncoderDecoderOutput`, `GenerateBeamDecoderOnlyOutput`, `GenerateBeamEncoderDecoderOutput`, `GenerationMixin` |

### Surveyed (some prior excerpts)

| File | Purpose or evidence |
|---|---|
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/build_lm_loss_samples.py` | `create_lm_loss_samples`, `test_create_lm_loss_samples`, `test_zero_retrieval_input_consistency` |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/check_nan.py` | `check_file`, `main` |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/model_input.py` | `Document`, `SampleObject`, `SampleObjectForLMLoss`, `append_retrieval_observation`, `create_model_input` |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/plot_lm_loss_vs_decision_score.py` | `get_loss_columns`, `compute_curves`, `plot_curves`, `compute_total_curves`, `print_info` |
| `phase_1/stage_2_hypothesis_verification/H1_attention_modulation/run_lm_loss_evaluation.py` | Static symbol scan |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/check_nan.py` | `check_file`, `main` |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/plot_gradient_vs_decision_score.py` | `get_gradient_columns`, `compute_curves`, `plot_curves`, `compute_total_curves`, `print_info` |
| `phase_1/stage_2_hypothesis_verification/H2_gradient_direction/run_gradient_evaluation.py` | `GradientCapture` |
| `REVIVE_training/evaluate_art_as_tool.py` | `parse_args`, `get_output_file`, `extract_answer`, `load_results`, `main` |
| `REVIVE_training/evaluate_art_as_tool_generation.py` | Top-level second-stage answer generation from ART passages |
| `REVIVE_training/evaluate_art_pure_generation.py` | `seed_everything` |
| `REVIVE_training/evaluate_art_pure_generation_no_limit.py` | `seed_everything` |
| `REVIVE_training/evaluate_art_pure_with_demonstration_generation.py` | `load_demonstrations` |
| `REVIVE_training/evaluate_art_with_demonstration_as_tool_generation.py` | Top-level second-stage answer generation from ART passages |
| `REVIVE_training/full_training.py` | `set_seed`, `load_teacher_generated_training`, `build_combined_dataset`, `tokenize_training_sample`, `preprocess_dataset` |
| `REVIVE_training/get_demonstration.py` | `get_question`, `get_teacher_generation_output`, `main` |
| `REVIVE_training/test_model_after_training.py` | `main` |
| `REVIVE_training/train_fsdp2.py` | `get_rank`, `get_local_rank`, `get_world_size`, `is_main_process`, `rank_print` |
| `REVIVE_training/training_70.py` | `set_seed`, `load_teacher_generated_training`, `build_combined_dataset`, `tokenize_training_sample`, `preprocess_dataset` |
| `REVIVE_training/training_with_2_gpu.py` | `set_seed`, `load_teacher_generated_training`, `build_combined_dataset`, `tokenize_training_sample`, `preprocess_dataset` |
| `REVIVE_training/upload_to_hf.py` | `upload_model_to_hf` |
| `shared/ART_Models/ART_REVIVE/cache_utils.py` | `ARTCache`, `copy_dynamic_cache` |
| `shared/ART_Models/ART_REVIVE/decoder_layer.py` | `AttentionMixingProbeDecoderLayer` |
| `shared/ART_Models/ART_REVIVE/hotpotqa_retrieval_db.py` | `truncate_passages`, `append_retrieval_observation`, `HotpotQARetrievalDB` |
| `shared/ART_Models/ART_REVIVE/initialize_art_revive.py` | `print_names`, `verify_qwen_compatibility`, `verify_transferred_values`, `initialize_retrieval_decision_head`, `verify_retrieval_decision_head` |
| `shared/ART_Models/ART_REVIVE/initialize_art_revive_with_seed.py` | `set_seed`, `print_names`, `verify_qwen_compatibility`, `verify_transferred_values`, `initialize_retrieval_decision_head` |
| `shared/ART_Models/ART_REVIVE/musique_retrieval_db.py` | `truncate_passages`, `MuSiQueRetrievalDB`, `append_retrieval_observation`, `format_search_query` |
| `shared/ART_Models/ART_REVIVE/two_wikimultihopqa_db.py` | `truncate_passages`, `append_retrieval_observation`, `format_search_and_observation`, `TwoWikiMultiHopQARetrievalDB` |
| `shared/ART_Models/ART_REVIVE/utils.py` | `ensure_document_attention_mask`, `create_position_ids_without_doc`, `align_document_attention_mask`, `create_document_attention_mask`, `update_attention_mask` |
| `shared/HotpotQA_Distractor/build_hotpotqa_e5_index.py` | `load_model`, `pooling`, `load_corpus`, `encode_all`, `save_embeddings` |
| `shared/HotpotQA_Distractor/build_rsearch_hotpotqa_corpus.py` | `format_rsearch_document`, `process_record`, `build_hotpotqa_corpus`, `main` |
| `shared/HotpotQA_Distractor/download_rsearch_hotpotqa_corpus.py` | `main` |
| `shared/HotpotQA_Distractor/hotpotqa_sample.py` | `Document`, `HotpotQASample`, `HotpotQASampleConverter`, `test_hotpotqa_sample_converter` |
| `shared/HotpotQA_Distractor/rsearch_hotpotqa_test_dataset.py` | `get_rsearch_hotpotqa_path`, `download_rsearch_hotpotqa`, `load_rsearch_hotpotqa` |
| `shared/HotpotQA_Distractor/teacher_prompt.py` | `create_teacher_prompt`, `test_create_teacher_prompt` |
| `shared/HotpotQA_Distractor/upload_hotpotqa_index.py` | `main` |
| `shared/MuSiQue/build_musique_corpus.py` | `load_jsonl`, `load_all_musique`, `format_rsearch_document`, `build_musique_corpus`, `save_corpus` |
| `shared/MuSiQue/build_musique_e5_index.py` | `load_model`, `pooling`, `load_corpus`, `encode_all`, `save_embeddings` |
| `shared/MuSiQue/clean_teacher_training_dataset_v2.py` | `normalize_answer`, `extract_final_answer`, `check_sample`, `analyze_dataset`, `clean_config` |
| `shared/MuSiQue/clean_teacher_training_dataset_with_csv.py` | `normalize_answer`, `extract_final_answer`, `has_verbatim_question_prefix`, `remove_verbatim_question_prefix`, `check_sample` |
| `shared/MuSiQue/compare_r_search_nlp_book.py` | `format_size`, `sha256_file`, `get_file_size`, `count_lines`, `compare_file` |
| `shared/MuSiQue/dataset.py` | `get_musique_path`, `download_musique` |
| `shared/MuSiQue/hypothesis_validation_statistic.py` | `load_teacher_generated_hypothesis_validation`, `analyze_document_token_distribution` |
| `shared/MuSiQue/inspect_hop_samples.py` | `wrap_text`, `get_hop_count`, `find_sample_by_hop`, `print_sample`, `main` |
| `shared/MuSiQue/load_dataset.py` | `load_musique_ans` |
| `shared/MuSiQue/musique_hypothesis_validation_dataset_upload.py` | `load_teacher_dataset`, `validate_teacher_dataset`, `upload_hop_dataset`, `main` |
| `shared/MuSiQue/musique_hypothesis_validation_teacher_generation.py` | `set_seed`, `load_teacher_model`, `prepare_teacher_generation_samples`, `generate_teacher_outputs`, `save_jsonl` |
| `shared/MuSiQue/musique_sample.py` | `Document`, `QuestionDecompositionStep`, `MuSiQueSample`, `MuSiQueSampleConverter`, `test_musique_sample_converter` |
| `shared/MuSiQue/musique_training_teacher_generation.py` | `set_seed`, `parse_args`, `get_target_hops`, `load_teacher_model`, `clean_cuda_memory` |
| `shared/MuSiQue/musique_training_teacher_generation_v2.py` | `set_seed`, `parse_args`, `get_target_hops`, `load_teacher_model`, `clean_cuda_memory` |
| `shared/MuSiQue/normalize_teacher_hypothesis_validation_dataset.py` | `has_verbatim_question_prefix`, `remove_verbatim_question_prefix`, `count_question_prefixes`, `normalize_config`, `main` |
| `shared/MuSiQue/review_removed_samples_v2.py` | `load_removed_samples`, `safe_string`, `get_sample_data`, `get_question`, `get_sample_id` |
| `shared/MuSiQue/rsearch_musique_test_dataset.py` | `get_rsearch_musique_path`, `download_rsearch_musique`, `load_rsearch_musique` |
| `shared/MuSiQue/test.py` | `load_tokenizer`, `load_corpus`, `check_token_lengths`, `print_summary`, `main` |
| `shared/MuSiQue/test_qwen.py` | `load_tokenizer`, `load_corpus`, `check_token_lengths`, `print_summary`, `main` |
| `shared/MuSiQue/test_teacher_generation.py` | `load_teacher_model`, `test_single_sample` |
| `shared/MuSiQue/test_teacher_generation_v2.py` | `load_teacher_model`, `test_single_sample` |
| `shared/TwoWikiMultiHopQA/build_index_2wikimultihopqa.py` | `load_model`, `pooling`, `load_corpus`, `encode_all`, `save_embeddings` |
| `shared/TwoWikiMultiHopQA/build_rsearch_2wikimultihopqa_corups.py` | `format_rsearch_document`, `process_context`, `build_2wikimultihopqa_corpus`, `main` |
| `shared/TwoWikiMultiHopQA/download_rsearch_2wikimultihopqa.py` | `main` |
| `shared/TwoWikiMultiHopQA/rsearch_2wikimultihopqa_test_dataset.py` | `get_rsearch_2wikimultihopqa_path`, `download_rsearch_2wikimultihopqa`, `load_rsearch_2wikimultihopqa` |

### Traced

| File | Purpose or evidence |
|---|---|
| `REVIVE_training/evaluate_art_pure.py` | `parse_args`, `get_output_file`, `extract_answer`, `load_results`, `main` |
| `REVIVE_training/evaluate_art_pure_generation_top5.py` | `seed_everything` |
| `REVIVE_training/trainer.py` | `ARTTrainer` |
| `REVIVE_training/training.py` | `set_seed`, `load_teacher_generated_training`, `build_combined_dataset`, `tokenize_training_sample`, `preprocess_dataset` |
| `REVIVE_training/training_v2.py` | `set_seed`, `load_teacher_generated_training`, `build_combined_dataset`, `tokenize_training_sample`, `preprocess_dataset` |
| `REVIVE_training/utils.py` | `seed_everything`, `acc_score`, `normalize_answer`, `f1_score`, `qa_f1_score` |
| `shared/ART_Models/ART_REVIVE/attention_mixing.py` | `attention_mixing_forward`, `AttentionMixingProbeAttention` |
| `shared/ART_Models/ART_REVIVE/model.py` | `flip_coin`, `increase_probability`, `decrease_probability`, `compute_retrieval_cost`, `AttentionMixingProbePreTrainedModel` |
| `shared/ART_Models/ART_REVIVE/retrieval_context_processor.py` | `SentenceContextWindow`, `RetrievalContextProcessor` |
| `shared/MuSiQue/musique_teacher_prompt.py` | `create_teacher_prompt`, `test_create_teacher_prompt` |
| `shared/MuSiQue/musique_teacher_prompt_v2.py` | `load_teacher_demonstrations`, `create_system_prompt`, `create_user_instructions`, `format_golden_passages`, `format_question` |
