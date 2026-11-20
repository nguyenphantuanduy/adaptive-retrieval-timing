# Codebase map and workflows

## Repository map

| Path | Role and entry points |
|---|---|
| [`REVIVE_training/`](../REVIVE_training/) | Demo training entry points, `ARTTrainer`, checkpoint upload and post-training checks. `full_training.py`, `training_70.py`, `training_with_2_gpu.py` and `train_fsdp2.py` are separate entry points. Use the author mappings for demo attribution. |
| [`shared/ART_Models/ART_REVIVE/`](../shared/ART_Models/ART_REVIVE/) | Qwen2-derived decoder, decision head, generation and replay logic, masks, online sentence processing and dataset-specific retrieval adapters. |
| [`shared/ART_Models/`](../shared/ART_Models/) | `attention_mixing_probe` contains Qwen3-4B and Qwen2.5-3B probe implementations. `ART_demo_version`, `ART_HOPE` and `ART_MU` contain separate model branches. `demo_training/` and `HOPE_training/` have their own trainer, entry point and upload/check scripts. |
| [`shared/MuSiQue/`](../shared/MuSiQue/) | Original dataset acquisition and conversion, teacher prompt/generation, cleanup, hypothesis-validation datasets, corpus/index building, R-Search evaluation loading. |
| [`shared/HotpotQA_Distractor/`](../shared/HotpotQA_Distractor/), [`shared/TwoWikiMultiHopQA/`](../shared/TwoWikiMultiHopQA/) | R-Search test loaders, dataset-specific corpus construction and E5 indexing. HotpotQA also has sample and teacher prompt utilities. |
| [`phase_1/`](../phase_1/) | Literature notes for IRCoT, Self-RAG and R-Search, plus H1 attention modulation and H2 gradient direction experiments. |
| [`test/`](../test/), root `test_*.py` | Manual and diagnostic paths covering sentence segmentation, cache/model probes, gradient flow, indexing, data handling and VRAM. |
| [`temp_model/`](../temp_model/), [`art-backup/`](../art-backup/) | Local model copies or experiment artifacts and archived demo aggregates. The root `model.py` is another model implementation. Root `main.py` inspects teacher demonstration records. |

Runtime dependencies include PyTorch, Transformers (the current requirements file pins 5.15.1), Datasets, Accelerate, Hugging Face Hub, NumPy, FAISS CPU, sentencesplit, and TensorBoard. Training entry points use BF16 and check CUDA support. See [requirements.txt](../requirements.txt) and [GPU_SETUP.md](../GPU_SETUP.md). Package declarations describe the checkout and cannot recover the historical runtime.

## Data, index and model preparation

1. **Teacher data.** `dataset.py` acquires original MuSiQue v1.0. `load_dataset.load_musique_ans` loads its answerable splits. `MuSiQueSampleConverter` extracts question, answer, hop/decomposition and supporting documents for prompt generation. The two `musique_training_teacher_generation*` scripts generate and publish teacher outputs by hop. The cleaning scripts check answer formatting/agreement and provide statistics or rejected rows. Student training reads the published hop configurations. The hypothesis-validation generator and uploader are separate from student training data. See [teacher comparison](TRAINING.md).
2. **Retrieval corpus.** `build_musique_corpus.py` collects passages from answerable and full MuSiQue splits. `compare_r_search_nlp_book.py` has file-level comparison tools for original and R-Search MuSiQue data. HotpotQA's `download_rsearch_hotpotqa_corpus.py` and `build_rsearch_hotpotqa_corpus.py` transform R-Search corpus inputs. The corresponding 2Wiki scripts are `download_rsearch_2wikimultihopqa.py` and `build_rsearch_2wikimultihopqa_corups.py`, with the spelling used by the repository.
3. **Dense index.** `build_musique_e5_index.py`, `build_hotpotqa_e5_index.py` and `build_index_2wikimultihopqa.py` default to `intfloat/e5-base-v2`, a 512-token encoder limit and a CPU FAISS flat inner-product index. Builders encode passages with E5 inputs, masked mean pooling and normalization. Retrieval adapters use compatible query encoding and map ranked index rows back to the aligned corpus rows. The adapters format observations and can return passages and metadata. Preserve corpus/index row alignment when tracing an artifact.
4. **Model initialization.** The two `initialize_art_revive*.py` scripts load Qwen weights, check compatibility and transferred tensors, initialize the retrieval decision head, and package custom model code for Hugging Face. Resolve the actual model revision before reconstructing a run that uses `trust_remote_code=True`. In the current local REVIVE `model.py:399`, `_get_retriever` defaults to 2Wiki at construction, with other retriever choices commented out. This local default alone does not identify the corpus used by archived runs.
5. **Training.** `training.py` and `training_v2.py` flow through `ARTDataCollator`, `ARTTrainer.training_step`, `forward_collect`, and `forward_training`. They save weights and tokenizer. [Training configuration](TRAINING.md) records the historical epoch correction.

## Inference and scoring

The three `rsearch_*_test_dataset.py` loaders resolve evaluation files distributed in `qingfei1/R-Search_datasets`. The plain generators format the question in a chat prompt, call the ART causal LM through `generate`, and write the completion and retrieval diagnostics. The top-5 script currently imports the 2Wiki loader and requests five passages with `max_retrieval_context_tokens=-1`. It uses seed 44, a 1,024-token output cap, sampling temperature 0.1, top-p 0.9, sampling top-k 0 and one beam. Sampling top-k is independent of retrieval top-k. Local `forward_generate` defaults to three retrieval results and a 500-token passage-context argument. The no-limit variant overrides the context argument. The demonstrations variant adds examples to the ART prompt.

The as-tool generators use the passages saved by the first-stage ART run as context for a separate Qwen2.5-3B-Instruct answer generator. Read `evaluate_art_as_tool_generation.py` and `evaluate_art_with_demonstration_as_tool_generation.py` for the corresponding inputs. This path changes the answer-generation setup.

`evaluate_art_pure.py` and `evaluate_art_as_tool.py` extract the last complete `<answer>...</answer>` match or fall back to the stripped response. The metrics file records answer-tag coverage separately. `REVIVE_training/utils.py` implements case-insensitive gold-answer substring accuracy, normalized exact match, and normalized token-multiset F1. EM and F1 take the best score over available aliases and average across examples. Normalization lowercases, removes ASCII punctuation and English articles, and collapses whitespace. F1 uses multiset overlap for precision and recall, with score zero if there is no overlap. These metrics evaluate answers rather than the retrieved evidence.

Generators and scorers can write files. Scorers load result JSONL. The present audit used aggregate metrics and did not execute either stage.

## Alternate implementations and diagnostics

- `ART_HOPE` uses a separate decision branch and decision cache. `ARTConfig` defaults to a branch starting at decoder index 22 with 14 decision layers. The HOPE cost has an exponential term. Its training entry point unfreezes the decision layers, decision normalization and head. REVIVE's loss description applies specifically to REVIVE.
- REVIVE's `full_training.py` enables full fine-tuning, `training_70.py` uses the last 70% of decoder layers, and `training_with_2_gpu.py` dispatches modules via a device map. `train_fsdp2.py` uses a repeated-index sampler intended to deliver the same example to each rank under Accelerate sharding. Runtime behavior was outside the static audit.
- `debug_server.py` is a Streamlit reader for generations, queries, passages and matching original MuSiQue samples. Its default path omits the dataset directory present in the archived threshold runs. `shared/MuSiQue/review_removed_samples_v2.py` is a Streamlit review interface for per-sample cleaning results.
- Stage 1 includes a profiling notebook for Qwen generation and VRAM. Phase 1 H1 and H2 each have a saved aggregate CSV. H2's two cost-analysis notebooks explore a gradient-informed threshold and retrieval cost functions. The current `temp_model/model.py` uses a retrieval score coefficient of 0.1, with no verified linkage to the saved temporary-model metrics.

See the [file-level audit](CODEBASE_COVERAGE.md) for individual paths and review depth.
