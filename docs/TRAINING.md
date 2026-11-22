# REVIVE training and teacher supervision

## Demo mappings and current scripts

| Item | Demo-2 | Demo-3 |
|---|---|---|
| Author-confirmed training entry point | [`training.py`](../REVIVE_training/training.py) | [`training_v2.py`](../REVIVE_training/training_v2.py) |
| Current source dataset | `adaptive-retrieval-timing/MuSiQue-ART-Training` | `adaptive-retrieval-timing/MuSiQue-ART-Training-v2` |
| Historical epochs (author) | **0.75** | **0.75** |
| Current source epochs | 0.1, accidental post-experiment edit (author) | 0.75 |
| Layer selection in current source | Last 40%, rounded up, plus decision head | Same selection, with explicit `last` region and validation |
| Target validation in current source | Builds target and checks combined lengths | Also rejects non-string and whitespace-only teacher targets |

Both historical checkpoints were uploaded to Hugging Face (author). The scripts currently load `adaptive-retrieval-timing/ART-REVIVE` with trusted remote code, plus the Qwen2.5-3B-Instruct tokenizer. That model repository literal does not identify the immutable revision used as an initializer or the repository and revision of either historical export.

## Shared training pipeline (source)

1. `load_teacher_generated_training` loads 2-, 3- and 4-hop training configurations. The entry point concatenates and shuffles the configurations with seed 42.
2. `tokenize_training_sample` formats a chat prompt for the question, tokenizes the question and teacher continuation separately, appends EOS to the target, and concatenates their attention masks. Labels mask all question tokens with `-100` and retain the continuation token IDs.
3. `ARTDataCollator` accepts one example, preserves the raw question for retrieval, and adds a batch dimension to token, mask and label tensors.
4. `freeze_backbone` initially freezes all parameters, enables the final ceil(0.40 × decoder depth) layers and enables the decision head. With 36 layers the selected indices are 21–35. The embedding and LM output weights stay frozen under this selection.
5. `ARTTrainer.training_step` collects retrieval trajectories with checkpointing temporarily disabled, restores that setting, and delegates differentiable replay and backpropagation through Hugging Face Trainer. See [methodology](METHODOLOGY.md).

The two current scripts use the same source defaults for seed 42, per-device batch 1, accumulation 32, learning rate 1e-5, weight decay 0, gradient clipping at 1, BF16, checkpointing, nominal threshold 0.4297, TensorBoard logging every step, checkpoint saves every 100 steps, and no Trainer evaluation. The actual gate lives in the loaded model implementation. Effective update counts depend on dataset size and runtime configuration. The scripts save the final model and tokenizer to `./art-training-output`.

## Teacher v1 to v2

The author created v2 to address limitations in v1 and improve supervision quality. These are source-observed changes in the pipeline.

| Dimension | v1 | v2 |
|---|---|---|
| Generator default | Qwen3-4B | Qwen3-8B |
| Prompt structure | One prompt with question, target answer, titled gold documents, reasoning instruction and task | Separate system and user prompts serialized through the chat template, with teacher demonstrations and a current-sample section |
| Gold answer and guidance | Target answer supplied and reasoning instructions guide the steps | Answer is supplementary and independently checked. Reasoning guidance may be adapted or rejected when unsupported |
| Evidence and reasoning | Follow dependencies and refer to exact document titles | Keeps title grounding and dependency order, distinguishes evidence, prior knowledge, inference and assumptions, and stops at the first unsupported dependency |
| Output conventions | Natural reasoning and a short answer in `<answer>` tags | Same tags, prescribed opening that avoids a question restatement, and a standardized insufficient-evidence response |
| Cleaning and review | `clean_teacher_training_dataset_with_csv.py` checks targets and can strip verbatim question prefixes | `clean_teacher_training_dataset_v2.py` validates outputs and gold-answer agreement, handles insufficient-evidence cases, and records removals for review |

Sources include [v1 prompt](../shared/MuSiQue/musique_teacher_prompt.py), [v2 prompt](../shared/MuSiQue/musique_teacher_prompt_v2.py), [v1 generator](../shared/MuSiQue/musique_training_teacher_generation.py), [v2 generator](../shared/MuSiQue/musique_training_teacher_generation_v2.py), [demonstration notes](../shared/MuSiQue/teacher_demonstrations.md) and [v2 cleaner](../shared/MuSiQue/clean_teacher_training_dataset_v2.py). In v2 the demonstration formatter includes question, dataset answer and reasoning while omitting golden passages from the example block. The current-sample formatter supplies its own passages. Teacher prompting and evaluation-time demonstration prompting are separate stages.

These changes bundle teacher size, prompt structure and data cleaning. The demo results provide a checkpoint-level comparison. They do not measure the separate contribution of any one teacher pipeline change or independently establish that every v2 continuation is higher quality.

The saved [v1 cleaning summary](../shared/MuSiQue/teacher_training_cleaning_statistics.csv) records 19,919 valid of 19,938 inputs and 10,082 cleaned question prefixes in `clean-and-upload` mode. The saved v2 dry-run summary at `shared/MuSiQue/data/teacher_training_cleaning_statistics_v2.csv` records 19,350 valid of 19,938 inputs and 588 removals. These are cleaner reports with different modes and criteria. Exact rows and counts for the historical training runs require dataset revision evidence.
