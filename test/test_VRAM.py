import gc
import time

import torch

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from shared.MuSiQue.load_hypothesis_validation import (
    load_teacher_generated_hypothesis_validation,
)

from ..phase_1.stage_2_hypothesis_verification.H1_attention_modulation.build_lm_loss_samples import (
    create_lm_loss_samples,
)

from ..phase_1.stage_2_hypothesis_verification.H1_attention_modulation.model_input import (
    create_model_input_for_lm_loss,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-4B"

HF_REPO = (
    "adaptive-retrieval-timing/"
    "ART-Attention-Mixing-Probe"
)

OUTPUT_DIR = (
    "phase_1/stage_1_hypothesis_verification/"
    "H1_attention_modulation/"
    "layer_1_training_test"
)

DEVICE_ID = 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("ART — LAYER 1 TRAINING OOM TEST")
    print("=" * 80)

    device = torch.device(
        f"cuda:{DEVICE_ID}"
    )

    torch.cuda.set_device(device)

    print()
    print(
        f"GPU : {torch.cuda.get_device_name(device)}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        HF_REPO,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE_ID},
        trust_remote_code=True,
        attn_implementation="eager",
    )

    print("Model loaded.")

    # ========================================================
    # FREEZE EVERYTHING
    # ========================================================

    for param in model.parameters():
        param.requires_grad = False

    # ========================================================
    # UNFREEZE FIRST TRANSFORMER LAYER
    # ========================================================

    for param in model.model.layers[0].parameters():
        param.requires_grad = True

    # ========================================================
    # CHECK TRAINABLE PARAMS
    # ========================================================

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("=" * 80)
    print("PARAMETERS")
    print("=" * 80)

    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable ratio      : "
        f"{100 * trainable_params / total_params:.4f}%"
    )

    # ========================================================
    # GRADIENT CHECKPOINTING
    # ========================================================

    model.gradient_checkpointing_enable()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print()
    print(
        "Gradient checkpointing : ENABLED"
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print()
    print("Loading validation dataset...")

    datasets = (
        load_teacher_generated_hypothesis_validation()
    )

    # --------------------------------------------------------
    # Take the first hop and first original sample
    # --------------------------------------------------------

    hop = next(iter(datasets))

    sample = datasets[hop][0]

    print()
    print(
        f"Hop      : {hop}"
    )

    print(
        f"Question : {sample['question']}"
    )

    # ========================================================
    # CREATE 33 DECISION SAMPLES
    # ========================================================

    lm_loss_samples = (
        create_lm_loss_samples(sample)
    )

    ratio = next(iter(lm_loss_samples))

    lm_sample = lm_loss_samples[ratio]

    model_input = create_model_input_for_lm_loss(
        lm_sample,
        tokenizer,
    )

    print()
    print(
        f"Ratio : {ratio[0]}:{ratio[1]}"
    )

    print(
        f"Decision samples before slicing : "
        f"{model_input['input_ids'].shape[0]}"
    )

    # ========================================================
    # CRITICAL:
    # 33 → 1 SAMPLE
    # ========================================================

    model_input = {
        key: value[:1]
        if torch.is_tensor(value)
        else value
        for key, value in model_input.items()
    }

    print(
        f"Training samples after slicing  : "
        f"{model_input['input_ids'].shape[0]}"
    )

    print(
        f"Sequence length                  : "
        f"{model_input['input_ids'].shape[1]}"
    )

    print(
        f"Retrieval decision               : "
        f"{model_input['retrieval_decision'][0].flatten()[0].item():.6f}"
    )

    # ========================================================
    # PYTORCH → HF DATASET
    # ========================================================

    dataset_dict = {}

    for key, value in model_input.items():

        if torch.is_tensor(value):
            dataset_dict[key] = value.cpu().tolist()

    train_dataset = Dataset.from_dict(
        dataset_dict
    )

    # ========================================================
    # CLEAN CUDA MEMORY
    # ========================================================

    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # ========================================================
    # TRAINER
    # ========================================================

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=1,

        max_steps=1,

        learning_rate=1e-5,

        bf16=True,

        gradient_checkpointing=True,

        save_strategy="no",

        logging_steps=1,

        report_to="none",

        remove_unused_columns=False,

        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # ========================================================
    # MEMORY BEFORE
    # ========================================================

    torch.cuda.synchronize(device)

    memory_before = (
        torch.cuda.memory_allocated(device)
        / 1024**3
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 80)
    print("START TRAINING")
    print("=" * 80)

    print()
    print(
        "Running exactly 1 optimizer step..."
    )

    start_time = time.perf_counter()

    oom = False

    try:

        trainer.train()

    except torch.cuda.OutOfMemoryError:

        oom = True

        print()
        print("!!! CUDA OUT OF MEMORY !!!")

    torch.cuda.synchronize(device)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # MEMORY
    # ========================================================

    memory_after = (
        torch.cuda.memory_allocated(device)
        / 1024**3
    )

    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        / 1024**3
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print()

    if oom:
        print("Status       : ❌ OOM")
    else:
        print("Status       : ✅ NO OOM")

    print(
        f"Time         : {elapsed:.3f} s"
    )

    print(
        f"Memory before: {memory_before:.3f} GB"
    )

    print(
        f"Memory after : {memory_after:.3f} GB"
    )

    print(
        f"Peak memory  : {peak_memory:.3f} GB"
    )

    print()
    print(
        "Configuration:"
    )

    print(
        "  Batch size              : 1"
    )

    print(
        "  Decision samples        : 33 → 1"
    )

    print(
        "  Trainable layers        : layer 0 only"
    )

    print(
        "  Other parameters        : frozen"
    )

    print(
        "  Gradient checkpointing  : ON"
    )

    print(
        "  Precision               : BF16"
    )

    print(
        "  Optimizer steps         : 1"
    )