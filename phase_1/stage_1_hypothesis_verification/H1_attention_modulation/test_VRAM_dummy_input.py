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
    "dummy_vram_test"
)

DEVICE_ID = 0

BATCH_SIZE = 1
SEQ_LENGTH = 3000

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("ART — DUMMY INPUT VRAM TEST")
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
    # TRAINABLE PARAMS
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
    # DUMMY INPUT
    # ========================================================

    print()
    print("=" * 80)
    print("CREATING DUMMY INPUT")
    print("=" * 80)

    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(BATCH_SIZE, SEQ_LENGTH),
        dtype=torch.long,
    )

    attention_mask = torch.ones(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Document attention mask
    #
    # Shape:
    #   (B, L)
    #
    # Values:
    #   0       = allowed
    #   -inf    = masked
    # --------------------------------------------------------

    document_attention_mask = torch.zeros(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype=torch.float32,
    )

    # Example:
    # mask roughly half of the sequence.
    #
    # Change this if your real document mask has
    # a different structure.

    document_attention_mask[
        :,
        SEQ_LENGTH // 2:
    ] = float("-inf")

    # --------------------------------------------------------
    # Retrieval decision
    #
    # Shape:
    #   (B,)
    #
    # Must be strictly inside (0, 1)
    # --------------------------------------------------------

    retrieval_decision = torch.full(
        (BATCH_SIZE,),
        0.5,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Labels
    #
    # Standard causal LM setup.
    # Here every token contributes to LM loss.
    # --------------------------------------------------------

    labels = input_ids.clone()

    print()
    print(
        f"Batch size              : {BATCH_SIZE}"
    )

    print(
        f"Sequence length          : {SEQ_LENGTH}"
    )

    print(
        f"input_ids                : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"attention_mask           : "
        f"{tuple(attention_mask.shape)}"
    )

    print(
        f"document_attention_mask  : "
        f"{tuple(document_attention_mask.shape)}"
    )

    print(
        f"retrieval_decision       : "
        f"{tuple(retrieval_decision.shape)}"
    )

    print(
        f"retrieval_decision value : "
        f"{retrieval_decision[0].item():.4f}"
    )

    # ========================================================
    # PYTORCH → HF DATASET
    # ========================================================

    dataset_dict = {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "document_attention_mask":
            document_attention_mask.tolist(),
        "retrieval_decision":
            retrieval_decision.tolist(),
        "labels": labels.tolist(),
    }

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
        print(
            "!!! CUDA OUT OF MEMORY !!!"
        )

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
    print("Configuration:")

    print(
        f"  Batch size              : {BATCH_SIZE}"
    )

    print(
        f"  Sequence length         : {SEQ_LENGTH}"
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

    print(
        f"  Retrieval decision      : "
        f"{retrieval_decision[0].item():.4f}"
    )