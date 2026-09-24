from __future__ import annotations

import gc

import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_REPO = "adaptive-retrieval-timing/ART-Demo-Qwen2.5-3B"
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"

OUTPUT_DIR = "./art_training_test"

DTYPE = torch.bfloat16

SEED = 42

MAX_ANSWER_LENGTH = 512

LEARNING_RATE = 1e-5

MAX_STEPS = 1


# =============================================================================
# ONE TEST SAMPLE
# =============================================================================

QUESTION = (
    "Who was the person that started the Bethel branch of the religion "
    "founded by the black community in the former US capital?"
)

ANSWER = (
    "The religion referred to is the African Methodist Episcopal Church. "
    "The Bethel branch was started by Richard Allen, who founded Mother "
    "Bethel AME Church in Philadelphia."
)

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, explain your reasoning.
If additional information is needed, retrieval may be performed.
When you know the final answer, provide the final answer clearly."""


# =============================================================================
# INITIALIZE RETRIEVAL DECISION HEAD
# =============================================================================

def initialize_decision_head(model):
    """
    Initialize retrieval decision head so that the initial decision score
    is slightly above the current retrieval threshold.

    weight = 0
    bias   = 0.1

    sigmoid(0.1) ~= 0.525
    """

    print()
    print("=" * 80)
    print("INITIALIZING RETRIEVAL DECISION HEAD")
    print("=" * 80)

    with torch.no_grad():

        model.retrieval_decision_head.projection.weight.zero_()

        model.retrieval_decision_head.projection.bias.fill_(0.1)

    expected_score = torch.sigmoid(
        torch.tensor(
            0.1,
            dtype=torch.float32,
        )
    ).item()

    print("Weight initialized to : 0")
    print("Bias initialized to   : 0.1")
    print(f"Expected score        : {expected_score:.6f}")

    print("=" * 80)


# =============================================================================
# FREEZE BACKBONE
# =============================================================================

def freeze_backbone(model):
    """
    Train ONLY:

        1. Last decoder layer
        2. Retrieval decision head
        3. LM head / vocabulary projection

    Freeze:

        - All earlier decoder layers
        - Token embedding
        - Rotary embedding
        - Final norm

    For Qwen2.5-3B:

        36 decoder layers

        Frozen:
            layers 0 ... 34

        Trainable:
            layer 35
            retrieval decision head
            lm_head
    """

    print()
    print("=" * 80)
    print("FREEZING BACKBONE")
    print("=" * 80)

    layers = model.model.layers

    num_layers = len(layers)

    last_layer_idx = num_layers - 1

    print(f"Total decoder layers : {num_layers}")
    print(f"Trainable layer      : {last_layer_idx}")
    print()

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False

    print("Embedding: FROZEN")

    # -------------------------------------------------------------------------
    # Decoder layers
    # -------------------------------------------------------------------------

    for layer_idx, layer in enumerate(layers):

        is_trainable = layer_idx == last_layer_idx

        for param in layer.parameters():
            param.requires_grad = is_trainable

        status = (
            "TRAINABLE"
            if is_trainable
            else "FROZEN"
        )

        print(
            f"Layer {layer_idx:02d}: {status}"
        )

    # -------------------------------------------------------------------------
    # Final norm
    # -------------------------------------------------------------------------

    for param in model.model.norm.parameters():
        param.requires_grad = False

    print()
    print("Final norm: FROZEN")

    # -------------------------------------------------------------------------
    # Rotary embedding
    # -------------------------------------------------------------------------

    for param in model.model.rotary_emb.parameters():
        param.requires_grad = False

    print("Rotary embedding: FROZEN")

    # -------------------------------------------------------------------------
    # LM head / vocabulary projection
    # -------------------------------------------------------------------------

    for param in model.lm_head.parameters():
        param.requires_grad = True

    print("LM head / vocabulary projection: TRAINABLE")

    # -------------------------------------------------------------------------
    # Retrieval decision head
    # -------------------------------------------------------------------------

    for param in model.retrieval_decision_head.parameters():
        param.requires_grad = True

    print("Retrieval decision head: TRAINABLE")

    print("=" * 80)


# =============================================================================
# PARAMETER SUMMARY
# =============================================================================

def print_parameter_summary(model):

    total_params = sum(
        param.numel()
        for param in model.parameters()
    )

    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    frozen_params = total_params - trainable_params

    trainable_percentage = (
        100.0 * trainable_params / total_params
    )

    print()
    print("=" * 80)
    print("PARAMETER SUMMARY")
    print("=" * 80)

    print(
        f"Total parameters     : {total_params:,}"
    )

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    print(
        f"Frozen parameters    : {frozen_params:,}"
    )

    print(
        f"Trainable percentage : {trainable_percentage:.4f}%"
    )

    print("=" * 80)


# =============================================================================
# BUILD ONE SAMPLE
# =============================================================================

def build_sample(tokenizer):

    print()
    print("=" * 80)
    print("BUILDING ONE SAMPLE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Build question using Qwen chat template
    # -------------------------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": QUESTION,
        },
    ]

    question = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(question, torch.Tensor):
        question_input_ids = question
    else:
        question_input_ids = question["input_ids"]

    question_input_ids = question_input_ids[0]

    # -------------------------------------------------------------------------
    # Tokenize answer
    # -------------------------------------------------------------------------

    answer_input_ids = tokenizer(
        ANSWER,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ANSWER_LENGTH,
        add_special_tokens=False,
    )["input_ids"][0]

    # -------------------------------------------------------------------------
    # Question attention mask
    # -------------------------------------------------------------------------

    attention_mask = torch.ones(
        question_input_ids.shape[0],
        dtype=torch.long,
    )

    # -------------------------------------------------------------------------
    # Initial labels
    # -------------------------------------------------------------------------

    labels = torch.cat(
        [
            question_input_ids,
            answer_input_ids,
        ],
        dim=0,
    )

    print(
        f"Question tokens : {question_input_ids.shape[0]}"
    )

    print(
        f"Answer tokens   : {answer_input_ids.shape[0]}"
    )

    print(
        f"Initial labels  : {labels.shape[0]}"
    )

    print()

    return {
        "question": QUESTION,

        "question_input_ids": question_input_ids,

        "answer_input_ids": answer_input_ids,

        "attention_mask": attention_mask,

        "labels": labels,
    }


# =============================================================================
# DATA COLLATOR
# =============================================================================

class ARTDataCollator:

    def __call__(self, features):

        if len(features) != 1:
            raise ValueError(
                "This test only supports batch_size=1."
            )

        sample = features[0]

        question_input_ids = torch.as_tensor(
            sample["question_input_ids"],
            dtype=torch.long,
        )

        answer_input_ids = torch.as_tensor(
            sample["answer_input_ids"],
            dtype=torch.long,
        )

        attention_mask = torch.as_tensor(
            sample["attention_mask"],
            dtype=torch.long,
        )

        labels = torch.as_tensor(
            sample["labels"],
            dtype=torch.long,
        )

        return {
            "question": sample["question"],

            "question_input_ids": (
                question_input_ids.unsqueeze(0)
            ),

            "answer_input_ids": (
                answer_input_ids.unsqueeze(0)
            ),

            "attention_mask": (
                attention_mask.unsqueeze(0)
            ),

            "labels": (
                labels.unsqueeze(0)
            ),
        }


# =============================================================================
# GRADIENT INSPECTION
# =============================================================================

def inspect_gradients(model):

    print()
    print("=" * 80)
    print("GRADIENT INSPECTION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Retrieval decision head
    # -------------------------------------------------------------------------

    decision_weight = (
        model.retrieval_decision_head
        .projection
        .weight
    )

    decision_bias = (
        model.retrieval_decision_head
        .projection
        .bias
    )

    print()
    print("Retrieval decision head:")

    if decision_weight.grad is None:
        print("  Weight gradient : NONE")
    else:
        print(
            "  Weight gradient : "
            f"mean={decision_weight.grad.mean().item():+.6e}, "
            f"abs_mean={decision_weight.grad.abs().mean().item():.6e}, "
            f"max={decision_weight.grad.abs().max().item():.6e}"
        )

    if decision_bias.grad is None:
        print("  Bias gradient   : NONE")
    else:
        print(
            "  Bias gradient   : "
            f"{decision_bias.grad.detach().cpu().numpy()}"
        )

    # -------------------------------------------------------------------------
    # Last decoder layer
    # -------------------------------------------------------------------------

    layers = model.model.layers

    last_layer_idx = len(layers) - 1

    last_layer = layers[last_layer_idx]

    last_layer_grads = [
        param.grad
        for param in last_layer.parameters()
        if param.grad is not None
    ]

    print()
    print(
        f"Layer {last_layer_idx:02d} (TRAINABLE):"
    )

    if len(last_layer_grads) == 0:

        print("  Gradient: NONE")

    else:

        total_grad_norm = 0.0

        for grad in last_layer_grads:

            total_grad_norm += (
                grad.detach()
                .float()
                .norm()
                .item() ** 2
            )

        total_grad_norm = total_grad_norm ** 0.5

        print(
            f"  Gradient tensors : {len(last_layer_grads)}"
        )

        print(
            f"  Total grad norm  : {total_grad_norm:.6e}"
        )

    # -------------------------------------------------------------------------
    # First decoder layer
    # -------------------------------------------------------------------------

    first_layer = layers[0]

    first_layer_grads = [
        param.grad
        for param in first_layer.parameters()
        if param.grad is not None
    ]

    print()
    print("Layer 00 (FROZEN):")

    if len(first_layer_grads) == 0:

        print("  Gradient: NONE ✓")

    else:

        print(
            f"  Gradient tensors found: "
            f"{len(first_layer_grads)}"
        )

    # -------------------------------------------------------------------------
    # LM head
    # -------------------------------------------------------------------------

    lm_head_grads = [
        param.grad
        for param in model.lm_head.parameters()
        if param.grad is not None
    ]

    print()
    print("LM head / vocabulary projection:")

    if len(lm_head_grads) == 0:

        print("  Gradient: NONE")

    else:

        total_lm_grad_norm = 0.0

        for grad in lm_head_grads:

            total_lm_grad_norm += (
                grad.detach()
                .float()
                .norm()
                .item() ** 2
            )

        total_lm_grad_norm = (
            total_lm_grad_norm ** 0.5
        )

        print(
            f"  Gradient tensors : {len(lm_head_grads)}"
        )

        print(
            f"  Total grad norm  : "
            f"{total_lm_grad_norm:.6e}"
        )

    print("=" * 80)


# =============================================================================
# DECISION HEAD STATE
# =============================================================================

def inspect_decision_head(model, title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    weight = (
        model.retrieval_decision_head
        .projection
        .weight
    )

    bias = (
        model.retrieval_decision_head
        .projection
        .bias
    )

    print(
        f"Weight mean: "
        f"{weight.detach().float().mean().item():+.6e}"
    )

    print(
        f"Weight abs mean: "
        f"{weight.detach().float().abs().mean().item():.6e}"
    )

    print(
        f"Bias: "
        f"{bias.detach().float().cpu().numpy()}"
    )

    hidden_size = model.config.hidden_size

    zero_hidden = torch.zeros(
        1,
        1,
        hidden_size,
        device=weight.device,
        dtype=weight.dtype,
    )

    with torch.no_grad():

        decision_score = (
            model.retrieval_decision_head(
                zero_hidden
            )
        )

    print(
        f"Decision score: "
        f"{decision_score.detach().float().cpu().numpy()}"
    )

    print("=" * 80)


# =============================================================================
# GPU MEMORY
# =============================================================================

def print_gpu_memory(title):

    if not torch.cuda.is_available():
        return

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for gpu_idx in range(torch.cuda.device_count()):

        allocated = (
            torch.cuda.memory_allocated(gpu_idx)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(gpu_idx)
            / (1024 ** 3)
        )

        peak = (
            torch.cuda.max_memory_allocated(gpu_idx)
            / (1024 ** 3)
        )

        print(
            f"GPU {gpu_idx}: "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB, "
            f"peak={peak:.2f} GB"
        )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # 1. Seed
    # =========================================================================

    print()
    print("=" * 80)
    print("ART ONE-SAMPLE TRAINING TEST")
    print("=" * 80)

    print()
    print("1. Setting seed...")

    set_seed(SEED)

    # =========================================================================
    # 2. Device
    # =========================================================================

    print()
    print("2. Checking device...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =========================================================================
    # 3. Load tokenizer
    # =========================================================================

    print()
    print("3. Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO,
        trust_remote_code=True,
    )

    print(
        f"Tokenizer loaded: {TOKENIZER_REPO}"
    )

    # =========================================================================
    # 4. Load model
    # =========================================================================

    print()
    print("4. Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        dtype=DTYPE,
        attn_implementation="eager",
    )

    print(
        f"Model loaded: {MODEL_REPO}"
    )

    # =========================================================================
    # 5. Move model to device
    # =========================================================================

    print()
    print("5. Moving model to device...")

    model = model.to(device)

    # =========================================================================
    # Enable ART training mode
    # =========================================================================

    print()
    print("Enabling ART training mode...")

    model.enable_training_mode()

    print(
        f"model.isTraining = {model.isTraining}"
    )

    # =========================================================================
    # Initialize decision head
    # =========================================================================

    initialize_decision_head(model)

    # =========================================================================
    # Freeze backbone
    # =========================================================================

    freeze_backbone(model)

    # =========================================================================
    # Parameter summary
    # =========================================================================

    print_parameter_summary(model)

    # =========================================================================
    # 6. Build one sample
    # =========================================================================

    print()
    print("6. Building one sample...")

    sample = build_sample(tokenizer)

    # =========================================================================
    # Dataset
    # =========================================================================

    dataset = Dataset.from_list(
        [sample]
    )

    dataset = dataset.with_format(
        "torch",
        columns=[
            "question_input_ids",
            "answer_input_ids",
            "attention_mask",
            "labels",
        ],
        output_all_columns=True,
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    # =========================================================================
    # Verify dataset format
    # =========================================================================

    print()
    print("Checking dataset sample types...")

    dataset_sample = dataset[0]

    print(
        "question_input_ids :",
        type(dataset_sample["question_input_ids"]),
        tuple(dataset_sample["question_input_ids"].shape),
    )

    print(
        "answer_input_ids   :",
        type(dataset_sample["answer_input_ids"]),
        tuple(dataset_sample["answer_input_ids"].shape),
    )

    print(
        "attention_mask     :",
        type(dataset_sample["attention_mask"]),
        tuple(dataset_sample["attention_mask"].shape),
    )

    print(
        "labels              :",
        type(dataset_sample["labels"]),
        tuple(dataset_sample["labels"].shape),
    )

    # =========================================================================
    # 7. TrainingArguments
    # =========================================================================

    print()
    print("7. Creating TrainingArguments...")

    bf16_enabled = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print(
        f"BF16 enabled: {bf16_enabled}"
    )

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=1,

        max_steps=MAX_STEPS,

        learning_rate=LEARNING_RATE,

        lr_scheduler_type="constant",

        warmup_steps=0,

        optim="adamw_torch",

        weight_decay=0.0,

        bf16=bf16_enabled,

        fp16=False,

        # ---------------------------------------------------------------------
        # IMPORTANT:
        # NO gradient checkpointing.
        # ---------------------------------------------------------------------

        gradient_checkpointing=False,

        remove_unused_columns=False,

        logging_strategy="steps",

        logging_steps=1,

        logging_first_step=True,

        eval_strategy="no",

        save_strategy="no",

        report_to="none",

        dataloader_num_workers=0,

        dataloader_pin_memory=False,

        max_grad_norm=1.0,

        seed=SEED,

        data_seed=SEED,
    )

    # =========================================================================
    # 8. Trainer
    # =========================================================================

    print()
    print("8. Creating Trainer...")

    data_collator = ARTDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(
        "Trainer created successfully."
    )

    # =========================================================================
    # 9. Initial decision head
    # =========================================================================

    print()
    print("9. Checking initial decision head...")

    inspect_decision_head(
        model,
        "INITIAL DECISION HEAD",
    )

    # =========================================================================
    # GPU memory before training
    # =========================================================================

    print_gpu_memory(
        "GPU MEMORY BEFORE TRAINING"
    )

    # =========================================================================
    # 10. TRAIN
    # =========================================================================

    print()
    print("=" * 80)
    print("10. START TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    # =========================================================================
    # Training result
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAINING RESULT")
    print("=" * 80)

    print(
        f"Global step: "
        f"{train_result.global_step}"
    )

    print(
        f"Training loss: "
        f"{train_result.training_loss}"
    )

    print("=" * 80)

    # =========================================================================
    # Gradient inspection
    # =========================================================================

    inspect_gradients(model)

    # =========================================================================
    # Final decision head
    # =========================================================================

    inspect_decision_head(
        model,
        "FINAL DECISION HEAD",
    )

    # =========================================================================
    # ART state
    # =========================================================================

    print()
    print("=" * 80)
    print("ART TRAINING STATE")
    print("=" * 80)

    print(
        f"isTraining       : "
        f"{model.isTraining}"
    )

    print(
        f"retrieval_count  : "
        f"{model.retrieval_count}"
    )

    if hasattr(model, "retrieval_scores"):

        retrieval_scores = getattr(
            model,
            "retrieval_scores",
            [],
        )

        print(
            f"retrieval_scores : "
            f"{len(retrieval_scores)}"
        )

        for idx, score in enumerate(
            retrieval_scores
        ):

            if isinstance(score, torch.Tensor):

                score_value = (
                    score.detach()
                    .float()
                    .cpu()
                    .item()
                )

                print(
                    f"  [{idx}] "
                    f"{score_value:.6f}"
                )

    print("=" * 80)

    # =========================================================================
    # GPU memory after training
    # =========================================================================

    print_gpu_memory(
        "GPU MEMORY AFTER TRAINING"
    )

    # =========================================================================
    # Cleanup
    # =========================================================================

    print()
    print("Cleaning up...")

    del trainer
    del dataset
    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print("Done.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()