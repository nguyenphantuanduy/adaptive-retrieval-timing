from __future__ import annotations

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

FREEZE_RATIO = 0.60


# =============================================================================
# TEST SAMPLE
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
# DECISION HEAD INITIALIZATION
# =============================================================================

def initialize_decision_head(model):
    """
    Initialize the ART decision head so that:

        sigmoid(Wx + b) ~= 0.525

    for every hidden state.

    This is only for the one-sample training sanity check.
    """

    with torch.no_grad():

        model.retrieval_decision_head.projection.weight.zero_()

        model.retrieval_decision_head.projection.bias.fill_(0.1)

    expected_score = torch.sigmoid(
        torch.tensor(0.1)
    ).item()

    print("\n" + "=" * 80)
    print("DECISION HEAD INITIALIZATION")
    print("=" * 80)

    print("Weight : ZERO")
    print("Bias   : 0.1")

    print(
        f"Initial decision score: "
        f"{expected_score:.6f}"
    )

    print("=" * 80)


# =============================================================================
# FREEZE 60% OF DECODER LAYERS
# =============================================================================

def freeze_backbone(model):
    """
    Freeze the first 60% of decoder layers.

    Trainable:
        - last 40% decoder layers
        - final RMSNorm
        - RetrievalDecisionHead

    Frozen:
        - embedding
        - first 60% decoder layers
        - LM head
        - rotary embedding
    """

    layers = model.model.layers

    num_layers = len(layers)

    num_frozen_layers = int(
        num_layers * FREEZE_RATIO
    )

    print("\n" + "=" * 80)
    print("FREEZE CONFIGURATION")
    print("=" * 80)

    print(
        f"Total decoder layers : {num_layers}"
    )

    print(
        f"Freeze ratio         : "
        f"{FREEZE_RATIO:.0%}"
    )

    print(
        f"Frozen layers        : "
        f"{num_frozen_layers}"
    )

    print(
        f"Trainable layers     : "
        f"{num_layers - num_frozen_layers}"
    )

    # =========================================================================
    # Embedding
    # =========================================================================

    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False

    # =========================================================================
    # Decoder layers
    # =========================================================================

    for layer_idx, layer in enumerate(layers):

        frozen = (
            layer_idx < num_frozen_layers
        )

        for param in layer.parameters():
            param.requires_grad = not frozen

        status = (
            "FROZEN"
            if frozen
            else "TRAINABLE"
        )

        print(
            f"Layer {layer_idx:02d}: {status}"
        )

    # =========================================================================
    # Final normalization
    # =========================================================================

    for param in model.model.norm.parameters():
        param.requires_grad = True

    # =========================================================================
    # Rotary embedding
    # =========================================================================

    for param in model.model.rotary_emb.parameters():
        param.requires_grad = False

    # =========================================================================
    # LM head
    #
    # Qwen uses tied embeddings.
    # =========================================================================

    for param in model.lm_head.parameters():
        param.requires_grad = False

    # =========================================================================
    # ART decision head
    # =========================================================================

    for param in model.retrieval_decision_head.parameters():
        param.requires_grad = True

    print("=" * 80)


# =============================================================================
# PARAMETER SUMMARY
# =============================================================================

def print_parameter_summary(model):

    total_params = 0
    trainable_params = 0

    for param in model.parameters():

        num_params = param.numel()

        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

    percentage = (
        100.0 * trainable_params / total_params
    )

    print("\n" + "=" * 80)
    print("PARAMETER SUMMARY")
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
        f"Frozen parameters    : "
        f"{total_params - trainable_params:,}"
    )

    print(
        f"Trainable percentage : "
        f"{percentage:.4f}%"
    )

    print("=" * 80)


# =============================================================================
# BUILD ONE SAMPLE
# =============================================================================

def build_sample(tokenizer):

    # =========================================================================
    # Question
    # =========================================================================

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

    question_input_ids = (
        question_input_ids[0]
    )

    # =========================================================================
    # Answer
    # =========================================================================

    answer_input_ids = tokenizer(
        ANSWER,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_ANSWER_LENGTH,
        add_special_tokens=False,
    )["input_ids"][0]

    # =========================================================================
    # Initial attention mask
    #
    # Only question tokens exist at this point.
    #
    # forward_training() will extend this mask when:
    #
    #   question
    #   -> document
    #   -> sentence
    #   -> document
    #   -> ...
    #
    # =========================================================================

    question_length = (
        question_input_ids.shape[0]
    )

    attention_mask = torch.ones(
        question_length,
        dtype=torch.long,
    )

    # =========================================================================
    # Labels
    #
    # forward_training() expects labels covering:
    #
    #   question + answer
    #
    # and dynamically inserts -100 for retrieved documents.
    # =========================================================================

    labels = torch.cat(
        [
            question_input_ids,
            answer_input_ids,
        ],
        dim=0,
    )

    return {
        "question": QUESTION,

        "question_input_ids": (
            question_input_ids
        ),

        "answer_input_ids": (
            answer_input_ids
        ),

        "attention_mask": (
            attention_mask
        ),

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

        return {
            "question": sample["question"],

            "question_input_ids": (
                sample["question_input_ids"]
                .unsqueeze(0)
            ),

            "answer_input_ids": (
                sample["answer_input_ids"]
                .unsqueeze(0)
            ),

            "attention_mask": (
                sample["attention_mask"]
                .unsqueeze(0)
            ),

            "labels": (
                sample["labels"]
                .unsqueeze(0)
            ),
        }


# =============================================================================
# GRADIENT INSPECTION
# =============================================================================

def inspect_gradients(model):

    print("\n" + "=" * 80)
    print("GRADIENT INSPECTION")
    print("=" * 80)

    # =========================================================================
    # Decision head
    # =========================================================================

    print("\n[Decision Head]")

    for name, param in (
        model.retrieval_decision_head.named_parameters()
    ):

        if param.grad is None:

            print(
                f"{name}: grad=None"
            )

        else:

            grad = param.grad.detach().float()

            print(
                f"{name}: "
                f"shape={tuple(grad.shape)}, "
                f"mean={grad.mean().item():+.6e}, "
                f"abs_mean={grad.abs().mean().item():.6e}, "
                f"max={grad.abs().max().item():.6e}"
            )

    # =========================================================================
    # First frozen layer
    # =========================================================================

    print("\n[First Frozen Layer]")

    first_layer = model.model.layers[0]

    frozen_has_grad = False

    for name, param in first_layer.named_parameters():

        if param.grad is not None:

            frozen_has_grad = True

            print(
                f"WARNING: {name} has gradient!"
            )

    if not frozen_has_grad:

        print(
            "OK: no gradients."
        )

    # =========================================================================
    # First trainable layer
    # =========================================================================

    trainable_layer_idx = int(
        len(model.model.layers) * FREEZE_RATIO
    )

    print(
        f"\n[First Trainable Layer: "
        f"{trainable_layer_idx}]"
    )

    trainable_layer = model.model.layers[
        trainable_layer_idx
    ]

    trainable_has_grad = False

    for name, param in trainable_layer.named_parameters():

        if param.grad is not None:

            trainable_has_grad = True

            grad = param.grad.detach().float()

            print(
                f"{name}: "
                f"mean={grad.mean().item():+.6e}, "
                f"abs_mean={grad.abs().mean().item():.6e}"
            )

    if not trainable_has_grad:

        print(
            "WARNING: no gradients."
        )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed(SEED)

    print("=" * 80)
    print("ART — ONE SAMPLE TRAINING TEST")
    print("=" * 80)

    # =========================================================================
    # DEVICE
    # =========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    # =========================================================================
    # TOKENIZER
    # =========================================================================

    print("\n1. Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO,
        trust_remote_code=True,
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    print("\n2. Loading ART model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        dtype=DTYPE,
        attn_implementation="eager",
    )

    model = model.to(device)

    print(
        f"Model class: "
        f"{type(model).__name__}"
    )

    print(
        f"Model dtype: "
        f"{model.model.embed_tokens.weight.dtype}"
    )

    # =========================================================================
    # ART TRAINING MODE
    # =========================================================================

    print("\n3. Enabling ART training mode...")

    model.enable_training_mode()

    print(
        f"isTraining = "
        f"{model.isTraining}"
    )

    # =========================================================================
    # DECISION HEAD
    # =========================================================================

    print("\n4. Initializing decision head...")

    initialize_decision_head(model)

    # =========================================================================
    # FREEZE BACKBONE
    # =========================================================================

    print("\n5. Freezing backbone...")

    freeze_backbone(model)

    print_parameter_summary(model)

    # =========================================================================
    # ONE SAMPLE
    # =========================================================================

    print("\n6. Building one sample...")

    sample = build_sample(
        tokenizer
    )

    print(
        f"Question tokens : "
        f"{sample['question_input_ids'].shape[0]}"
    )

    print(
        f"Answer tokens   : "
        f"{sample['answer_input_ids'].shape[0]}"
    )

    print(
        f"Initial labels  : "
        f"{sample['labels'].shape[0]}"
    )

    # =========================================================================
    # DATASET
    # =========================================================================

    dataset = Dataset.from_list(
        [sample]
    )

    print(
        f"Dataset size: "
        f"{len(dataset)}"
    )

    # =========================================================================
    # TRAINING ARGUMENTS
    # =========================================================================

    print("\n7. Creating TrainingArguments...")

    bf16_enabled = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        # ---------------------------------------------------------------------
        # Batch
        # ---------------------------------------------------------------------

        per_device_train_batch_size=1,

        gradient_accumulation_steps=1,

        # ---------------------------------------------------------------------
        # One optimizer step
        # ---------------------------------------------------------------------

        max_steps=MAX_STEPS,

        # ---------------------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------------------

        learning_rate=LEARNING_RATE,

        lr_scheduler_type="constant",

        warmup_steps=0,

        optim="adamw_torch",

        weight_decay=0.0,

        # ---------------------------------------------------------------------
        # Precision
        # ---------------------------------------------------------------------

        bf16=bf16_enabled,

        fp16=False,

        # ---------------------------------------------------------------------
        # HF gradient checkpointing
        #
        # No custom checkpoint code is used.
        # ---------------------------------------------------------------------

        gradient_checkpointing=True,

        # ---------------------------------------------------------------------
        # Important because our model has custom inputs:
        #
        # question
        # question_input_ids
        # answer_input_ids
        # attention_mask
        # labels
        # ---------------------------------------------------------------------

        remove_unused_columns=False,

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------

        logging_strategy="steps",

        logging_steps=1,

        logging_first_step=True,

        # ---------------------------------------------------------------------
        # No evaluation / saving
        # ---------------------------------------------------------------------

        eval_strategy="no",

        save_strategy="no",

        # ---------------------------------------------------------------------
        # No external reporting
        # ---------------------------------------------------------------------

        report_to="none",

        # ---------------------------------------------------------------------
        # Dataloader
        # ---------------------------------------------------------------------

        dataloader_num_workers=0,

        dataloader_pin_memory=False,

        # ---------------------------------------------------------------------
        # Gradient clipping
        # ---------------------------------------------------------------------

        max_grad_norm=1.0,

        # ---------------------------------------------------------------------
        # Reproducibility
        # ---------------------------------------------------------------------

        seed=SEED,

        data_seed=SEED,
    )

    # =========================================================================
    # TRAINER
    # =========================================================================

    print("\n8. Creating Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ARTDataCollator(),
    )

    # =========================================================================
    # INITIAL DECISION SCORE
    # =========================================================================

    print("\n9. Checking initial decision head...")

    with torch.no_grad():

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
            f"{weight.float().mean().item():+.6e}"
        )

        print(
            f"Weight abs mean: "
            f"{weight.float().abs().mean().item():.6e}"
        )

        print(
            f"Bias: "
            f"{bias.float().tolist()}"
        )

        print(
            f"Decision score: "
            f"{torch.sigmoid(bias).float().tolist()}"
        )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print("\n" + "=" * 80)
    print("10. START TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    # =========================================================================
    # TRAINING RESULT
    # =========================================================================

    print("\n" + "=" * 80)
    print("TRAINING RESULT")
    print("=" * 80)

    print(
        f"Training loss: "
        f"{train_result.training_loss}"
    )

    print(
        f"Global step: "
        f"{trainer.state.global_step}"
    )

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    print("\n" + "=" * 80)
    print("ART RETRIEVAL STATE")
    print("=" * 80)

    print(
        f"Retrieval count: "
        f"{model.retrieval_count}"
    )

    print(
        f"Max retrievals: "
        f"{model.max_retrievals}"
    )

    print(
        f"isTraining: "
        f"{model.isTraining}"
    )

    print("=" * 80)

    # =========================================================================
    # GRADIENTS
    # =========================================================================

    inspect_gradients(model)

    # =========================================================================
    # FINAL DECISION HEAD
    # =========================================================================

    print("\n" + "=" * 80)
    print("FINAL DECISION HEAD")
    print("=" * 80)

    with torch.no_grad():

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
            f"{weight.float().mean().item():+.6e}"
        )

        print(
            f"Weight abs mean: "
            f"{weight.float().abs().mean().item():.6e}"
        )

        print(
            f"Bias: "
            f"{bias.float().tolist()}"
        )

        print(
            f"Decision score from bias: "
            f"{torch.sigmoid(bias).float().tolist()}"
        )

    # =========================================================================
    # GPU MEMORY
    # =========================================================================

    if device.type == "cuda":

        print("\n" + "=" * 80)
        print("GPU MEMORY")
        print("=" * 80)

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Allocated : "
            f"{allocated:.2f} GB"
        )

        print(
            f"Reserved  : "
            f"{reserved:.2f} GB"
        )

        print(
            f"Peak      : "
            f"{peak:.2f} GB"
        )

    print("\n" + "=" * 80)
    print("TEST FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()