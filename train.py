from __future__ import annotations

import random
from typing import Dict, Any, List

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
)

# Import ART model
# ------------------------------------------------------------
# Sửa path này nếu model class của bạn nằm ở file/module khác.
# ------------------------------------------------------------

# Import custom Trainer
# ------------------------------------------------------------
from trainer import ARTTrainer


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = (
    "adaptive-retrieval-timing/"
    "MuSiQue-ART-Training"
)

MODEL_REPO = (
    "adaptive-retrieval-timing/"
    "ART-Demo-Qwen2.5-3B"
)

TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"

OUTPUT_DIR = "./art-training-output"

SEED = 42

NUM_TRAIN_EPOCHS = 0.5

PER_DEVICE_TRAIN_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 32

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.0

MAX_GRAD_NORM = 1.0

LOGGING_STEPS = 1

SAVE_STEPS = 500

BF16 = True

# IMPORTANT:
# ART currently relies on KV cache during training.
GRADIENT_CHECKPOINTING = False


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step. 
For each step, start by explaining your thought process. 
If additional information is needed, the system will perform retrieval and return the top search results within <observation> and </observation>. Continue your reasoning using the retrieved information. 
When you know the final answer, provide the final answer within <answer> and </answer> tags."""


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD DATASET
# ============================================================

def load_teacher_generated_training() -> Dict[str, Any]:
    """
    Load teacher-generated training datasets.

    Returns
    -------
    Dict[str, Any]
        {
            "2-hop": train_split,
            "3-hop": train_split,
            "4-hop": train_split,
        }
    """

    datasets = {}

    for hop in (2, 3, 4):

        config_name = f"{hop}-hop"

        print()
        print("=" * 80)
        print(f"LOADING {config_name.upper()} DATASET")
        print("=" * 80)

        dataset = load_dataset(
            DATASET_NAME,
            config_name,
        )

        datasets[config_name] = dataset["train"]

        print(
            f"{config_name}: "
            f"{len(datasets[config_name]):,} samples"
        )

    return datasets


# ============================================================
# COMBINE + SHUFFLE
# ============================================================

def build_combined_dataset():
    """
    Combine the 2-hop, 3-hop and 4-hop training splits,
    then shuffle deterministically using seed 42.
    """

    datasets = load_teacher_generated_training()

    print()
    print("=" * 80)
    print("COMBINING TRAINING DATASETS")
    print("=" * 80)

    combined_dataset = concatenate_datasets(
        [
            datasets["2-hop"],
            datasets["3-hop"],
            datasets["4-hop"],
        ]
    )

    print(
        f"Combined dataset size: "
        f"{len(combined_dataset):,}"
    )

    print()
    print("=" * 80)
    print(f"SHUFFLING DATASET — SEED {SEED}")
    print("=" * 80)

    combined_dataset = combined_dataset.shuffle(
        seed=SEED
    )

    print("Dataset shuffled.")

    return combined_dataset


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_training_sample(
    sample: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:
    """
    Convert one teacher-generated sample into
    ART training inputs.

    Input
    -----
    sample["question"]
        Original question.

    sample["teacher_generation_output"]
        Teacher-generated reasoning + answer.

    Output
    ------
    question_input_ids
        Chat-templated system + user prompt.

    answer_input_ids
        Teacher generation followed by EOS.

    attention_mask
        Mask for the question tokens.

    labels
        Question positions = -100.
        Answer positions = answer token IDs.
    """

    question = sample["question"]

    teacher_generation_output = (
        sample["teacher_generation_output"]
    )

    # --------------------------------------------------------
    # QUESTION CHAT TEMPLATE
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    question_input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )

    # --------------------------------------------------------
    # TEACHER ANSWER + EOS
    # --------------------------------------------------------

    answer_text = (
        teacher_generation_output
        + tokenizer.eos_token
    )

    answer_input_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

    # --------------------------------------------------------
    # QUESTION ATTENTION MASK
    # --------------------------------------------------------

    question_attention_mask = [
        1
        for _ in question_input_ids
    ]

    # --------------------------------------------------------
    # LABELS
    #
    # Question tokens do not contribute to LM loss.
    # --------------------------------------------------------

    labels = (
        [-100] * len(question_input_ids)
        + answer_input_ids
    )

    return {
        "question_input_ids": question_input_ids,
        "answer_input_ids": answer_input_ids,
        "attention_mask": question_attention_mask,
        "labels": labels,
    }


# ============================================================
# PREPROCESS DATASET
# ============================================================

def preprocess_dataset(
    dataset,
    tokenizer,
):
    """
    Tokenize the complete training dataset.
    """

    print()
    print("=" * 80)
    print("TOKENIZING TRAINING DATASET")
    print("=" * 80)

    tokenized_dataset = dataset.map(
        lambda sample: tokenize_training_sample(
            sample,
            tokenizer,
        ),
        remove_columns=dataset.column_names,
        desc="Tokenizing ART training samples",
    )

    return tokenized_dataset


# ============================================================
# DATA COLLATOR
# ============================================================

class ARTDataCollator:
    """
    Collator for ART training.

    Each sample is processed independently by
    AttentionMixingProbeForCausalLM.forward_training().

    We intentionally keep batch size = 1.
    """

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:

        if len(features) != 1:
            raise ValueError(
                "ARTDataCollator expects "
                "per_device_train_batch_size=1."
            )

        feature = features[0]

        return {
            "question_input_ids": torch.tensor(
                feature["question_input_ids"],
                dtype=torch.long,
            ).unsqueeze(0),

            "answer_input_ids": torch.tensor(
                feature["answer_input_ids"],
                dtype=torch.long,
            ).unsqueeze(0),

            "attention_mask": torch.tensor(
                feature["attention_mask"],
                dtype=torch.long,
            ).unsqueeze(0),

            "labels": torch.tensor(
                feature["labels"],
                dtype=torch.long,
            ).unsqueeze(0),
        }


# ============================================================
# INITIALIZE DECISION HEAD
# ============================================================

def initialize_decision_head(model) -> None:
    """
    Initialize retrieval decision head so that the initial
    decision score is approximately sigmoid(0.1) = 0.525.
    """

    print()
    print("=" * 80)
    print("INITIALIZING RETRIEVAL DECISION HEAD")
    print("=" * 80)

    with torch.no_grad():

        model.retrieval_decision_head.projection.weight.zero_()

        model.retrieval_decision_head.projection.bias.fill_(
            0.1
        )

    expected_score = torch.sigmoid(
        torch.tensor(0.1)
    ).item()

    print(
        f"Weight initialized to : "
        f"{model.retrieval_decision_head.projection.weight.mean().item():.6f}"
    )

    print(
        f"Bias initialized to   : "
        f"{model.retrieval_decision_head.projection.bias.item():.6f}"
    )

    print(
        f"Expected score        : "
        f"{expected_score:.6f}"
    )


# ============================================================
# FREEZE BACKBONE
# ============================================================

def freeze_backbone(model) -> None:
    """
    Train only:

        - last decoder layer
        - retrieval decision head

    Everything else is frozen.

    Gradient checkpointing is disabled because ART training
    relies on KV cache.
    """

    print()
    print("=" * 80)
    print("FREEZING BACKBONE")
    print("=" * 80)

    # --------------------------------------------------------
    # Freeze everything first
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Last decoder layer
    # --------------------------------------------------------

    last_layer = model.model.layers[-1]

    for parameter in last_layer.parameters():
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Retrieval decision head
    # --------------------------------------------------------

    for parameter in model.retrieval_decision_head.parameters():
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    num_layers = len(model.model.layers)

    print(
        f"Total decoder layers : {num_layers}"
    )

    print()

    for index, layer in enumerate(model.model.layers):

        trainable = any(
            parameter.requires_grad
            for parameter in layer.parameters()
        )

        print(
            f"Layer {index:02d}: "
            f"{'TRAINABLE' if trainable else 'FROZEN'}"
        )

    print()

    print(
        "Retrieval decision head: TRAINABLE"
    )

    print(
        "LM head: FROZEN"
    )

    print(
        "Embedding: FROZEN"
    )

    print(
        "Final norm: FROZEN"
    )

    print(
        "Rotary embedding: FROZEN"
    )


# ============================================================
# PARAMETER SUMMARY
# ============================================================

def print_parameter_summary(model) -> None:

    total_parameters = 0
    trainable_parameters = 0

    for parameter in model.parameters():

        count = parameter.numel()

        total_parameters += count

        if parameter.requires_grad:
            trainable_parameters += count

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    trainable_percentage = (
        100.0
        * trainable_parameters
        / total_parameters
    )

    print()
    print("=" * 80)
    print("PARAMETER SUMMARY")
    print("=" * 80)

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen parameters    : "
        f"{frozen_parameters:,}"
    )

    print(
        f"Trainable percentage : "
        f"{trainable_percentage:.4f}%"
    )


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory(
    title: str,
) -> None:

    if not torch.cuda.is_available():
        return

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for gpu_index in range(
        torch.cuda.device_count()
    ):

        allocated = (
            torch.cuda.memory_allocated(
                gpu_index
            )
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(
                gpu_index
            )
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated(
                gpu_index
            )
            / 1024**3
        )

        print(
            f"GPU {gpu_index}: "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB, "
            f"peak={peak:.2f} GB"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("ART TRAINING")
    print("=" * 80)

    # ========================================================
    # 1. SEED
    # ========================================================

    print()
    print("1. Setting seed...")

    set_seed(SEED)

    # ========================================================
    # 2. DEVICE
    # ========================================================

    print()
    print("2. Checking device...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # 3. TOKENIZER
    # ========================================================

    print()
    print("3. Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO,
        trust_remote_code=True,
    )

    print(
        f"Tokenizer loaded: "
        f"{TOKENIZER_REPO}"
    )

    if tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer does not have an EOS token."
        )

    print(
        f"EOS token: "
        f"{repr(tokenizer.eos_token)}"
    )

    print(
        f"EOS token ID: "
        f"{tokenizer.eos_token_id}"
    )

    # ========================================================
    # 4. DATASET
    # ========================================================

    print()
    print("4. Loading training dataset...")

    raw_dataset = build_combined_dataset()

    print()
    print(
        f"Final shuffled dataset size: "
        f"{len(raw_dataset):,}"
    )

    # ========================================================
    # 5. TOKENIZE DATASET
    # ========================================================

    print()
    print("5. Preprocessing dataset...")

    train_dataset = preprocess_dataset(
        raw_dataset,
        tokenizer,
    )

    print(
        f"Tokenized training samples: "
        f"{len(train_dataset):,}"
    )

    # ========================================================
    # 6. INSPECT FIRST SAMPLE
    # ========================================================

    print()
    print("=" * 80)
    print("FIRST TOKENIZED SAMPLE")
    print("=" * 80)

    first_sample = train_dataset[0]

    print(
        f"Question tokens : "
        f"{len(first_sample['question_input_ids'])}"
    )

    print(
        f"Answer tokens   : "
        f"{len(first_sample['answer_input_ids'])}"
    )

    print(
        f"Label tokens    : "
        f"{len(first_sample['labels'])}"
    )

    print()

    # ========================================================
    # 7. LOAD MODEL
    # ========================================================

    print()
    print("6. Loading model...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    )

    print(
        f"Model loaded: "
        f"{MODEL_REPO}"
    )

    # ========================================================
    # 8. MOVE MODEL
    # ========================================================

    print()
    print("7. Moving model to device...")

    model = model.to(device)

    # ========================================================
    # 9. ART TRAINING MODE
    # ========================================================

    print()
    print("Enabling ART training mode...")

    model.enable_training_mode()

    print(
        f"model.isTraining = "
        f"{model.isTraining}"
    )

    # ========================================================
    # 10. DISABLE GRADIENT CHECKPOINTING
    # ========================================================

    print()
    print("=" * 80)
    print("CONFIGURING GRADIENT CHECKPOINTING / CACHE")
    print("=" * 80)

    if hasattr(
        model,
        "gradient_checkpointing_disable",
    ):
        model.gradient_checkpointing_disable()

    model.config.use_cache = True

    print(
        "Gradient checkpointing: DISABLED"
    )

    print(
        f"model.config.use_cache: "
        f"{model.config.use_cache}"
    )

    # ========================================================
    # 11. INITIALIZE DECISION HEAD
    # ========================================================

    initialize_decision_head(model)

    # ========================================================
    # 12. FREEZE BACKBONE
    # ========================================================

    freeze_backbone(model)

    print_parameter_summary(model)

    # ========================================================
    # 13. DATA COLLATOR
    # ========================================================

    print()
    print("8. Creating data collator...")

    data_collator = ARTDataCollator()

    print(
        "Data collator created."
    )

    # ========================================================
    # 14. TRAINING ARGUMENTS
    # ========================================================

    print()
    print("9. Creating TrainingArguments...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # ----------------------------------------------------
        # Dataset / epochs
        # ----------------------------------------------------

        num_train_epochs=NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        max_grad_norm=MAX_GRAD_NORM,

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        bf16=BF16,

        # ----------------------------------------------------
        # IMPORTANT FOR ART
        # ----------------------------------------------------

        gradient_checkpointing=(
            GRADIENT_CHECKPOINTING
        ),

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logging_strategy="steps",

        logging_steps=LOGGING_STEPS,

        logging_first_step=True,

        # ----------------------------------------------------
        # Saving
        # ----------------------------------------------------

        save_strategy="steps",

        save_steps=SAVE_STEPS,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=SEED,

        data_seed=SEED,

        # ----------------------------------------------------
        # Dataset columns
        # ----------------------------------------------------

        remove_unused_columns=False,

        # ----------------------------------------------------
        # Reporting
        # ----------------------------------------------------

        report_to="none",

        # ----------------------------------------------------
        # Dataloader
        # ----------------------------------------------------

        dataloader_num_workers=0,

        # ----------------------------------------------------
        # No evaluation
        # ----------------------------------------------------

        eval_strategy="no",
    )

    print(
        f"Epochs: "
        f"{NUM_TRAIN_EPOCHS}"
    )

    print(
        f"Per-device batch size: "
        f"{PER_DEVICE_TRAIN_BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Effective batch size: "
        f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Gradient checkpointing: "
        f"{GRADIENT_CHECKPOINTING}"
    )

    # ========================================================
    # 15. TRAINER
    # ========================================================

    print()
    print("10. Creating ART Trainer...")

    trainer = ARTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Tell HF Trainer that our model does NOT use
    # num_items_in_batch.
    #
    # This also makes Trainer perform the normal
    # gradient-accumulation scaling.
    # --------------------------------------------------------

    trainer.model_accepts_loss_kwargs = False

    print(
        "Trainer created successfully."
    )

    print(
        "num_items_in_batch: DISABLED"
    )

    # ========================================================
    # 16. MEMORY BEFORE TRAINING
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

        print_gpu_memory(
            "GPU MEMORY BEFORE TRAINING"
        )

    # ========================================================
    # 17. TRAIN
    # ========================================================

    print()
    print("=" * 80)
    print("11. START TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    # ========================================================
    # 18. TRAINING RESULT
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING RESULT")
    print("=" * 80)

    print(
        f"Global step: "
        f"{trainer.state.global_step}"
    )

    print(
        f"Training loss: "
        f"{train_result.training_loss}"
    )

    # ========================================================
    # 19. MEMORY AFTER TRAINING
    # ========================================================

    if torch.cuda.is_available():

        print_gpu_memory(
            "GPU MEMORY AFTER TRAINING"
        )

    # ========================================================
    # 20. SAVE FINAL MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print(
        f"Saved to: "
        f"{OUTPUT_DIR}"
    )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()