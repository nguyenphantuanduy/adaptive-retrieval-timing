
from __future__ import annotations

import gc
import math
import random
from typing import Dict, Any, List

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
)

from REVIVE_training.trainer import ARTTrainer


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = (
    "adaptive-retrieval-timing/"
    "MuSiQue-ART-Training-v2"
)

MODEL_REPO = (
    "adaptive-retrieval-timing/"
    "ART-REVIVE"
)

TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"

OUTPUT_DIR = "./art-training-output"

SEED = 42


# ------------------------------------------------------------
# Training epochs
# ------------------------------------------------------------

NUM_TRAIN_EPOCHS = 0.75


# ------------------------------------------------------------
# Percentage of decoder layers to unfreeze
# ------------------------------------------------------------

UNFREEZE_LAYER_PERCENTAGE = 0.40

# Available: "last"
UNFREEZE_LAYER_REGION = "last"


# ------------------------------------------------------------
# Batch
# ------------------------------------------------------------

PER_DEVICE_TRAIN_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 32


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.0

MAX_GRAD_NORM = 1.0


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

LOGGING_STEPS = 1


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------

SAVE_STEPS = 100


# ------------------------------------------------------------
# Precision
# ------------------------------------------------------------

BF16 = True


# ------------------------------------------------------------
# Gradient checkpointing
# ------------------------------------------------------------

GRADIENT_CHECKPOINTING = True


# ------------------------------------------------------------
# Retrieval threshold
# ------------------------------------------------------------

THRESHOLD = 0.4297


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
    Load teacher-generated training datasets from v2.

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
    then shuffle deterministically.
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
    Convert one teacher-generated sample into ART training inputs.
    """

    question = sample["question"]

    teacher_generation_output = (
        sample["teacher_generation_output"]
    )

    if not isinstance(teacher_generation_output, str):
        raise ValueError(
            "Invalid teacher_generation_output: "
            f"expected str, got "
            f"{type(teacher_generation_output).__name__}"
        )

    if not teacher_generation_output.strip():
        raise ValueError(
            "Empty teacher_generation_output."
        )

    # ========================================================
    # 1. BUILD CHAT-TEMPLATED QUESTION STRING
    # ========================================================

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

    question_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ========================================================
    # 2. TOKENIZE QUESTION
    # ========================================================

    question_tokenized = tokenizer(
        question_text,
        add_special_tokens=False,
        return_tensors=None,
    )

    question_input_ids = (
        question_tokenized["input_ids"]
    )

    # ========================================================
    # 3. TEACHER ANSWER + EOS
    # ========================================================

    answer_text = (
        teacher_generation_output
        + tokenizer.eos_token
    )

    answer_tokenized = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors=None,
    )

    answer_input_ids = (
        answer_tokenized["input_ids"]
    )

    # ========================================================
    # 4. ATTENTION MASK
    # ========================================================

    question_attention_mask = [
        1 for _ in question_input_ids
    ]

    answer_attention_mask = [
        1 for _ in answer_input_ids
    ]

    attention_mask = (
        question_attention_mask
        + answer_attention_mask
    )

    # ========================================================
    # 5. LABELS
    # ========================================================

    labels = (
        [-100] * len(question_input_ids)
        + answer_input_ids
    )

    # ========================================================
    # 6. SANITY CHECKS
    # ========================================================

    expected_length = (
        len(question_input_ids)
        + len(answer_input_ids)
    )

    if len(attention_mask) != expected_length:
        raise ValueError(
            "attention_mask length mismatch."
        )

    if len(labels) != expected_length:
        raise ValueError(
            "labels length mismatch."
        )

    # ========================================================
    # 7. RETURN
    # ========================================================

    return {
        "question": question,
        "question_input_ids": question_input_ids,
        "answer_input_ids": answer_input_ids,
        "attention_mask": attention_mask,
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

    Current ART implementation operates one sample at a time,
    therefore batch_size must be 1.
    """

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if len(features) != 1:
            raise ValueError(
                "ARTDataCollator expects "
                "per_device_train_batch_size=1."
            )

        feature = features[0]

        question_input_ids = torch.as_tensor(
            feature["question_input_ids"],
            dtype=torch.long,
        )

        answer_input_ids = torch.as_tensor(
            feature["answer_input_ids"],
            dtype=torch.long,
        )

        attention_mask = torch.as_tensor(
            feature["attention_mask"],
            dtype=torch.long,
        )

        labels = torch.as_tensor(
            feature["labels"],
            dtype=torch.long,
        )

        expected_length = (
            question_input_ids.shape[0]
            + answer_input_ids.shape[0]
        )

        if attention_mask.shape[0] != expected_length:
            raise ValueError(
                "Attention mask length mismatch: "
                f"expected={expected_length}, "
                f"actual={attention_mask.shape[0]}"
            )

        if labels.shape[0] != expected_length:
            raise ValueError(
                "Labels length mismatch: "
                f"expected={expected_length}, "
                f"actual={labels.shape[0]}"
            )

        return {
            "question": feature["question"],

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


# ============================================================
# FREEZE BACKBONE
# ============================================================

def freeze_backbone(model) -> None:
    """
    Freeze the entire model first, then unfreeze:

        - the last 40% of decoder layers
        - retrieval decision head

    Example:
        36 decoder layers
        40% = 14.4
        ceil(14.4) = 15

        Last 15 layers:
            layers 21-35

    Everything else is frozen.
    """

    print()
    print("=" * 80)
    print("FREEZING / UNFREEZING BACKBONE")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Freeze everything first.
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # 2. Get decoder layers.
    # --------------------------------------------------------

    layers = model.model.layers

    num_layers = len(layers)

    if num_layers == 0:
        raise RuntimeError(
            "No decoder layers found."
        )

    # --------------------------------------------------------
    # 3. Calculate number of layers to unfreeze.
    # --------------------------------------------------------

    num_layers_to_unfreeze = math.ceil(
        num_layers * UNFREEZE_LAYER_PERCENTAGE
    )

    if not 0 < num_layers_to_unfreeze <= num_layers:
        raise ValueError(
            "Invalid number of layers to unfreeze: "
            f"{num_layers_to_unfreeze}"
        )

    # --------------------------------------------------------
    # 4. Determine unfreeze region.
    # --------------------------------------------------------

    if UNFREEZE_LAYER_REGION == "last":

        first_trainable_layer = (
            num_layers - num_layers_to_unfreeze
        )

        last_trainable_layer = num_layers - 1

    else:
        raise ValueError(
            "Unsupported unfreeze region: "
            f"{UNFREEZE_LAYER_REGION}"
        )

    # --------------------------------------------------------
    # 5. Unfreeze selected decoder layers.
    # --------------------------------------------------------

    for layer_index in range(
        first_trainable_layer,
        last_trainable_layer + 1,
    ):

        for parameter in layers[layer_index].parameters():
            parameter.requires_grad = True

    # --------------------------------------------------------
    # 6. Unfreeze retrieval decision head.
    # --------------------------------------------------------

    for parameter in (
        model.retrieval_decision_head.parameters()
    ):
        parameter.requires_grad = True

    # --------------------------------------------------------
    # 7. Report configuration.
    # --------------------------------------------------------

    actual_percentage = (
        100.0
        * num_layers_to_unfreeze
        / num_layers
    )

    print(
        f"Total decoder layers       : "
        f"{num_layers}"
    )

    print(
        f"Requested unfreeze         : "
        f"{UNFREEZE_LAYER_PERCENTAGE * 100:.1f}%"
    )

    print(
        f"Unfreeze region             : "
        f"{UNFREEZE_LAYER_REGION.upper()}"
    )

    print(
        f"Number of trainable layers : "
        f"{num_layers_to_unfreeze}"
    )

    print(
        f"Actual unfreeze percentage : "
        f"{actual_percentage:.2f}%"
    )

    print(
        f"First trainable layer      : "
        f"{first_trainable_layer}"
    )

    print(
        f"Last trainable layer       : "
        f"{last_trainable_layer}"
    )

    print()

    # --------------------------------------------------------
    # 8. Print every decoder layer.
    # --------------------------------------------------------

    for index, layer in enumerate(layers):

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

    print("=" * 80)


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
        total_parameters - trainable_parameters
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

    print("=" * 80)


# ============================================================
# RETRIEVAL DECISION HEAD HELPERS
# ============================================================

def get_decision_head_linear_layers(model):
    """
    Return all Linear layers inside the ART retrieval decision head.
    """

    linear_layers = [
        module
        for module in model.retrieval_decision_head.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    if not linear_layers:
        raise RuntimeError(
            "No Linear layers found inside "
            "retrieval_decision_head."
        )

    return linear_layers


def print_decision_head_summary(
    model,
    title: str,
) -> None:
    """
    Print architecture and parameter statistics for the complete
    retrieval decision MLP.
    """

    linear_layers = get_decision_head_linear_layers(
        model
    )

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"Number of Linear layers: "
        f"{len(linear_layers)}"
    )

    total_head_parameters = 0

    for index, layer in enumerate(linear_layers):

        layer_parameters = layer.weight.numel()

        if layer.bias is not None:
            layer_parameters += layer.bias.numel()

        total_head_parameters += layer_parameters

        print()
        print(
            f"Linear {index + 1}: "
            f"{layer.in_features} → "
            f"{layer.out_features}"
        )

        print(
            f"  Weight shape    : "
            f"{tuple(layer.weight.shape)}"
        )

        print(
            f"  Weight abs mean : "
            f"{layer.weight.detach().float().abs().mean().item():.6e}"
        )

        print(
            f"  Weight std      : "
            f"{layer.weight.detach().float().std().item():.6e}"
        )

        if layer.bias is not None:

            print(
                f"  Bias mean       : "
                f"{layer.bias.detach().float().mean().item():+.6e}"
            )

            print(
                f"  Bias std        : "
                f"{layer.bias.detach().float().std().item():.6e}"
            )

        print(
            f"  Parameters      : "
            f"{layer_parameters:,}"
        )

    print()
    print(
        f"Total decision head parameters: "
        f"{total_head_parameters:,}"
    )

    print("=" * 80)


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
            torch.cuda.memory_allocated(gpu_index)
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(gpu_index)
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated(gpu_index)
            / 1024**3
        )

        print(
            f"GPU {gpu_index}: "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB, "
            f"peak={peak:.2f} GB"
        )

    print("=" * 80)


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
        f"EOS token: {repr(tokenizer.eos_token)}"
    )

    print(
        f"EOS token ID: {tokenizer.eos_token_id}"
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

    question_length = len(
        first_sample["question_input_ids"]
    )

    answer_length = len(
        first_sample["answer_input_ids"]
    )

    attention_length = len(
        first_sample["attention_mask"]
    )

    label_length = len(
        first_sample["labels"]
    )

    print(f"Question tokens : {question_length}")
    print(f"Answer tokens   : {answer_length}")
    print(f"Attention mask  : {attention_length}")
    print(f"Label tokens    : {label_length}")

    print()

    expected_length = (
        question_length + answer_length
    )

    print(
        f"Expected full sequence length: "
        f"{expected_length}"
    )

    if attention_length != expected_length:
        raise ValueError(
            "Attention mask does not cover "
            "question + answer."
        )

    if label_length != expected_length:
        raise ValueError(
            "Labels do not cover "
            "question + answer."
        )

    print(
        "✓ Attention mask = "
        "question mask + answer mask"
    )

    print(
        "✓ Labels = "
        "[-100] question + answer IDs"
    )

    print("=" * 80)

    # ========================================================
    # 7. LOAD MODEL
    # ========================================================

    print()
    print("6. Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    print(f"Model loaded: {MODEL_REPO}")

    print(
        f"Model hidden size: "
        f"{model.config.hidden_size}"
    )

    # ========================================================
    # 8. MOVE MODEL
    # ========================================================

    print()
    print("7. Moving model to device...")

    model = model.to(device)

    print(f"Model device: {device}")

    # ========================================================
    # 9. VERIFY LOADED DECISION HEAD
    # ========================================================

    print_decision_head_summary(
        model=model,
        title="LOADED RETRIEVAL DECISION HEAD",
    )

    # ========================================================
    # 10. VERIFY INITIAL DECISION OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("VERIFYING INITIAL RETRIEVAL DECISION HEAD")
    print("=" * 80)

    hidden_size = model.config.hidden_size

    linear_layers = get_decision_head_linear_layers(
        model
    )

    first_weight = linear_layers[0].weight

    test_hidden = torch.randn(
        8,
        1,
        hidden_size,
        device=first_weight.device,
        dtype=first_weight.dtype,
    )

    with torch.no_grad():

        initial_scores = (
            model.retrieval_decision_head(
                test_hidden
            )
        )

    initial_scores = (
        initial_scores
        .detach()
        .float()
        .cpu()
    )

    print(
        f"Random hidden decision mean: "
        f"{initial_scores.mean().item():.8f}"
    )

    print(
        f"Random hidden decision std : "
        f"{initial_scores.std().item():.8f}"
    )

    print(
        f"Random hidden decision min : "
        f"{initial_scores.min().item():.8f}"
    )

    print(
        f"Random hidden decision max : "
        f"{initial_scores.max().item():.8f}"
    )

    # --------------------------------------------------------
    # Zero hidden state
    # --------------------------------------------------------

    zero_hidden = torch.zeros(
        1,
        1,
        hidden_size,
        device=first_weight.device,
        dtype=first_weight.dtype,
    )

    with torch.no_grad():

        zero_score = (
            model.retrieval_decision_head(
                zero_hidden
            )
        )

    zero_score = (
        zero_score
        .detach()
        .float()
        .cpu()
        .item()
    )

    print(
        f"Zero-hidden decision score: "
        f"{zero_score:.8f}"
    )

    print(
        f"Retrieval threshold: "
        f"{THRESHOLD:.8f}"
    )

    print("=" * 80)

    # ========================================================
    # 11. ART TRAINING MODE
    # ========================================================

    print()
    print("8. Enabling ART training mode...")

    model.enable_training_mode()

    print(
        f"model.isTraining = "
        f"{model.isTraining}"
    )

    # ========================================================
    # 12. ENABLE GRADIENT CHECKPOINTING
    # ========================================================

    print()
    print("=" * 80)
    print("CONFIGURING GRADIENT CHECKPOINTING / CACHE")
    print("=" * 80)

    if GRADIENT_CHECKPOINTING:

        if not hasattr(
            model,
            "gradient_checkpointing_enable",
        ):
            raise RuntimeError(
                "Model does not support "
                "gradient checkpointing."
            )

        model.gradient_checkpointing_enable()

        print(
            "Gradient checkpointing: ENABLED"
        )

    else:

        if hasattr(
            model,
            "gradient_checkpointing_disable",
        ):
            model.gradient_checkpointing_disable()

        print(
            "Gradient checkpointing: DISABLED"
        )

    model.config.use_cache = True

    print(
        f"model.config.use_cache: "
        f"{model.config.use_cache}"
    )

    print("=" * 80)

    # ========================================================
    # 13. FREEZE / UNFREEZE
    # ========================================================

    print()
    print("9. Configuring trainable layers...")

    freeze_backbone(model)

    print_parameter_summary(model)

    # ========================================================
    # 14. DATA COLLATOR
    # ========================================================

    print()
    print("10. Creating data collator...")

    data_collator = ARTDataCollator()

    print("Data collator created.")

    # ========================================================
    # 15. TRAINING ARGUMENTS
    # ========================================================

    print()
    print("11. Creating TrainingArguments...")

    if BF16:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "BF16 training is enabled but CUDA "
                "is not available."
            )

        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 training is enabled but "
                "this GPU does not support BF16."
            )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        max_grad_norm=MAX_GRAD_NORM,

        bf16=BF16,

        fp16=False,

        gradient_checkpointing=(
            GRADIENT_CHECKPOINTING
        ),

        remove_unused_columns=False,

        logging_strategy="steps",

        logging_steps=LOGGING_STEPS,

        logging_first_step=True,

        save_strategy="steps",

        save_steps=SAVE_STEPS,

        eval_strategy="no",

        report_to="tensorboard",

        run_name="ART-REVIVE",

        dataloader_num_workers=0,

        dataloader_pin_memory=False,

        seed=SEED,

        data_seed=SEED,
    )

    print(
        f"Epochs: {NUM_TRAIN_EPOCHS}"
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
        "Effective batch size: "
        f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Unfreeze percentage: "
        f"{UNFREEZE_LAYER_PERCENTAGE * 100:.1f}%"
    )

    print(
        f"Unfreeze region: "
        f"{UNFREEZE_LAYER_REGION}"
    )

    print(
        f"Gradient checkpointing: "
        f"{GRADIENT_CHECKPOINTING}"
    )

    print(
        f"Decision threshold: {THRESHOLD}"
    )

    # ========================================================
    # 16. TRAINER
    # ========================================================

    print()
    print("12. Creating ART Trainer...")

    trainer = ARTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.model_accepts_loss_kwargs = False

    print(
        "Trainer created successfully."
    )

    print(
        "num_items_in_batch: DISABLED"
    )

    # ========================================================
    # 17. MEMORY BEFORE TRAINING
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

        print_gpu_memory(
            "GPU MEMORY BEFORE TRAINING"
        )

    # ========================================================
    # 18. TRAIN
    # ========================================================

    print()
    print("=" * 80)
    print("13. START TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    # ========================================================
    # 19. TRAINING RESULT
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

    print("=" * 80)

    # ========================================================
    # 20. MEMORY AFTER TRAINING
    # ========================================================

    if torch.cuda.is_available():

        print_gpu_memory(
            "GPU MEMORY AFTER TRAINING"
        )

    # ========================================================
    # 21. FINAL DECISION HEAD
    # ========================================================

    print_decision_head_summary(
        model=model,
        title="FINAL RETRIEVAL DECISION HEAD",
    )

    # ========================================================
    # 22. FINAL DECISION OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL RETRIEVAL DECISION OUTPUT")
    print("=" * 80)

    hidden_size = model.config.hidden_size

    linear_layers = get_decision_head_linear_layers(
        model
    )

    first_weight = linear_layers[0].weight

    test_hidden = torch.randn(
        8,
        1,
        hidden_size,
        device=first_weight.device,
        dtype=first_weight.dtype,
    )

    with torch.no_grad():

        final_scores = (
            model.retrieval_decision_head(
                test_hidden
            )
        )

    final_scores = (
        final_scores
        .detach()
        .float()
        .cpu()
    )

    print(
        f"Random hidden decision mean: "
        f"{final_scores.mean().item():.8f}"
    )

    print(
        f"Random hidden decision std : "
        f"{final_scores.std().item():.8f}"
    )

    print(
        f"Random hidden decision min : "
        f"{final_scores.min().item():.8f}"
    )

    print(
        f"Random hidden decision max : "
        f"{final_scores.max().item():.8f}"
    )

    zero_hidden = torch.zeros(
        1,
        1,
        hidden_size,
        device=first_weight.device,
        dtype=first_weight.dtype,
    )

    with torch.no_grad():

        final_zero_score = (
            model.retrieval_decision_head(
                zero_hidden
            )
        )

    final_zero_score = (
        final_zero_score
        .detach()
        .float()
        .cpu()
        .item()
    )

    print(
        f"Zero-hidden decision score: "
        f"{final_zero_score:.8f}"
    )

    print(
        f"Threshold: {THRESHOLD:.8f}"
    )

    print("=" * 80)

    # ========================================================
    # 23. ART STATE
    # ========================================================

    print()
    print("=" * 80)
    print("ART TRAINING STATE")
    print("=" * 80)

    print(
        f"isTraining      : "
        f"{model.isTraining}"
    )

    print(
        f"retrieval_count : "
        f"{model.retrieval_count}"
    )

    if hasattr(
        model,
        "retrieval_count_for_logging",
    ):

        retrieval_counts = getattr(
            model,
            "retrieval_count_for_logging",
            [],
        )

        print(
            f"retrieval_count_for_logging: "
            f"{len(retrieval_counts)} entries"
        )

        if retrieval_counts:

            print(
                f"Average retrieval count: "
                f"{sum(retrieval_counts) / len(retrieval_counts):.6f}"
            )

    if hasattr(
        model,
        "retrieval_scores",
    ):

        retrieval_scores = getattr(
            model,
            "retrieval_scores",
            [],
        )

        print(
            f"retrieval_scores: "
            f"{len(retrieval_scores)}"
        )

        for index, score in enumerate(
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
                    f"  [{index}] "
                    f"{score_value:.6f}"
                )

    print("=" * 80)

    # ========================================================
    # 24. SAVE FINAL MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)

    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved to: {OUTPUT_DIR}")

    print("=" * 80)

    # ========================================================
    # 25. CLEANUP
    # ========================================================

    print()
    print("Cleaning up...")

    del trainer
    del train_dataset
    del raw_dataset
    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
