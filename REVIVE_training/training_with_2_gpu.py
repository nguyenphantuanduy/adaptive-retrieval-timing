
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
    "MuSiQue-ART-Training"
)

MODEL_REPO = (
    "adaptive-retrieval-timing/"
    "ART-REVIVE"
)

TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"

OUTPUT_DIR = "./art-training-output"

SEED = 42


# ============================================================
# TRAINING
# ============================================================

NUM_TRAIN_EPOCHS = 0.3


# ============================================================
# FREEZE / UNFREEZE
# ============================================================

UNFREEZE_LAST_LAYER_PERCENTAGE = 0.40


# ============================================================
# BATCH
# ============================================================

PER_DEVICE_TRAIN_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 32


# ============================================================
# OPTIMIZATION
# ============================================================

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.0

MAX_GRAD_NORM = 1.0


# ============================================================
# LOGGING / SAVING
# ============================================================

LOGGING_STEPS = 1

SAVE_STEPS = 100


# ============================================================
# PRECISION
# ============================================================

BF16 = True


# ============================================================
# GRADIENT CHECKPOINTING
# ============================================================

GRADIENT_CHECKPOINTING = True


# ============================================================
# MULTI-GPU MODEL PLACEMENT
# ============================================================

DEVICE_MAP = "auto"


# ============================================================
# RETRIEVAL THRESHOLD
# ============================================================

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
    Load the 2-hop, 3-hop and 4-hop training splits.
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

    combined_dataset = combined_dataset.shuffle(
        seed=SEED
    )

    print(f"Dataset shuffled with seed {SEED}.")

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

    Labels:
        Question tokens -> -100
        Answer tokens   -> token IDs

    Attention mask:
        Question mask + answer mask
    """

    question = sample["question"]

    teacher_generation_output = (
        sample["teacher_generation_output"]
    )

    # --------------------------------------------------------
    # 1. Build chat-templated question
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

    question_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # 2. Tokenize question
    # --------------------------------------------------------

    question_tokenized = tokenizer(
        question_text,
        add_special_tokens=False,
        return_tensors=None,
    )

    question_input_ids = (
        question_tokenized["input_ids"]
    )

    # --------------------------------------------------------
    # 3. Tokenize teacher answer + EOS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 4. Attention mask
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. Labels
    # --------------------------------------------------------

    labels = (
        [-100] * len(question_input_ids)
        + answer_input_ids
    )

    # --------------------------------------------------------
    # 6. Sanity checks
    # --------------------------------------------------------

    expected_length = (
        len(question_input_ids)
        + len(answer_input_ids)
    )

    if len(attention_mask) != expected_length:
        raise ValueError(
            "Attention mask length mismatch: "
            f"{len(attention_mask)} != {expected_length}"
        )

    if len(labels) != expected_length:
        raise ValueError(
            "Labels length mismatch: "
            f"{len(labels)} != {expected_length}"
        )

    # --------------------------------------------------------
    # 7. Return
    # --------------------------------------------------------

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
    Collator for the existing single-sample ART training logic.
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
                f"{attention_mask.shape[0]} != {expected_length}"
            )

        if labels.shape[0] != expected_length:
            raise ValueError(
                "Labels length mismatch: "
                f"{labels.shape[0]} != {expected_length}"
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
            "labels": labels.unsqueeze(0),
        }


# ============================================================
# FREEZE / UNFREEZE BACKBONE
# ============================================================

def freeze_backbone(model) -> None:
    """
    Freeze the entire model first, then unfreeze:

        - The last 40% of decoder layers
        - The retrieval decision head

    All other parameters remain frozen, except parameters
    shared with an explicitly unfrozen module.
    """

    print()
    print("=" * 80)
    print("FREEZING / UNFREEZING BACKBONE")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Freeze all parameters
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # 2. Get decoder layers
    # --------------------------------------------------------

    layers = model.model.layers

    num_layers = len(layers)

    if num_layers == 0:
        raise RuntimeError("No decoder layers found.")

    # --------------------------------------------------------
    # 3. Calculate the number of layers to unfreeze
    # --------------------------------------------------------

    num_layers_to_unfreeze = math.ceil(
        num_layers * UNFREEZE_LAST_LAYER_PERCENTAGE
    )

    first_trainable_layer = (
        num_layers - num_layers_to_unfreeze
    )

    # --------------------------------------------------------
    # 4. Unfreeze last N decoder layers
    # --------------------------------------------------------

    for layer_index in range(
        first_trainable_layer,
        num_layers,
    ):
        for parameter in layers[layer_index].parameters():
            parameter.requires_grad = True

    # --------------------------------------------------------
    # 5. Unfreeze retrieval decision head
    # --------------------------------------------------------

    for parameter in model.retrieval_decision_head.parameters():
        parameter.requires_grad = True

    # --------------------------------------------------------
    # 6. Report layer configuration
    # --------------------------------------------------------

    actual_percentage = (
        100.0 * num_layers_to_unfreeze / num_layers
    )

    print(f"Total decoder layers       : {num_layers}")
    print(
        f"Requested unfreeze         : "
        f"{UNFREEZE_LAST_LAYER_PERCENTAGE * 100:.1f}%"
    )
    print(
        f"Number of trainable layers : "
        f"{num_layers_to_unfreeze}"
    )
    print(
        f"Actual unfreeze percentage : "
        f"{actual_percentage:.2f}%"
    )
    print(f"First trainable layer      : {first_trainable_layer}")
    print(f"Last trainable layer       : {num_layers - 1}")

    print()

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

    print("Retrieval decision head: TRAINABLE")
    print("Embedding: FROZEN unless parameters are shared")
    print("LM head: FROZEN unless parameters are shared")
    print("Final norm: FROZEN")

    print("=" * 80)

    print_parameter_summary(model)


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
        100.0 * trainable_parameters / total_parameters
        if total_parameters
        else 0.0
    )

    print()
    print("=" * 80)
    print("PARAMETER SUMMARY")
    print("=" * 80)

    print(f"Total parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")
    print(f"Frozen parameters    : {frozen_parameters:,}")
    print(f"Trainable percentage : {trainable_percentage:.4f}%")

    print("=" * 80)


# ============================================================
# DECISION HEAD HELPERS
# ============================================================

def get_decision_head_linear_layers(model):

    linear_layers = [
        module
        for module in model.retrieval_decision_head.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    if not linear_layers:
        raise RuntimeError(
            "No Linear layers found inside retrieval_decision_head."
        )

    return linear_layers


def print_decision_head_summary(
    model,
    title: str,
) -> None:

    linear_layers = get_decision_head_linear_layers(model)

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"Number of Linear layers: {len(linear_layers)}")

    total_head_parameters = 0

    for index, layer in enumerate(linear_layers):

        layer_parameters = layer.weight.numel()

        if layer.bias is not None:
            layer_parameters += layer.bias.numel()

        total_head_parameters += layer_parameters

        print()
        print(
            f"Linear {index + 1}: "
            f"{layer.in_features} -> {layer.out_features}"
        )

        print(f"  Weight shape: {tuple(layer.weight.shape)}")

        print(
            f"  Weight abs mean: "
            f"{layer.weight.detach().float().abs().mean().item():.6e}"
        )

        print(
            f"  Weight std: "
            f"{layer.weight.detach().float().std().item():.6e}"
        )

        if layer.bias is not None:
            print(
                f"  Bias mean: "
                f"{layer.bias.detach().float().mean().item():+.6e}"
            )

            print(
                f"  Bias std: "
                f"{layer.bias.detach().float().std().item():.6e}"
            )

        print(f"  Parameters: {layer_parameters:,}")

    print()
    print(
        f"Total decision head parameters: "
        f"{total_head_parameters:,}"
    )

    print("=" * 80)


@torch.no_grad()
def inspect_decision_head(
    model,
    title: str,
) -> None:

    print_decision_head_summary(model, title)

    linear_layers = get_decision_head_linear_layers(model)

    first_weight = linear_layers[0].weight

    hidden_size = model.config.hidden_size

    test_hidden = torch.randn(
        8,
        1,
        hidden_size,
        device=first_weight.device,
        dtype=first_weight.dtype,
    )

    scores = (
        model.retrieval_decision_head(test_hidden)
        .detach()
        .float()
        .cpu()
    )

    zero_hidden = torch.zeros(
        1,
        1,
        hidden_size,
        device=first_weight.device,
        dtype=first_weight.dtype,
    )

    zero_score = (
        model.retrieval_decision_head(zero_hidden)
        .detach()
        .float()
        .cpu()
        .item()
    )

    print()
    print("=" * 80)
    print(f"{title} - OUTPUT")
    print("=" * 80)

    print(f"Random hidden decision mean: {scores.mean().item():.8f}")
    print(f"Random hidden decision std : {scores.std().item():.8f}")
    print(f"Random hidden decision min : {scores.min().item():.8f}")
    print(f"Random hidden decision max : {scores.max().item():.8f}")
    print(f"Zero-hidden decision score : {zero_score:.8f}")
    print(f"Retrieval threshold        : {THRESHOLD:.8f}")

    print("=" * 80)


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory(title: str) -> None:

    if not torch.cuda.is_available():
        return

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for gpu_index in range(torch.cuda.device_count()):

        allocated = (
            torch.cuda.memory_allocated(gpu_index) / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(gpu_index) / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated(gpu_index) / 1024**3
        )

        print(
            f"GPU {gpu_index}: "
            f"allocated={allocated:.2f} GB, "
            f"reserved={reserved:.2f} GB, "
            f"peak={peak:.2f} GB"
        )

    print("=" * 80)


# ============================================================
# ART STATE
# ============================================================

def print_art_state(model) -> None:

    print()
    print("=" * 80)
    print("ART TRAINING STATE")
    print("=" * 80)

    print(f"isTraining: {getattr(model, 'isTraining', None)}")
    print(f"retrieval_count: {getattr(model, 'retrieval_count', None)}")

    if hasattr(model, "retrieval_count_for_logging"):

        retrieval_counts = model.retrieval_count_for_logging

        print(
            f"retrieval_count_for_logging: "
            f"{len(retrieval_counts)} entries"
        )

        if retrieval_counts:
            print(
                f"Average retrieval count: "
                f"{sum(retrieval_counts) / len(retrieval_counts):.6f}"
            )

    if hasattr(model, "retrieval_scores"):

        retrieval_scores = model.retrieval_scores

        print(f"retrieval_scores: {len(retrieval_scores)}")

        for index, score in enumerate(retrieval_scores):

            if isinstance(score, torch.Tensor):
                score_value = (
                    score.detach().float().cpu().item()
                )
                print(f"  [{index}] {score_value:.6f}")
            else:
                print(f"  [{index}] {score}")

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("ART TRAINING - PARTIAL UNFREEZING - 2 GPUS")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Seed
    # --------------------------------------------------------

    print("\n1. Setting seed...")

    set_seed(SEED)

    # --------------------------------------------------------
    # 2. Device check
    # --------------------------------------------------------

    print("\n2. Checking devices...")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    num_gpus = torch.cuda.device_count()

    print(f"Available GPUs: {num_gpus}")

    for gpu_index in range(num_gpus):
        print(
            f"GPU {gpu_index}: "
            f"{torch.cuda.get_device_name(gpu_index)}"
        )

    if num_gpus < 2:
        raise RuntimeError(
            "This script expects at least two visible GPUs."
        )

    if BF16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is enabled but the current CUDA device "
            "does not report BF16 support."
        )

    # --------------------------------------------------------
    # 3. Tokenizer
    # --------------------------------------------------------

    print("\n3. Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO,
        trust_remote_code=True,
    )

    if tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer does not have an EOS token."
        )

    print(f"Tokenizer: {TOKENIZER_REPO}")
    print(f"EOS token: {repr(tokenizer.eos_token)}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    # --------------------------------------------------------
    # 4. Dataset
    # --------------------------------------------------------

    print("\n4. Loading training dataset...")

    raw_dataset = build_combined_dataset()

    print(
        f"Final shuffled dataset size: "
        f"{len(raw_dataset):,}"
    )

    # --------------------------------------------------------
    # 5. Preprocess
    # --------------------------------------------------------

    print("\n5. Preprocessing dataset...")

    train_dataset = preprocess_dataset(
        raw_dataset,
        tokenizer,
    )

    print(
        f"Tokenized training samples: "
        f"{len(train_dataset):,}"
    )

    # --------------------------------------------------------
    # 6. Inspect first sample
    # --------------------------------------------------------

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

    expected_length = question_length + answer_length

    print(f"Question tokens: {question_length}")
    print(f"Answer tokens: {answer_length}")
    print(f"Attention mask: {attention_length}")
    print(f"Label tokens: {label_length}")
    print(f"Expected length: {expected_length}")

    if attention_length != expected_length:
        raise ValueError(
            "Attention mask does not cover question + answer."
        )

    if label_length != expected_length:
        raise ValueError(
            "Labels do not cover question + answer."
        )

    print("Attention mask and labels passed sanity checks.")
    print("=" * 80)

    # --------------------------------------------------------
    # 7. Load model with device_map="auto"
    # --------------------------------------------------------

    print("\n6. Loading model across GPUs...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16 if BF16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map=DEVICE_MAP,
    )

    # Do not call model.to(device) here.
    # The model has already been dispatched by device_map.

    print(f"Model loaded: {MODEL_REPO}")
    print(f"Model hidden size: {model.config.hidden_size}")

    print()
    print("=" * 80)
    print("MODEL DEVICE MAP")
    print("=" * 80)

    for module_name, device in model.hf_device_map.items():
        print(f"{module_name}: {device}")

    print("=" * 80)

    # --------------------------------------------------------
    # 8. Inspect loaded decision head
    # --------------------------------------------------------

    print("\n7. Inspecting loaded decision head...")

    inspect_decision_head(
        model,
        "LOADED RETRIEVAL DECISION HEAD",
    )

    # --------------------------------------------------------
    # 9. Enable ART training mode
    # --------------------------------------------------------

    print("\n8. Enabling ART training mode...")

    model.enable_training_mode()

    print(f"model.isTraining = {model.isTraining}")

    # --------------------------------------------------------
    # 10. Gradient checkpointing
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CONFIGURING GRADIENT CHECKPOINTING / CACHE")
    print("=" * 80)

    if GRADIENT_CHECKPOINTING:

        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError(
                "Model does not support gradient checkpointing."
            )

        model.gradient_checkpointing_enable()

        print("Gradient checkpointing: ENABLED")

    else:

        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        print("Gradient checkpointing: DISABLED")

    model.config.use_cache = True

    print(f"model.config.use_cache: {model.config.use_cache}")

    print("=" * 80)

    # --------------------------------------------------------
    # 11. Freeze / unfreeze
    # --------------------------------------------------------

    print("\n9. Configuring trainable layers...")

    freeze_backbone(model)

    # --------------------------------------------------------
    # 12. Data collator
    # --------------------------------------------------------

    print("\n10. Creating data collator...")

    data_collator = ARTDataCollator()

    print("Data collator created.")

    # --------------------------------------------------------
    # 13. Training arguments
    # --------------------------------------------------------

    print("\n11. Creating TrainingArguments...")

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

        gradient_checkpointing=GRADIENT_CHECKPOINTING,

        remove_unused_columns=False,

        logging_strategy="steps",

        logging_steps=LOGGING_STEPS,

        logging_first_step=True,

        save_strategy="steps",

        save_steps=SAVE_STEPS,

        eval_strategy="no",

        report_to="none",

        dataloader_num_workers=0,

        dataloader_pin_memory=False,

        seed=SEED,

        data_seed=SEED,
    )

    print(f"Epochs: {NUM_TRAIN_EPOCHS}")

    print(
        f"Per-device batch size: "
        f"{PER_DEVICE_TRAIN_BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        "Effective batch size for one process: "
        f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(f"Learning rate: {LEARNING_RATE}")

    print(
        f"Unfreeze last: "
        f"{UNFREEZE_LAST_LAYER_PERCENTAGE * 100:.1f}% decoder layers"
    )

    print(f"Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
    print(f"Decision threshold: {THRESHOLD}")
    print(f"Device map: {DEVICE_MAP}")

    # --------------------------------------------------------
    # 14. Trainer
    # --------------------------------------------------------

    print("\n12. Creating ARTTrainer...")

    trainer = ARTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.model_accepts_loss_kwargs = False

    print("ARTTrainer created successfully.")
    print("num_items_in_batch: DISABLED")

    # --------------------------------------------------------
    # 15. GPU memory before training
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()

    print_gpu_memory("GPU MEMORY BEFORE TRAINING")

    # --------------------------------------------------------
    # 16. Train
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("START ART TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    # --------------------------------------------------------
    # 17. Training result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING RESULT")
    print("=" * 80)

    print(f"Global step: {trainer.state.global_step}")
    print(f"Training loss: {train_result.training_loss}")

    print("=" * 80)

    # --------------------------------------------------------
    # 18. GPU memory after training
    # --------------------------------------------------------

    print_gpu_memory("GPU MEMORY AFTER TRAINING")

    # --------------------------------------------------------
    # 19. Final decision head
    # --------------------------------------------------------

    inspect_decision_head(
        model,
        "FINAL RETRIEVAL DECISION HEAD",
    )

    # --------------------------------------------------------
    # 20. ART state
    # --------------------------------------------------------

    print_art_state(model)

    # --------------------------------------------------------
    # 21. Save final model
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)

    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved to: {OUTPUT_DIR}")

    print("=" * 80)

    # --------------------------------------------------------
    # 22. Cleanup
    # --------------------------------------------------------

    print("\nCleaning up...")

    del trainer
    del train_dataset
    del raw_dataset
    del model

    gc.collect()

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