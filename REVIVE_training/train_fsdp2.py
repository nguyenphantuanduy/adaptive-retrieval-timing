
from __future__ import annotations

import os
import gc
import math
import random

from typing import Dict, Any, List

import torch

from datasets import (
    load_dataset,
    concatenate_datasets,
)

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

TOKENIZER_REPO = (
    "Qwen/Qwen2.5-3B-Instruct"
)

OUTPUT_DIR = "./art-training-output"

SEED = 42


# ------------------------------------------------------------
# Distributed training
# ------------------------------------------------------------

NUM_GPUS = 2

FSDP_VERSION = 2

FSDP_RESHARD_AFTER_FORWARD = True

FSDP_CPU_OFFLOAD = False

FSDP_STATE_DICT_TYPE = "FULL_STATE_DICT"


# ------------------------------------------------------------
# Training epochs
# ------------------------------------------------------------

NUM_TRAIN_EPOCHS = 0.75


# ------------------------------------------------------------
# Percentage of decoder layers to unfreeze
# ------------------------------------------------------------

UNFREEZE_LAST_LAYER_PERCENTAGE = 0.40


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

SAVE_TOTAL_LIMIT = 2


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
# DISTRIBUTED HELPERS
# ============================================================

def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def rank_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def initialize_distributed_environment():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FSDP2 requires CUDA."
        )

    world_size = get_world_size()

    if world_size != NUM_GPUS:
        raise RuntimeError(
            f"Expected {NUM_GPUS} processes, "
            f"received {world_size}."
        )

    local_rank = get_local_rank()

    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    if BF16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is not supported by this GPU."
        )

    rank_print("=" * 80)
    rank_print("DISTRIBUTED ENVIRONMENT")
    rank_print("=" * 80)

    rank_print(f"World size: {world_size}")
    rank_print(f"Local rank: {local_rank}")
    rank_print(f"Device: {device}")

    rank_print(
        f"GPU: {torch.cuda.get_device_name(local_rank)}"
    )

    return device


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD DATASET
# ============================================================

def load_teacher_generated_training() -> Dict[str, Any]:

    datasets = {}

    for hop in (2, 3, 4):

        config_name = f"{hop}-hop"

        rank_print()
        rank_print("=" * 80)
        rank_print(f"LOADING {config_name.upper()} DATASET")
        rank_print("=" * 80)

        dataset = load_dataset(
            DATASET_NAME,
            config_name,
        )

        datasets[config_name] = dataset["train"]

        rank_print(
            f"{config_name}: "
            f"{len(datasets[config_name]):,} samples"
        )

    return datasets


# ============================================================
# COMBINE + SHUFFLE
# ============================================================

def build_combined_dataset():

    datasets = load_teacher_generated_training()

    rank_print()
    rank_print("=" * 80)
    rank_print("COMBINING TRAINING DATASETS")
    rank_print("=" * 80)

    combined_dataset = concatenate_datasets(
        [
            datasets["2-hop"],
            datasets["3-hop"],
            datasets["4-hop"],
        ]
    )

    rank_print(
        f"Combined dataset size: "
        f"{len(combined_dataset):,}"
    )

    combined_dataset = combined_dataset.shuffle(
        seed=SEED
    )

    rank_print("Dataset shuffled.")

    return combined_dataset


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_training_sample(
    sample: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:

    question = sample["question"]

    teacher_generation_output = (
        sample["teacher_generation_output"]
    )

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

    question_tokenized = tokenizer(
        question_text,
        add_special_tokens=False,
        return_tensors=None,
    )

    question_input_ids = (
        question_tokenized["input_ids"]
    )

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

    attention_mask = (
        [1] * len(question_input_ids)
        + [1] * len(answer_input_ids)
    )

    labels = (
        [-100] * len(question_input_ids)
        + answer_input_ids
    )

    expected_length = (
        len(question_input_ids)
        + len(answer_input_ids)
    )

    if (
        len(attention_mask) != expected_length
        or len(labels) != expected_length
    ):
        raise ValueError(
            "Tokenization length mismatch."
        )

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

def preprocess_dataset(dataset, tokenizer):

    rank_print()
    rank_print("=" * 80)
    rank_print("TOKENIZING TRAINING DATASET")
    rank_print("=" * 80)

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
                "Attention mask length mismatch."
            )

        if labels.shape[0] != expected_length:
            raise ValueError(
                "Labels length mismatch."
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

    rank_print()
    rank_print("=" * 80)
    rank_print("FREEZING / UNFREEZING BACKBONE")
    rank_print("=" * 80)

    for parameter in model.parameters():
        parameter.requires_grad = False

    layers = model.model.layers

    num_layers = len(layers)

    num_layers_to_unfreeze = math.ceil(
        num_layers
        * UNFREEZE_LAST_LAYER_PERCENTAGE
    )

    first_trainable_layer = (
        num_layers
        - num_layers_to_unfreeze
    )

    for layer_index in range(
        first_trainable_layer,
        num_layers,
    ):

        for parameter in (
            layers[layer_index].parameters()
        ):
            parameter.requires_grad = True

    for parameter in (
        model.retrieval_decision_head.parameters()
    ):
        parameter.requires_grad = True

    rank_print(
        f"Total decoder layers: {num_layers}"
    )

    rank_print(
        f"Trainable decoder layers: "
        f"{num_layers_to_unfreeze}"
    )

    rank_print(
        f"First trainable layer: "
        f"{first_trainable_layer}"
    )

    for index, layer in enumerate(layers):

        trainable = any(
            parameter.requires_grad
            for parameter in layer.parameters()
        )

        rank_print(
            f"Layer {index:02d}: "
            f"{'TRAINABLE' if trainable else 'FROZEN'}"
        )

    rank_print(
        "Retrieval decision head: TRAINABLE"
    )

    rank_print(
        "Embedding / LM head / Final norm: FROZEN"
    )


# ============================================================
# PARAMETER SUMMARY
# ============================================================

def print_parameter_summary(model):

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

    percentage = (
        100.0
        * trainable_parameters
        / total_parameters
    )

    rank_print()
    rank_print("=" * 80)
    rank_print("PARAMETER SUMMARY")
    rank_print("=" * 80)

    rank_print(
        f"Total: {total_parameters:,}"
    )

    rank_print(
        f"Trainable: {trainable_parameters:,}"
    )

    rank_print(
        f"Frozen: {frozen_parameters:,}"
    )

    rank_print(
        f"Trainable percentage: {percentage:.4f}%"
    )


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
            "No Linear layers found in decision head."
        )

    return linear_layers


def print_decision_head_summary(
    model,
    title: str,
):

    linear_layers = get_decision_head_linear_layers(
        model
    )

    rank_print()
    rank_print("=" * 80)
    rank_print(title)
    rank_print("=" * 80)

    for index, layer in enumerate(linear_layers):

        rank_print(
            f"Linear {index + 1}: "
            f"{layer.in_features} -> "
            f"{layer.out_features}"
        )

        rank_print(
            f"Weight shape: {tuple(layer.weight.shape)}"
        )

        rank_print(
            f"Parameters: {layer.weight.numel():,}"
        )

        if layer.bias is not None:
            rank_print(
                f"Bias parameters: {layer.bias.numel():,}"
            )


# ============================================================
# FSDP2 CONFIGURATION
# ============================================================

def build_fsdp_config(model):

    layers = model.model.layers

    if len(layers) == 0:
        raise RuntimeError(
            "No decoder layers found."
        )

    decoder_layer_class = type(
        layers[0]
    ).__name__

    config = {
        "version": 2,

        "reshard_after_forward": (
            FSDP_RESHARD_AFTER_FORWARD
        ),

        "cpu_offload": FSDP_CPU_OFFLOAD,

        "auto_wrap_policy": (
            "TRANSFORMER_BASED_WRAP"
        ),

        "transformer_layer_cls_to_wrap": [
            decoder_layer_class
        ],

        "state_dict_type": (
            FSDP_STATE_DICT_TYPE
        ),

        "cpu_ram_efficient_loading": False,

        "activation_checkpointing": False,
    }

    rank_print()
    rank_print("=" * 80)
    rank_print("FSDP2 CONFIGURATION")
    rank_print("=" * 80)

    for key, value in config.items():
        rank_print(f"{key}: {value}")

    rank_print("=" * 80)

    return config


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory(title: str):

    if not torch.cuda.is_available():
        return

    gpu_index = torch.cuda.current_device()

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

    rank_print()
    rank_print("=" * 80)

    rank_print(
        f"{title} | GPU {gpu_index}"
    )

    rank_print(
        f"Allocated: {allocated:.2f} GB"
    )

    rank_print(
        f"Reserved: {reserved:.2f} GB"
    )

    rank_print(
        f"Peak: {peak:.2f} GB"
    )

    rank_print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    rank_print()
    rank_print("=" * 80)
    rank_print("ART FSDP2 TRAINING")
    rank_print("=" * 80)

    # ========================================================
    # 1. DISTRIBUTED ENVIRONMENT
    # ========================================================

    device = initialize_distributed_environment()

    # ========================================================
    # 2. SEED
    # ========================================================

    rank_print("Setting seed...")

    set_seed(SEED)

    # ========================================================
    # 3. TOKENIZER
    # ========================================================

    rank_print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO,
        trust_remote_code=True,
    )

    if tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer does not have an EOS token."
        )

    rank_print(
        f"Tokenizer: {TOKENIZER_REPO}"
    )

    rank_print(
        f"EOS: {repr(tokenizer.eos_token)}"
    )

    # ========================================================
    # 4. DATASET
    # ========================================================

    rank_print("Loading dataset...")

    raw_dataset = build_combined_dataset()

    rank_print(
        f"Dataset size: {len(raw_dataset):,}"
    )

    # ========================================================
    # 5. TOKENIZE
    # ========================================================

    train_dataset = preprocess_dataset(
        raw_dataset,
        tokenizer,
    )

    rank_print(
        f"Tokenized samples: {len(train_dataset):,}"
    )

    # ========================================================
    # 6. INSPECT SAMPLE
    # ========================================================

    first_sample = train_dataset[0]

    question_length = len(
        first_sample["question_input_ids"]
    )

    answer_length = len(
        first_sample["answer_input_ids"]
    )

    rank_print()
    rank_print("=" * 80)
    rank_print("FIRST SAMPLE")
    rank_print("=" * 80)

    rank_print(
        f"Question tokens: {question_length}"
    )

    rank_print(
        f"Answer tokens: {answer_length}"
    )

    rank_print(
        f"Full sequence: "
        f"{question_length + answer_length}"
    )

    # ========================================================
    # 7. LOAD MODEL
    # ========================================================

    rank_print("Loading ART model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,

        torch_dtype=torch.bfloat16,

        trust_remote_code=True,

        attn_implementation="eager",
    )

    rank_print(
        f"Loaded model: {MODEL_REPO}"
    )

    # ========================================================
    # 8. ART TRAINING MODE
    # ========================================================

    model.enable_training_mode()

    rank_print(
        f"model.isTraining: {model.isTraining}"
    )

    # ========================================================
    # 9. FREEZE / UNFREEZE
    # ========================================================

    freeze_backbone(model)

    print_parameter_summary(model)

    # ========================================================
    # 10. GRADIENT CHECKPOINTING
    # ========================================================

    if GRADIENT_CHECKPOINTING:

        if not hasattr(
            model,
            "gradient_checkpointing_enable",
        ):
            raise RuntimeError(
                "Gradient checkpointing is unsupported."
            )

        model.gradient_checkpointing_enable()

        rank_print(
            "Gradient checkpointing: ENABLED"
        )

    else:

        if hasattr(
            model,
            "gradient_checkpointing_disable",
        ):
            model.gradient_checkpointing_disable()

        rank_print(
            "Gradient checkpointing: DISABLED"
        )

    model.config.use_cache = True

    # ========================================================
    # 11. DECISION HEAD
    # ========================================================

    print_decision_head_summary(
        model,
        "INITIAL DECISION HEAD",
    )

    # ========================================================
    # 12. INITIAL DECISION TEST
    # ========================================================

    linear_layers = get_decision_head_linear_layers(
        model
    )

    first_weight = linear_layers[0].weight

    hidden_size = model.config.hidden_size

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
        initial_scores.detach().float().cpu()
    )

    rank_print(
        f"Initial decision mean: "
        f"{initial_scores.mean().item():.8f}"
    )

    rank_print(
        f"Retrieval threshold: {THRESHOLD}"
    )

    # ========================================================
    # 13. FSDP2 CONFIG
    # ========================================================

    fsdp_config = build_fsdp_config(model)

    # ========================================================
    # 14. DATA COLLATOR
    # ========================================================

    data_collator = ARTDataCollator()

    # ========================================================
    # 15. TRAINING ARGUMENTS
    # ========================================================

    effective_batch_size = (
        PER_DEVICE_TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
        * NUM_GPUS
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

        # FSDP2
        fsdp="full_shard",

        fsdp_config=fsdp_config,

        # ART custom inputs
        remove_unused_columns=False,

        # Logging
        logging_strategy="steps",

        logging_steps=LOGGING_STEPS,

        logging_first_step=True,

        # Saving
        save_strategy="steps",

        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        # Evaluation
        eval_strategy="no",

        # Reporting
        report_to="tensorboard",

        run_name="ART-REVIVE-FSDP2",

        # Dataloader
        dataloader_num_workers=0,

        dataloader_pin_memory=False,

        # Reproducibility
        seed=SEED,

        data_seed=SEED,

        # Distributed
        ddp_find_unused_parameters=False,
    )

    rank_print()
    rank_print("=" * 80)
    rank_print("TRAINING CONFIGURATION")
    rank_print("=" * 80)

    rank_print(
        f"GPUs: {NUM_GPUS}"
    )

    rank_print(
        f"Per-device batch: "
        f"{PER_DEVICE_TRAIN_BATCH_SIZE}"
    )

    rank_print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    rank_print(
        f"Global effective batch: "
        f"{effective_batch_size}"
    )

    rank_print(
        f"Learning rate: {LEARNING_RATE}"
    )

    rank_print(
        f"Epochs: {NUM_TRAIN_EPOCHS}"
    )

    rank_print(
        f"Gradient checkpointing: "
        f"{GRADIENT_CHECKPOINTING}"
    )

    # ========================================================
    # 16. ART TRAINER
    # ========================================================

    rank_print("Creating ARTTrainer...")

    trainer = ARTTrainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        data_collator=data_collator,
    )

    trainer.model_accepts_loss_kwargs = False

    rank_print(
        "ARTTrainer created."
    )

    # ========================================================
    # 17. MEMORY BEFORE TRAINING
    # ========================================================

    torch.cuda.reset_peak_memory_stats()

    print_gpu_memory(
        "GPU MEMORY BEFORE TRAINING"
    )

    # ========================================================
    # 18. TRAIN
    # ========================================================

    rank_print()
    rank_print("=" * 80)
    rank_print("START ART FSDP2 TRAINING")
    rank_print("=" * 80)

    train_result = trainer.train()

    # ========================================================
    # 19. TRAINING RESULT
    # ========================================================

    rank_print()
    rank_print("=" * 80)
    rank_print("TRAINING RESULT")
    rank_print("=" * 80)

    rank_print(
        f"Global step: "
        f"{trainer.state.global_step}"
    )

    rank_print(
        f"Training loss: "
        f"{train_result.training_loss}"
    )

    # ========================================================
    # 20. MEMORY AFTER TRAINING
    # ========================================================

    print_gpu_memory(
        "GPU MEMORY AFTER TRAINING"
    )

    # ========================================================
    # 21. SAVE FINAL MODEL
    # ========================================================

    rank_print()
    rank_print("=" * 80)
    rank_print("SAVING FINAL MODEL")
    rank_print("=" * 80)

    trainer.accelerator.wait_for_everyone()

    trainer.save_model(
        OUTPUT_DIR
    )

    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():

        tokenizer.save_pretrained(
            OUTPUT_DIR
        )

    trainer.accelerator.wait_for_everyone()

    rank_print(
        f"Saved model to: {OUTPUT_DIR}"
    )

    # ========================================================
    # 22. CLEANUP
    # ========================================================

    del trainer
    del train_dataset
    del raw_dataset
    del model

    gc.collect()

    torch.cuda.synchronize()

    torch.cuda.empty_cache()

    rank_print()
    rank_print("=" * 80)
    rank_print("ART FSDP2 TRAINING COMPLETE")
    rank_print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
