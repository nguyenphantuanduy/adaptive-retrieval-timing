# ============================================================
# MUSIQUE TEACHER GENERATION
# QWEN3-4B
# STRATIFIED SAMPLING + BATCH GENERATION + JSONL OUTPUT
# ============================================================

import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .load_dataset import load_musique_ans
from .musique_sample import MuSiQueSampleConverter
from .musique_teacher_prompt import create_teacher_prompt


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-4B"

TEACHER_GENERATION_SAMPLES = 1024

# ------------------------------------------------------------
# Batch size is hop-dependent because longer-hop samples
# generally produce longer teacher prompts and consume more
# VRAM during generation.
#
# 2-hop : batch 8
# 3-hop : batch 4
# 4-hop : batch 4
# ------------------------------------------------------------

BATCH_SIZE_BY_HOP = {
    2: 8,
    3: 4,
    4: 4,
}

MAX_NEW_TOKENS = 1024

SEED = 42

TEMPERATURE = 0.6

TOP_P = 0.95


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

CURRENT_FILE_DIR = Path(
    __file__
).resolve().parent

OUTPUT_DIR = (
    CURRENT_FILE_DIR
    / "data"
    / "teacher_generation_for_hypothesis_validation"
)


# ============================================================
# SET RANDOM SEED
# ============================================================

def set_seed(
    seed: int = SEED,
):
    """
    Set random seeds for reproducible sampling
    and generation.
    """

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD TEACHER MODEL
# ============================================================

def load_teacher_model():
    """
    Load Qwen3-4B teacher model and tokenizer.

    The model is loaded with automatic device mapping
    so it can use the available GPU(s).
    """

    print("=" * 80)
    print("LOADING TEACHER MODEL")
    print("=" * 80)

    print()
    print(
        f"Model: {MODEL_NAME}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )

    # --------------------------------------------------------
    # Qwen3 should have EOS token.
    # Use EOS as padding token if no PAD token exists.
    # --------------------------------------------------------

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ========================================================
    # MODEL
    # ========================================================

    model_kwargs = {
        "device_map": "auto",
    }

    # --------------------------------------------------------
    # Use BF16 when supported.
    # --------------------------------------------------------

    if torch.cuda.is_available():

        if torch.cuda.is_bf16_supported():

            model_kwargs[
                "torch_dtype"
            ] = torch.bfloat16

        else:

            model_kwargs[
                "torch_dtype"
            ] = torch.float16

    else:

        model_kwargs[
            "torch_dtype"
        ] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    model.eval()

    print()
    print(
        "✓ Teacher model loaded"
    )

    print(
        f"  Model: {MODEL_NAME}"
    )

    print(
        f"  Device map: auto"
    )

    print(
        f"  Dtype: {model.dtype}"
    )

    if torch.cuda.is_available():

        print()
        print(
            f"CUDA devices: "
            f"{torch.cuda.device_count()}"
        )

        for i in range(
            torch.cuda.device_count()
        ):

            print(
                f"  GPU {i}: "
                f"{torch.cuda.get_device_name(i)}"
            )

    return model, tokenizer


# ============================================================
# PREPARE TEACHER GENERATION SAMPLES
# ============================================================

def prepare_teacher_generation_samples(
    dataset,
    num_samples: int = TEACHER_GENERATION_SAMPLES,
    seed: int = SEED,
):
    """
    Prepare MuSiQue samples for teacher generation.

    Samples are stratified according to the natural
    2-hop / 3-hop / 4-hop distribution.

    Returns:
        dict:
            {
                2: [...],
                3: [...],
                4: [...]
            }
    """

    if num_samples <= 0:

        raise ValueError(
            "num_samples must be greater than 0."
        )

    # ========================================================
    # 1. GROUP SAMPLES BY HOP
    # ========================================================

    samples_by_hop = {
        2: [],
        3: [],
        4: [],
    }

    for sample in dataset:

        decomposition = sample.get(
            "question_decomposition"
        )

        if not isinstance(
            decomposition,
            list,
        ):

            continue

        hop = len(
            decomposition
        )

        if hop in samples_by_hop:

            samples_by_hop[
                hop
            ].append(
                sample
            )

    # ========================================================
    # 2. VALIDATE DATASET
    # ========================================================

    for hop in (2, 3, 4):

        if len(
            samples_by_hop[hop]
        ) == 0:

            raise ValueError(
                f"No {hop}-hop samples found."
            )

    total_valid_samples = sum(
        len(
            samples_by_hop[hop]
        )
        for hop in (2, 3, 4)
    )

    if num_samples > total_valid_samples:

        raise ValueError(
            f"Requested {num_samples:,} samples, "
            f"but only {total_valid_samples:,} "
            f"valid 2/3/4-hop samples are available."
        )

    # ========================================================
    # 3. NATURAL HOP DISTRIBUTION
    # ========================================================

    hop_ratios = {
        hop:
            len(
                samples_by_hop[hop]
            )
            / total_valid_samples
        for hop in (2, 3, 4)
    }

    # ========================================================
    # 4. TARGET SAMPLE COUNTS
    # ========================================================

    raw_counts = {
        hop:
            hop_ratios[hop]
            * num_samples
        for hop in (2, 3, 4)
    }

    sample_counts = {
        hop:
            int(
                raw_counts[hop]
            )
        for hop in (2, 3, 4)
    }

    # ========================================================
    # 5. LARGEST REMAINDER METHOD
    # ========================================================

    assigned = sum(
        sample_counts.values()
    )

    remaining = (
        num_samples
        - assigned
    )

    remainder_order = sorted(
        (2, 3, 4),
        key=lambda hop: (
            raw_counts[hop]
            - sample_counts[hop]
        ),
        reverse=True,
    )

    for hop in remainder_order[
        :remaining
    ]:

        sample_counts[hop] += 1

    # ========================================================
    # 6. REPRODUCIBLE RANDOM SAMPLING
    # ========================================================

    rng = random.Random(
        seed
    )

    selected_samples_by_hop = {}

    for hop in (2, 3, 4):

        count = sample_counts[
            hop
        ]

        selected_samples_by_hop[
            hop
        ] = rng.sample(
            samples_by_hop[hop],
            count,
        )

    # ========================================================
    # 7. VALIDATE
    # ========================================================

    total_selected = sum(
        len(
            selected_samples_by_hop[hop]
        )
        for hop in (2, 3, 4)
    )

    assert (
        total_selected
        == num_samples
    )

    return selected_samples_by_hop


# ============================================================
# GENERATE TEACHER OUTPUTS
# ============================================================

def generate_teacher_outputs(
    samples,
    model,
    tokenizer,
    batch_size,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
):
    """
    Generate teacher reasoning for MuSiQue samples.

    Each returned sample preserves all original fields
    and receives:

        teacher_generation_output
    """

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be greater than 0."
        )

    if len(samples) == 0:

        return []

    # ========================================================
    # 1. MODEL DEVICE
    # ========================================================

    model_device = model.device

    # ========================================================
    # 2. SAVE ORIGINAL PADDING SIDE
    # ========================================================

    original_padding_side = (
        tokenizer.padding_side
    )

    tokenizer.padding_side = "left"

    # ========================================================
    # 3. CLEAN CUDA MEMORY
    # ========================================================

    gc.collect()

    if torch.cuda.is_available():

        for device_index in range(
            torch.cuda.device_count()
        ):

            with torch.cuda.device(
                device_index
            ):

                torch.cuda.empty_cache()

                torch.cuda.ipc_collect()

                torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # 4. OUTPUT CONTAINER
    # ========================================================

    generated_samples = []

    total_samples = len(
        samples
    )

    total_generated_tokens = 0

    total_generation_time = 0.0

    # ========================================================
    # 5. BATCH GENERATION
    # ========================================================

    try:

        for batch_start in range(
            0,
            total_samples,
            batch_size,
        ):

            batch_end = min(
                batch_start
                + batch_size,
                total_samples,
            )

            batch_samples = samples[
                batch_start:batch_end
            ]

            current_batch_size = len(
                batch_samples
            )

            # =================================================
            # CREATE TEACHER PROMPTS
            # =================================================

            teacher_prompts = []

            for raw_sample in batch_samples:

                sample = (
                    MuSiQueSampleConverter.from_musique(
                        raw_sample
                    )
                )

                teacher_prompt = (
                    create_teacher_prompt(
                        sample=sample
                    )
                )

                teacher_prompts.append(
                    teacher_prompt
                )

            # =================================================
            # APPLY CHAT TEMPLATE
            # =================================================

            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": teacher_prompt,
                    }
                ]
                for teacher_prompt
                in teacher_prompts
            ]

            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for messages
                in messages_batch
            ]

            # =================================================
            # TOKENIZE
            # =================================================

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            inputs = {
                key: value.to(
                    model_device
                )
                for key, value
                in inputs.items()
            }

            padded_input_length = (
                inputs[
                    "input_ids"
                ].shape[1]
            )

            # =================================================
            # GENERATE
            # =================================================

            batch_start_time = (
                time.perf_counter()
            )

            with torch.inference_mode():

                outputs = model.generate(
                    **inputs,

                    max_new_tokens=max_new_tokens,

                    do_sample=True,

                    temperature=temperature,

                    top_p=top_p,

                    pad_token_id=(
                        tokenizer.eos_token_id
                    ),
                )

            if torch.cuda.is_available():

                torch.cuda.synchronize()

            batch_generation_time = (
                time.perf_counter()
                - batch_start_time
            )

            total_generation_time += (
                batch_generation_time
            )

            # =================================================
            # EXTRACT GENERATED REGION
            # =================================================

            generated_tokens_batch = (
                outputs[
                    :,
                    padded_input_length:
                ]
            )

            # =================================================
            # PROCESS EACH SAMPLE
            # =================================================

            for i, raw_sample in enumerate(
                batch_samples
            ):

                generated_tokens = (
                    generated_tokens_batch[
                        i
                    ]
                )

                # ---------------------------------------------
                # FIND EOS
                # ---------------------------------------------

                eos_positions = (
                    generated_tokens
                    == tokenizer.eos_token_id
                )

                eos_indices = torch.nonzero(
                    eos_positions,
                    as_tuple=False,
                )

                if eos_indices.numel() > 0:

                    eos_index = int(
                        eos_indices[
                            0
                        ].item()
                    )

                    generated_tokens = (
                        generated_tokens[
                            :eos_index
                        ]
                    )

                generated_length = int(
                    generated_tokens.shape[0]
                )

                total_generated_tokens += (
                    generated_length
                )

                # ---------------------------------------------
                # DECODE
                # ---------------------------------------------

                teacher_output = (
                    tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True,
                    )
                )

                # ---------------------------------------------
                # COPY ORIGINAL SAMPLE
                # ---------------------------------------------

                output_sample = dict(
                    raw_sample
                )

                # ---------------------------------------------
                # ATTACH TEACHER OUTPUT
                # ---------------------------------------------

                output_sample[
                    "teacher_generation_output"
                ] = teacher_output

                generated_samples.append(
                    output_sample
                )

            # =================================================
            # PROGRESS
            # =================================================

            generated_so_far = len(
                generated_samples
            )

            print(
                f"[{generated_so_far:>4}/"
                f"{total_samples}] "
                f"Generated batch "
                f"{batch_start // batch_size + 1}"
                f" | "
                f"batch size: {current_batch_size}"
                f" | "
                f"{batch_generation_time:.2f}s"
            )

            # -------------------------------------------------
            # Free batch tensors
            # -------------------------------------------------

            del outputs
            del inputs
            del generated_tokens_batch

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    finally:

        tokenizer.padding_side = (
            original_padding_side
        )

    # ========================================================
    # 6. FINAL VALIDATION
    # ========================================================

    assert (
        len(generated_samples)
        == len(samples)
    )

    for sample in generated_samples:

        assert (
            "teacher_generation_output"
            in sample
        )

    # ========================================================
    # 7. STATISTICS
    # ========================================================

    average_generated_tokens = (
        total_generated_tokens
        / len(generated_samples)
    )

    print()
    print("=" * 80)
    print("TEACHER GENERATION COMPLETED")
    print("=" * 80)

    print(
        f"Total samples            : "
        f"{len(generated_samples):,}"
    )

    print(
        f"Batch size               : "
        f"{batch_size}"
    )

    print(
        f"Total generated tokens   : "
        f"{total_generated_tokens:,}"
    )

    print(
        f"Average generated/sample : "
        f"{average_generated_tokens:.1f}"
    )

    print(
        f"Generation time          : "
        f"{total_generation_time:.2f} sec"
    )

    if total_generation_time > 0:

        print(
            f"Generated tokens/sec     : "
            f"{total_generated_tokens / total_generation_time:.2f}"
        )

        print(
            f"Samples/sec              : "
            f"{len(generated_samples) / total_generation_time:.2f}"
        )

    print("=" * 80)

    return generated_samples


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    samples,
    output_path,
):
    """
    Save samples to JSONL.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for sample in samples:

            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        f"✓ Saved {len(samples):,} samples"
    )

    print(
        f"  {output_path}"
    )


# ============================================================
# MAIN TEACHER GENERATION
# ============================================================

def run_teacher_generation(
    model,
    tokenizer,
    num_samples: int = TEACHER_GENERATION_SAMPLES,
    batch_size_by_hop = BATCH_SIZE_BY_HOP,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    seed: int = SEED,
):
    """
    Full MuSiQue teacher-generation pipeline.

    Batch size is hop-dependent:

        2-hop → batch 8
        3-hop → batch 4
        4-hop → batch 4

    Pipeline:

        Load MuSiQue
            ↓
        Stratified sampling
            ↓
        2-hop / 3-hop / 4-hop
            ↓
        Qwen3-4B generation
            ↓
        Save three JSONL files
    """

    print("=" * 80)
    print("MUSIQUE TEACHER GENERATION")
    print("QWEN3-4B")
    print("FULL PIPELINE")
    print("=" * 80)

    # ========================================================
    # 1. SET SEED
    # ========================================================

    set_seed(seed)

    # ========================================================
    # 2. VALIDATE BATCH CONFIG
    # ========================================================

    for hop in (2, 3, 4):

        if hop not in batch_size_by_hop:

            raise ValueError(
                f"Missing batch size for {hop}-hop."
            )

        if batch_size_by_hop[hop] <= 0:

            raise ValueError(
                f"Batch size for {hop}-hop "
                f"must be greater than 0."
            )

    # ========================================================
    # 3. PRINT BATCH CONFIG
    # ========================================================

    print()
    print("=" * 80)
    print("BATCH SIZE CONFIGURATION")
    print("=" * 80)

    for hop in (2, 3, 4):

        print(
            f"{hop}-hop : "
            f"batch size = "
            f"{batch_size_by_hop[hop]}"
        )

    # ========================================================
    # 4. LOAD DATASET
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING MUSIQUE")
    print("=" * 80)

    dataset = load_musique_ans()

    dev_dataset = dataset[
        "dev"
    ]

    print()
    print(
        f"✓ Dev dataset loaded: "
        f"{len(dev_dataset):,} samples"
    )

    # ========================================================
    # 5. STRATIFIED SAMPLING
    # ========================================================

    print()
    print("=" * 80)
    print("PREPARING TEACHER-GENERATION SAMPLES")
    print("=" * 80)

    samples_by_hop = (
        prepare_teacher_generation_samples(
            dataset=dev_dataset,
            num_samples=num_samples,
            seed=seed,
        )
    )

    # ========================================================
    # 6. PRINT DISTRIBUTION
    # ========================================================

    print()
    print("=" * 80)
    print("SELECTED SAMPLE DISTRIBUTION")
    print("=" * 80)

    for hop in (2, 3, 4):

        count = len(
            samples_by_hop[hop]
        )

        percentage = (
            count
            / num_samples
            * 100
        )

        print(
            f"{hop}-hop"
            f"{' ' * (8 - len(str(hop)))}: "
            f"{count:,} samples "
            f"({percentage:.2f}%) "
            f"| batch={batch_size_by_hop[hop]}"
        )

    print()
    print(
        f"Total selected: "
        f"{num_samples:,}"
    )

    # ========================================================
    # 7. OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "✓ Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    # ========================================================
    # 8. GENERATE EACH HOP
    # ========================================================

    output_paths = {}

    for hop in (2, 3, 4):

        print()
        print("=" * 80)
        print(
            f"GENERATING {hop}-HOP TEACHER OUTPUTS"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # Select hop-specific batch size
        # ----------------------------------------------------

        hop_batch_size = (
            batch_size_by_hop[hop]
        )

        print()
        print(
            f"Hop: {hop}"
        )

        print(
            f"Batch size: "
            f"{hop_batch_size}"
        )

        hop_samples = (
            samples_by_hop[hop]
        )

        generated_samples = (
            generate_teacher_outputs(
                samples=hop_samples,
                model=model,
                tokenizer=tokenizer,
                batch_size=hop_batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"musique_{hop}_hop.jsonl"
        )

        save_jsonl(
            samples=generated_samples,
            output_path=output_path,
        )

        output_paths[
            hop
        ] = output_path

        # ----------------------------------------------------
        # FREE MEMORY
        # ----------------------------------------------------

        del generated_samples

        gc.collect()

        if torch.cuda.is_available():

            for device_index in range(
                torch.cuda.device_count()
            ):

                with torch.cuda.device(
                    device_index
                ):

                    torch.cuda.empty_cache()

    # ========================================================
    # 9. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("MUSIQUE TEACHER GENERATION FINISHED")
    print("=" * 80)

    print()
    print(
        f"Total samples generated: "
        f"{num_samples:,}"
    )

    print()

    for hop in (2, 3, 4):

        print(
            f"{hop}-hop: "
            f"{len(samples_by_hop[hop]):,} samples "
            f"| batch={batch_size_by_hop[hop]}"
        )

        print(
            f"  → {output_paths[hop]}"
        )

    print()
    print("=" * 80)
    print(
        "✓ ALL TEACHER OUTPUTS GENERATED "
        "AND SAVED"
    )
    print("=" * 80)

    return output_paths


# ============================================================
# OFFICIAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 1. SET SEED
    # ========================================================

    set_seed(SEED)

    # ========================================================
    # 2. LOAD QWEN3-4B
    # ========================================================

    model, tokenizer = (
        load_teacher_model()
    )

    # ========================================================
    # 3. RUN FULL GENERATION
    # ========================================================

    run_teacher_generation(
        model=model,
        tokenizer=tokenizer,

        num_samples=(
            TEACHER_GENERATION_SAMPLES
        ),

        batch_size_by_hop=(
            BATCH_SIZE_BY_HOP
        ),

        max_new_tokens=MAX_NEW_TOKENS,

        temperature=TEMPERATURE,

        top_p=TOP_P,

        seed=SEED,
    )

    # ========================================================
    # 4. CLEAN UP
    # ========================================================

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():

        for device_index in range(
            torch.cuda.device_count()
        ):

            with torch.cuda.device(
                device_index
            ):

                torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("✓ PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)