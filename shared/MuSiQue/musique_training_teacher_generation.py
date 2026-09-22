# ============================================================
# MuSiQue TRAINING SET
# TEACHER GENERATION + HUGGING FACE UPLOAD
# QWEN3-4B
#
# FULL TRAINING SET
# HOP-BASED BATCH GENERATION
# OOM RECOVERY
# SAMPLE-LEVEL OOM FALLBACK
# JSONL OUTPUT
# HUGGING FACE DATASET UPLOAD
#
#
# Hugging Face repository:
#
#   adaptive-retrieval-timing/MuSiQue-ART-Training
#
#
# Repository structure:
#
#   2-hop
#       └── train
#
#   3-hop
#       └── train
#
#   4-hop
#       └── train
#
#
# Example:
#
#   from datasets import load_dataset
#
#   dataset = load_dataset(
#       "adaptive-retrieval-timing/MuSiQue-ART-Training",
#       "2-hop",
#   )
#
#   train = dataset["train"]
#
# ============================================================


import gc
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
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


# ------------------------------------------------------------
# Hugging Face repository
# ------------------------------------------------------------

HF_REPO_ID = (
    "adaptive-retrieval-timing/"
    "MuSiQue-ART-Training"
)


# ------------------------------------------------------------
# Each hop becomes a separate Hugging Face configuration.
# ------------------------------------------------------------

CONFIGS = {
    2: "2-hop",
    3: "3-hop",
    4: "4-hop",
}


# ------------------------------------------------------------
# Training data is uploaded using the "train" split.
# ------------------------------------------------------------

HF_SPLIT = "train"


# ------------------------------------------------------------
# Batch size is hop-dependent.
#
# These batch sizes are used only for generation efficiency.
#
# There is NO sampling.
#
# ALL training samples are generated.
# ------------------------------------------------------------

BATCH_SIZE_BY_HOP = {
    2: 8,
    3: 4,
    4: 4,
}


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

MAX_NEW_TOKENS = 1024

TEMPERATURE = 0.6

TOP_P = 0.95


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

CURRENT_FILE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


OUTPUT_DIR = (
    CURRENT_FILE_DIR
    / "data"
    / "teacher_generation_training"
)


# ============================================================
# SET RANDOM SEED
# ============================================================

def set_seed(
    seed: int = 42,
):
    """
    Set random seeds.

    Generation is stochastic because:

        do_sample=True
        temperature > 0

    The seed improves reproducibility as far as supported
    by the underlying CUDA/model implementation.
    """

    random.seed(seed)

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# LOAD TEACHER MODEL
# ============================================================

def load_teacher_model():
    """
    Load Qwen3-4B teacher model and tokenizer.

    Uses automatic device mapping so the model can
    use available GPU(s).
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
        f"  Model      : {MODEL_NAME}"
    )

    print(
        "  Device map : auto"
    )

    print(
        f"  Dtype      : {model.dtype}"
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
# CLEAN CUDA MEMORY
# ============================================================

def clean_cuda_memory():
    """
    Aggressively clean Python and CUDA cached memory.

    Used after OOM before retrying generation.
    """

    gc.collect()

    if not torch.cuda.is_available():

        return

    for device_index in range(
        torch.cuda.device_count()
    ):

        with torch.cuda.device(
            device_index
        ):

            try:

                torch.cuda.empty_cache()

            except Exception:

                pass

            try:

                torch.cuda.ipc_collect()

            except Exception:

                pass


# ============================================================
# CHECK WHETHER EXCEPTION IS CUDA OOM
# ============================================================

def is_cuda_oom_error(
    error: Exception,
) -> bool:
    """
    Return True when an exception is caused by
    CUDA out-of-memory.
    """

    if isinstance(
        error,
        torch.cuda.OutOfMemoryError,
    ):

        return True

    error_message = str(
        error
    ).lower()

    oom_keywords = [
        "out of memory",
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
    ]

    return any(
        keyword in error_message
        for keyword in oom_keywords
    )


# ============================================================
# GROUP TRAINING SET BY HOP
# ============================================================

def group_training_samples_by_hop(
    dataset,
):
    """
    Split the FULL MuSiQue training set according to
    the number of question-decomposition hops.

    No sampling is performed.

    Returns
    -------
    dict
        {
            2: [...],
            3: [...],
            4: [...]
        }

    Samples whose decomposition is not a valid 2/3/4-hop
    sample are skipped.
    """

    samples_by_hop = {
        2: [],
        3: [],
        4: [],
    }

    skipped = 0

    for sample in dataset:

        decomposition = sample.get(
            "question_decomposition"
        )

        if not isinstance(
            decomposition,
            list,
        ):

            skipped += 1

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

        else:

            skipped += 1

    total_selected = sum(
        len(
            samples_by_hop[hop]
        )
        for hop in (2, 3, 4)
    )

    print()
    print("=" * 80)
    print("TRAINING SET HOP DISTRIBUTION")
    print("=" * 80)

    for hop in (2, 3, 4):

        count = len(
            samples_by_hop[hop]
        )

        percentage = (
            count
            / total_selected
            * 100
            if total_selected > 0
            else 0.0
        )

        print(
            f"{hop}-hop"
            f"{' ' * (8 - len(str(hop)))}: "
            f"{count:,} samples "
            f"({percentage:.2f}%) "
            f"| batch={BATCH_SIZE_BY_HOP[hop]}"
        )

    print()

    print(
        f"Valid 2/3/4-hop samples : "
        f"{total_selected:,}"
    )

    print(
        f"Skipped samples         : "
        f"{skipped:,}"
    )

    print("=" * 80)

    return samples_by_hop


# ============================================================
# CREATE TEACHER PROMPTS
# ============================================================

def create_batch_inputs(
    batch_samples,
    tokenizer,
    model_device,
):
    """
    Convert raw MuSiQue samples into tokenized teacher
    generation inputs.
    """

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

    # ========================================================
    # CHAT TEMPLATE
    # ========================================================

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

    # ========================================================
    # TOKENIZE
    # ========================================================

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

    return inputs


# ============================================================
# GENERATE ONE BATCH
# ============================================================

def generate_one_batch(
    batch_samples,
    model,
    tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    """
    Generate teacher outputs for one batch.

    OOM is propagated to the caller.
    """

    model_device = model.device

    # ========================================================
    # CREATE INPUTS
    # ========================================================

    inputs = create_batch_inputs(
        batch_samples=batch_samples,
        tokenizer=tokenizer,
        model_device=model_device,
    )

    padded_input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    # ========================================================
    # GENERATE
    # ========================================================

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

    # ========================================================
    # EXTRACT GENERATED REGION
    # ========================================================

    generated_tokens_batch = (
        outputs[
            :,
            padded_input_length:
        ]
    )

    generated_samples = []

    # ========================================================
    # PROCESS EACH SAMPLE
    # ========================================================

    for i, raw_sample in enumerate(
        batch_samples
    ):

        generated_tokens = (
            generated_tokens_batch[
                i
            ]
        )

        # ----------------------------------------------------
        # FIND EOS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        teacher_output = (
            tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
        )

        # ----------------------------------------------------
        # COPY ORIGINAL SAMPLE
        # ----------------------------------------------------

        output_sample = dict(
            raw_sample
        )

        # ----------------------------------------------------
        # ATTACH TEACHER OUTPUT
        # ----------------------------------------------------

        output_sample[
            "teacher_generation_output"
        ] = teacher_output

        generated_samples.append(
            output_sample
        )

    # ========================================================
    # CLEAN LOCAL TENSORS
    # ========================================================

    del outputs
    del inputs
    del generated_tokens_batch

    return generated_samples


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_one_sample(
    sample,
    model,
    tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    """
    Generate teacher output for exactly one sample.

    Used when a normal batch causes OOM.

    If the individual sample also causes OOM:

        teacher_generation_output = None
    """

    try:

        generated_samples = (
            generate_one_batch(
                batch_samples=[
                    sample
                ],
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

        return generated_samples[0]

    except Exception as error:

        if not is_cuda_oom_error(
            error
        ):

            raise

        # ====================================================
        # SAMPLE-LEVEL OOM
        # ====================================================

        print()

        print(
            "    ✗ SAMPLE OOM"
        )

        print(
            "      → Cleaning CUDA memory"
        )

        clean_cuda_memory()

        # ----------------------------------------------------
        # Keep original sample.
        #
        # None becomes NaN when loaded by pandas.
        # ----------------------------------------------------

        output_sample = dict(
            sample
        )

        output_sample[
            "teacher_generation_output"
        ] = None

        return output_sample


# ============================================================
# GENERATE ONE HOP
# ============================================================

def generate_hop(
    samples,
    hop,
    model,
    tokenizer,
    batch_size,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    """
    Generate teacher outputs for ALL samples of one hop.

    Normal path:
        configured batch size

    OOM path:
        failed batch
            ↓
        clean CUDA
            ↓
        retry each sample individually

    Sample-level OOM:
        teacher_generation_output = None
    """

    if len(samples) == 0:

        return []

    print()
    print("=" * 80)
    print(
        f"GENERATING {hop}-HOP TRAINING SAMPLES"
    )
    print("=" * 80)

    print(
        f"Total samples : "
        f"{len(samples):,}"
    )

    print(
        f"Batch size    : "
        f"{batch_size}"
    )

    generated_samples = []

    total_samples = len(
        samples
    )

    total_generation_time = 0.0

    batch_index = 0

    # ========================================================
    # BATCH LOOP
    # ========================================================

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

        batch_index += 1

        current_batch_size = len(
            batch_samples
        )

        print()

        print(
            f"[{batch_start + 1:>5}-"
            f"{batch_end:<5}/"
            f"{total_samples}] "
            f"batch={batch_index}"
            f" | size={current_batch_size}"
        )

        # ====================================================
        # NORMAL BATCH GENERATION
        # ====================================================

        batch_start_time = (
            time.perf_counter()
        )

        try:

            batch_outputs = (
                generate_one_batch(
                    batch_samples=batch_samples,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )

            batch_generation_time = (
                time.perf_counter()
                - batch_start_time
            )

            total_generation_time += (
                batch_generation_time
            )

            generated_samples.extend(
                batch_outputs
            )

            print(
                f"  ✓ Batch success "
                f"| {batch_generation_time:.2f}s"
            )

            del batch_outputs

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        # ====================================================
        # BATCH OOM
        # ====================================================

        except Exception as error:

            if not is_cuda_oom_error(
                error
            ):

                raise

            print()

            print(
                "  ✗ BATCH OOM"
            )

            print(
                f"    Failed batch size: "
                f"{current_batch_size}"
            )

            print(
                "    → Cleaning CUDA memory"
            )

            clean_cuda_memory()

            # =================================================
            # FALLBACK: SAMPLE-BY-SAMPLE
            # =================================================

            print(
                "    → Retrying each sample "
                "with batch size = 1"
            )

            for local_index, sample in enumerate(
                batch_samples
            ):

                global_index = (
                    batch_start
                    + local_index
                    + 1
                )

                print(
                    f"      "
                    f"[{global_index:>5}/"
                    f"{total_samples}] "
                    f"single sample"
                )

                sample_start_time = (
                    time.perf_counter()
                )

                output_sample = (
                    generate_one_sample(
                        sample=sample,
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )

                sample_generation_time = (
                    time.perf_counter()
                    - sample_start_time
                )

                total_generation_time += (
                    sample_generation_time
                )

                generated_samples.append(
                    output_sample
                )

                if (
                    output_sample.get(
                        "teacher_generation_output"
                    )
                    is None
                ):

                    print(
                        "        → Failed / null"
                    )

                else:

                    print(
                        f"        ✓ success "
                        f"| "
                        f"{sample_generation_time:.2f}s"
                    )

                # ------------------------------------------------
                # Clean after every fallback sample
                # ------------------------------------------------

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

    # ========================================================
    # VALIDATION
    # ========================================================

    assert (
        len(generated_samples)
        == len(samples)
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    failed_count = sum(
        sample.get(
            "teacher_generation_output"
        )
        is None
        for sample in generated_samples
    )

    successful_count = (
        len(generated_samples)
        - failed_count
    )

    print()
    print("=" * 80)
    print(
        f"{hop}-HOP GENERATION COMPLETED"
    )
    print("=" * 80)

    print(
        f"Total samples : "
        f"{len(generated_samples):,}"
    )

    print(
        f"Successful    : "
        f"{successful_count:,}"
    )

    print(
        f"Failed / null : "
        f"{failed_count:,}"
    )

    print(
        f"Generation time: "
        f"{total_generation_time:.2f} sec"
    )

    if total_generation_time > 0:

        print(
            f"Samples/sec   : "
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

    Failed OOM samples contain:

        "teacher_generation_output": null

    This is valid JSON.
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
# LOAD ONE GENERATED HOP AS HF DATASET
# ============================================================

def load_teacher_dataset(
    hop: int,
):
    """
    Load one generated MuSiQue training JSONL file
    as a Hugging Face Dataset.

    Expected file:

        data/
        teacher_generation_training/
            musique_train_2_hop.jsonl
            musique_train_3_hop.jsonl
            musique_train_4_hop.jsonl
    """

    if hop not in CONFIGS:

        raise ValueError(
            f"Unsupported hop: {hop}. "
            f"Expected one of: "
            f"{list(CONFIGS.keys())}"
        )

    input_path = (
        OUTPUT_DIR
        / f"musique_train_{hop}_hop.jsonl"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Teacher dataset not found:\n"
            f"{input_path}"
        )

    print()
    print("=" * 80)
    print(
        f"LOADING {hop}-HOP DATASET "
        f"FOR HUGGING FACE UPLOAD"
    )
    print("=" * 80)

    print()

    print(
        "Input file:"
    )

    print(
        f"  {input_path}"
    )

    # ========================================================
    # LOAD JSONL
    # ========================================================

    dataset = load_dataset(
        "json",
        data_files={
            HF_SPLIT: str(
                input_path
            ),
        },
    )

    train_dataset = dataset[
        HF_SPLIT
    ]

    print()

    print(
        "✓ Dataset loaded"
    )

    print(
        f"  Samples: "
        f"{len(train_dataset):,}"
    )

    print(
        "  Features:"
    )

    for feature in (
        train_dataset.column_names
    ):

        print(
            f"    - {feature}"
        )

    return train_dataset


# ============================================================
# VALIDATE ONE HOP DATASET
# ============================================================

def validate_teacher_dataset(
    dataset,
    hop: int,
):
    """
    Validate generated teacher dataset before upload.

    Required fields:

        id
        teacher_generation_output

    All original MuSiQue fields are preserved.

    IMPORTANT:
        teacher_generation_output is allowed to be None
        because None represents sample-level OOM.
    """

    print()
    print("=" * 80)
    print(
        f"VALIDATING {hop}-HOP DATASET"
    )
    print("=" * 80)

    # ========================================================
    # 1. CHECK EMPTY DATASET
    # ========================================================

    if len(dataset) == 0:

        raise ValueError(
            f"{hop}-hop dataset is empty."
        )

    # ========================================================
    # 2. REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "id",
        "teacher_generation_output",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in dataset.column_names
    ]

    if missing_fields:

        raise ValueError(
            f"{hop}-hop dataset is missing "
            f"required fields: "
            f"{missing_fields}"
        )

    # ========================================================
    # 3. VALIDATE IDS
    # ========================================================

    ids = dataset[
        "id"
    ]

    if len(ids) != len(
        set(ids)
    ):

        raise ValueError(
            f"{hop}-hop dataset contains "
            f"duplicate IDs."
        )

    # ========================================================
    # 4. VALIDATE TEACHER OUTPUTS
    # ========================================================

    teacher_outputs = dataset[
        "teacher_generation_output"
    ]

    failed_count = 0

    invalid_count = 0

    for teacher_output in (
        teacher_outputs
    ):

        # ----------------------------------------------------
        # None is an expected value for sample-level OOM.
        # ----------------------------------------------------

        if teacher_output is None:

            failed_count += 1

            continue

        # ----------------------------------------------------
        # Non-None values must be strings.
        # ----------------------------------------------------

        if not isinstance(
            teacher_output,
            str,
        ):

            invalid_count += 1

            continue

        # ----------------------------------------------------
        # Empty strings are invalid.
        # ----------------------------------------------------

        if not teacher_output.strip():

            invalid_count += 1

    if invalid_count > 0:

        raise ValueError(
            f"{hop}-hop dataset contains "
            f"{invalid_count:,} "
            f"invalid teacher outputs."
        )

    # ========================================================
    # 5. PRINT VALIDATION RESULT
    # ========================================================

    print()

    print(
        "✓ Validation passed"
    )

    print(
        f"  Samples         : "
        f"{len(dataset):,}"
    )

    print(
        f"  Unique IDs      : "
        f"{len(set(ids)):,}"
    )

    print(
        f"  Features        : "
        f"{len(dataset.column_names)}"
    )

    print(
        f"  Successful       : "
        f"{len(dataset) - failed_count:,}"
    )

    print(
        f"  Sample-level OOM : "
        f"{failed_count:,}"
    )


# ============================================================
# UPLOAD ONE HOP DATASET
# ============================================================

def upload_hop_dataset(
    dataset,
    hop: int,
):
    """
    Upload one hop as a separate Hugging Face
    Dataset configuration.

    Example:

        config_name = "2-hop"
        split       = "train"
    """

    config_name = CONFIGS[
        hop
    ]

    print()
    print("=" * 80)
    print(
        f"UPLOADING {hop}-HOP DATASET"
    )
    print("=" * 80)

    print()

    print(
        "Repository:"
    )

    print(
        f"  {HF_REPO_ID}"
    )

    print(
        "Configuration:"
    )

    print(
        f"  {config_name}"
    )

    print(
        "Split:"
    )

    print(
        f"  {HF_SPLIT}"
    )

    print(
        "Samples:"
    )

    print(
        f"  {len(dataset):,}"
    )

    # ========================================================
    # PUSH TO HUB
    # ========================================================

    dataset.push_to_hub(
        HF_REPO_ID,
        config_name=config_name,
        split=HF_SPLIT,
        commit_message=(
            f"Add MuSiQue {hop}-hop "
            f"teacher-generated training data"
        ),
    )

    print()

    print(
        f"✓ {hop}-hop dataset uploaded"
    )


# ============================================================
# GENERATE + SAVE + UPLOAD ONE HOP
# ============================================================

def generate_save_upload_hop(
    samples,
    hop,
    model,
    tokenizer,
    batch_size,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    """
    Complete pipeline for one hop:

        generate
            ↓
        save JSONL
            ↓
        load as HF Dataset
            ↓
        validate
            ↓
        upload to Hugging Face
    """

    # ========================================================
    # 1. GENERATE
    # ========================================================

    generated_samples = (
        generate_hop(
            samples=samples,
            hop=hop,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )

    # ========================================================
    # 2. SAVE LOCAL JSONL
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / f"musique_train_{hop}_hop.jsonl"
    )

    save_jsonl(
        samples=generated_samples,
        output_path=output_path,
    )

    # ========================================================
    # 3. LOAD AS HF DATASET
    # ========================================================

    dataset = (
        load_teacher_dataset(
            hop=hop
        )
    )

    # ========================================================
    # 4. VALIDATE
    # ========================================================

    validate_teacher_dataset(
        dataset=dataset,
        hop=hop,
    )

    # ========================================================
    # 5. UPLOAD
    # ========================================================

    upload_hop_dataset(
        dataset=dataset,
        hop=hop,
    )

    # ========================================================
    # 6. STATISTICS
    # ========================================================

    failed_count = sum(
        sample.get(
            "teacher_generation_output"
        )
        is None
        for sample in generated_samples
    )

    successful_count = (
        len(generated_samples)
        - failed_count
    )

    # ========================================================
    # CLEAN MEMORY
    # ========================================================

    del dataset
    del generated_samples

    gc.collect()

    clean_cuda_memory()

    return {
        "hop": hop,
        "samples": len(samples),
        "successful": successful_count,
        "failed": failed_count,
        "output_path": output_path,
    }


# ============================================================
# MAIN TRAINING GENERATION + UPLOAD PIPELINE
# ============================================================

def run_training_teacher_generation(
    model,
    tokenizer,
    batch_size_by_hop=BATCH_SIZE_BY_HOP,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
):
    """
    Full MuSiQue training-set teacher generation
    and Hugging Face upload.

    IMPORTANT:
        There is NO sampling.

    ALL valid 2-hop / 3-hop / 4-hop training samples
    are processed.

    Hop grouping exists only to select batch size.

    Each hop is uploaded immediately after generation:

        2-hop
            ↓
        generate
            ↓
        save
            ↓
        validate
            ↓
        upload

        3-hop
            ↓
        generate
            ↓
        save
            ↓
        validate
            ↓
        upload

        4-hop
            ↓
        generate
            ↓
        save
            ↓
        validate
            ↓
        upload
    """

    print("=" * 80)
    print(
        "MUSIQUE TRAINING TEACHER GENERATION"
    )
    print(
        "QWEN3-4B"
    )
    print(
        "FULL DATASET + HUGGING FACE UPLOAD"
    )
    print("=" * 80)

    # ========================================================
    # VALIDATE BATCH CONFIG
    # ========================================================

    for hop in (2, 3, 4):

        if hop not in batch_size_by_hop:

            raise ValueError(
                f"Missing batch size for {hop}-hop."
            )

        if (
            batch_size_by_hop[hop]
            <= 0
        ):

            raise ValueError(
                f"Batch size for {hop}-hop "
                f"must be greater than 0."
            )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOADING MUSIQUE TRAINING SET"
    )
    print("=" * 80)

    dataset = load_musique_ans()

    train_dataset = dataset[
        "train"
    ]

    print()

    print(
        f"✓ Training dataset loaded: "
        f"{len(train_dataset):,} samples"
    )

    # ========================================================
    # GROUP BY HOP
    # ========================================================

    samples_by_hop = (
        group_training_samples_by_hop(
            dataset=train_dataset
        )
    )

    # ========================================================
    # OUTPUT DIRECTORY
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

    print()

    print(
        "✓ Hugging Face repository:"
    )

    print(
        f"  {HF_REPO_ID}"
    )

    # ========================================================
    # PROCESS EACH HOP
    # ========================================================

    results = {}

    for hop in (2, 3, 4):

        print()
        print("#" * 80)
        print(
            f"STARTING {hop}-HOP PIPELINE"
        )
        print("#" * 80)

        hop_samples = (
            samples_by_hop[hop]
        )

        hop_batch_size = (
            batch_size_by_hop[hop]
        )

        # ----------------------------------------------------
        # Generate → Save → Validate → Upload
        # ----------------------------------------------------

        result = (
            generate_save_upload_hop(
                samples=hop_samples,
                hop=hop,
                model=model,
                tokenizer=tokenizer,
                batch_size=hop_batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

        results[
            hop
        ] = result

        print()
        print(
            f"✓ {hop}-hop COMPLETE"
        )

        print(
            f"  Generated : "
            f"{result['samples']:,}"
        )

        print(
            f"  Successful: "
            f"{result['successful']:,}"
        )

        print(
            f"  Failed    : "
            f"{result['failed']:,}"
        )

        print(
            f"  Uploaded  : "
            f"{HF_REPO_ID}"
            f" / {CONFIGS[hop]}"
            f" / {HF_SPLIT}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_samples = sum(
        result["samples"]
        for result in results.values()
    )

    total_successful = sum(
        result["successful"]
        for result in results.values()
    )

    total_failed = sum(
        result["failed"]
        for result in results.values()
    )

    print()
    print("=" * 80)
    print(
        "MUSIQUE TRAINING TEACHER GENERATION "
        "AND UPLOAD FINISHED"
    )
    print("=" * 80)

    print()

    print(
        f"Repository:"
    )

    print(
        f"  {HF_REPO_ID}"
    )

    print()

    print(
        f"Total samples : "
        f"{total_samples:,}"
    )

    print(
        f"Successful    : "
        f"{total_successful:,}"
    )

    print(
        f"Failed / null : "
        f"{total_failed:,}"
    )

    print()

    print(
        "Uploaded configurations:"
    )

    for hop in (2, 3, 4):

        result = results[
            hop
        ]

        print()

        print(
            f"  {CONFIGS[hop]}"
        )

        print(
            f"    Split     : "
            f"{HF_SPLIT}"
        )

        print(
            f"    Samples   : "
            f"{result['samples']:,}"
        )

        print(
            f"    Successful: "
            f"{result['successful']:,}"
        )

        print(
            f"    Failed    : "
            f"{result['failed']:,}"
        )

        print(
            f"    Local file: "
            f"{result['output_path']}"
        )

    print()
    print("=" * 80)

    print(
        "✓ ALL TRAINING DATASETS "
        "GENERATED AND UPLOADED"
    )

    print("=" * 80)

    return results


# ============================================================
# OFFICIAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 1. SET SEED
    # ========================================================

    set_seed()

    # ========================================================
    # 2. LOAD QWEN3-4B
    # ========================================================

    model, tokenizer = (
        load_teacher_model()
    )

    # ========================================================
    # 3. RUN FULL PIPELINE
    # ========================================================

    run_training_teacher_generation(
        model=model,
        tokenizer=tokenizer,

        batch_size_by_hop=(
            BATCH_SIZE_BY_HOP
        ),

        max_new_tokens=(
            MAX_NEW_TOKENS
        ),

        temperature=(
            TEMPERATURE
        ),

        top_p=(
            TOP_P
        ),
    )

    # ========================================================
    # 4. CLEAN UP
    # ========================================================

    del model
    del tokenizer

    gc.collect()

    clean_cuda_memory()

    print()
    print("=" * 80)

    print(
        "✓ PROCESS COMPLETED SUCCESSFULLY"
    )

    print("=" * 80)