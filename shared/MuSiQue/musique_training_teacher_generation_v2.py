# ============================================================
# MuSiQue TRAINING SET
# TEACHER GENERATION + HUGGING FACE UPLOAD (v2)
# QWEN3-8B
#
# FULL TRAINING SET
# HOP-BASED BATCH GENERATION
# OOM RECOVERY (FALLBACK TO BATCH SIZE 1)
# SAMPLE-LEVEL OOM FALLBACK (NULL)
# JSONL OUTPUT
# HUGGING FACE DATASET UPLOAD
#
#
# Hugging Face repository:
#
#   adaptive-retrieval-timing/MuSiQue-ART-Training-v2
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
# MODES:
#
#   --mode full
#       Generate 2-hop + 3-hop + 4-hop
#
#   --mode 2
#       Generate only 2-hop
#
#   --mode 3
#       Generate only 3-hop
#
#   --mode 4
#       Generate only 4-hop
#
# ============================================================

import argparse
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

import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.MuSiQue.load_dataset import load_musique_ans
    from shared.MuSiQue.musique_sample import MuSiQueSampleConverter
    from shared.MuSiQue.musique_teacher_prompt_v2 import (
        DEFAULT_DEMONSTRATION_FILE,
        create_teacher_prompt,
        create_teacher_prompts,
        load_teacher_demonstrations,
    )
except ImportError:
    from .load_dataset import load_musique_ans
    from .musique_sample import MuSiQueSampleConverter
    from .musique_teacher_prompt_v2 import (
        DEFAULT_DEMONSTRATION_FILE,
        create_teacher_prompt,
        create_teacher_prompts,
        load_teacher_demonstrations,
    )


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-8B"


# ------------------------------------------------------------
# Hugging Face repository
# ------------------------------------------------------------

HF_REPO_ID = (
    "adaptive-retrieval-timing/"
    "MuSiQue-ART-Training-v2"
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
# Fallback to batch size = 1 on OOM.
#
# ALL training samples are generated.
# ------------------------------------------------------------

BATCH_SIZE_BY_HOP = {
    2: 4,
    3: 2,
    4: 2,
}


# ------------------------------------------------------------
# Generation (Qwen3-8B Non-Thinking Mode Recommended)
# ------------------------------------------------------------

MAX_NEW_TOKENS = 2048

TEMPERATURE = 0.7

TOP_P = 0.8

TOP_K = 20

MIN_P = 0.0

REPETITION_PENALTY = 1.0

ENABLE_THINKING = False


# ============================================================
# OUTPUT DIRECTORY & DEMONSTRATION FILE
# ============================================================

CURRENT_FILE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


OUTPUT_DIR = (
    CURRENT_FILE_DIR
    / "data"
    / "teacher_generation_training_v2"
)


LOCAL_DEMONSTRATION_FILE = (
    CURRENT_FILE_DIR
    / "teacher_reasoning_demonstrations.json"
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

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PARSE COMMAND-LINE MODE
# ============================================================

def parse_args():
    """
    Parse command-line arguments.

    Supported modes:

        --mode full
        --mode 2
        --mode 3
        --mode 4
    """

    parser = argparse.ArgumentParser(
        description=(
            "MuSiQue training-set teacher generation v2 (Qwen3-8B) "
            "and Hugging Face upload."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "full",
            "2",
            "3",
            "4",
        ],
        default="full",
        help=(
            "Generation mode: "
            "'full' for 2/3/4-hop, "
            "or '2', '3', '4' for a single hop."
        ),
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help="Hugging Face repository ID for dataset upload.",
    )

    return parser.parse_args()


# ============================================================
# CONVERT MODE TO HOP LIST
# ============================================================

def get_target_hops(
    mode: str,
):
    """
    Convert command-line mode into the list of hops
    that should actually be generated.

    Examples
    --------
    full -> [2, 3, 4]
    2    -> [2]
    3    -> [3]
    4    -> [4]
    """

    if mode == "full":
        return [2, 3, 4]

    return [int(mode)]


# ============================================================
# LOAD TEACHER MODEL
# ============================================================

def load_teacher_model():
    """
    Load Qwen3-8B teacher model and tokenizer.

    Uses automatic device mapping so the model can
    use available GPU(s).
    """

    print("=" * 80)
    print("LOADING TEACHER MODEL (v2)")
    print("=" * 80)

    print()
    print(f"Model: {MODEL_NAME}")

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # MODEL
    # ========================================================

    model_kwargs = {
        "device_map": "auto",
    }

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    model.eval()

    print()
    print("✓ Teacher model loaded")
    print(f"  Model      : {MODEL_NAME}")
    print("  Device map : auto")
    print(f"  Dtype      : {model.dtype}")

    if torch.cuda.is_available():
        print()
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

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

    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
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

    error_message = str(error).lower()

    oom_keywords = [
        "out of memory",
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
    ]

    return any(keyword in error_message for keyword in oom_keywords)


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
        decomposition = sample.get("question_decomposition")

        if not isinstance(decomposition, list):
            skipped += 1
            continue

        hop = len(decomposition)

        if hop in samples_by_hop:
            samples_by_hop[hop].append(sample)
        else:
            skipped += 1

    total_selected = sum(
        len(samples_by_hop[hop])
        for hop in (2, 3, 4)
    )

    print()
    print("=" * 80)
    print("TRAINING SET HOP DISTRIBUTION")
    print("=" * 80)

    for hop in (2, 3, 4):
        count = len(samples_by_hop[hop])
        percentage = (
            count / total_selected * 100
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
    print(f"Valid 2/3/4-hop samples : {total_selected:,}")
    print(f"Skipped samples         : {skipped:,}")
    print("=" * 80)

    return samples_by_hop


# ============================================================
# CREATE TEACHER PROMPTS (v2)
# ============================================================

def create_batch_inputs(
    batch_samples,
    tokenizer,
    model_device,
    demonstrations=None,
    enable_thinking: bool = ENABLE_THINKING,
):
    """
    Convert raw MuSiQue samples into tokenized teacher
    generation inputs using prompt v2 format.
    """

    texts = []

    for raw_sample in batch_samples:
        sample = MuSiQueSampleConverter.from_musique(
            raw_sample
        )

        system_prompt, user_prompt = create_teacher_prompts(
            sample=sample,
            demonstrations=demonstrations,
        )

        serialized_prompt = create_teacher_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )

        texts.append(serialized_prompt)

    # ========================================================
    # TOKENIZE WITH LEFT PADDING
    # ========================================================

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    return inputs


# ============================================================
# GENERATE ONE BATCH (DIRECT EXECUTION)
# ============================================================

def generate_one_batch(
    batch_samples,
    model,
    tokenizer,
    demonstrations=None,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    repetition_penalty=REPETITION_PENALTY,
):
    """
    Generate teacher outputs for one batch.

    OOM is propagated to the caller for adaptive halving.
    """

    model_device = model.device

    # ========================================================
    # CREATE INPUTS
    # ========================================================

    inputs = create_batch_inputs(
        batch_samples=batch_samples,
        tokenizer=tokenizer,
        model_device=model_device,
        demonstrations=demonstrations,
    )

    padded_input_length = inputs["input_ids"].shape[1]

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
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ========================================================
    # EXTRACT GENERATED REGION
    # ========================================================

    generated_tokens_batch = outputs[:, padded_input_length:]

    generated_samples = []

    # ========================================================
    # PROCESS EACH SAMPLE
    # ========================================================

    for i, raw_sample in enumerate(batch_samples):
        generated_tokens = generated_tokens_batch[i]

        # ----------------------------------------------------
        # FIND EOS
        # ----------------------------------------------------

        eos_positions = (
            generated_tokens == tokenizer.eos_token_id
        )

        eos_indices = torch.nonzero(
            eos_positions,
            as_tuple=False,
        )

        if eos_indices.numel() > 0:
            eos_index = int(eos_indices[0].item())
            generated_tokens = generated_tokens[:eos_index]

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        teacher_output = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        # ----------------------------------------------------
        # COPY ORIGINAL SAMPLE & ATTACH TEACHER OUTPUT
        # ----------------------------------------------------

        output_sample = dict(raw_sample)
        output_sample["teacher_generation_output"] = teacher_output

        generated_samples.append(output_sample)

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
    demonstrations=None,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    repetition_penalty=REPETITION_PENALTY,
):
    """
    Generate teacher output for exactly one sample.

    Used when a normal batch causes OOM.

    If the individual sample also causes OOM:

        teacher_generation_output = None
    """

    try:
        generated_samples = generate_one_batch(
            batch_samples=[sample],
            model=model,
            tokenizer=tokenizer,
            demonstrations=demonstrations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )

        return generated_samples[0]

    except Exception as error:
        if not is_cuda_oom_error(error):
            raise

        # ====================================================
        # SAMPLE-LEVEL OOM
        # ====================================================

        print()
        print("    ✗ SAMPLE OOM (batch size = 1)")
        print("      → Cleaning CUDA memory")

        clean_cuda_memory()

        # ----------------------------------------------------
        # Keep original sample.
        # None is valid JSON and will be stored as null.
        # ----------------------------------------------------

        output_sample = dict(sample)
        output_sample["teacher_generation_output"] = None

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
    demonstrations,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    repetition_penalty=REPETITION_PENALTY,
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
        retry each sample individually with batch size = 1

    Sample-level OOM:
        teacher_generation_output = None
    """

    if len(samples) == 0:
        return []

    print()
    print("=" * 80)
    print(f"GENERATING {hop}-HOP TRAINING SAMPLES (v2)")
    print("=" * 80)

    print(f"Total samples : {len(samples):,}")
    print(f"Batch size    : {batch_size}")

    generated_samples = []
    total_samples = len(samples)
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
            batch_start + batch_size,
            total_samples,
        )

        batch_samples = samples[batch_start:batch_end]
        batch_index += 1
        current_batch_size = len(batch_samples)

        print()
        print(
            f"[{batch_start + 1:>5}-"
            f"{batch_end:<5}/"
            f"{total_samples}] "
            f"batch={batch_index} | size={current_batch_size}"
        )

        batch_start_time = time.perf_counter()

        # ====================================================
        # NORMAL BATCH GENERATION
        # ====================================================

        try:
            batch_outputs = generate_one_batch(
                batch_samples=batch_samples,
                model=model,
                tokenizer=tokenizer,
                demonstrations=demonstrations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )

            batch_generation_time = (
                time.perf_counter() - batch_start_time
            )
            total_generation_time += batch_generation_time

            generated_samples.extend(batch_outputs)

            print(
                f"  ✓ Batch success | {batch_generation_time:.2f}s"
            )

            del batch_outputs
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ====================================================
        # BATCH OOM -> FALLBACK TO BATCH SIZE = 1
        # ====================================================

        except Exception as error:
            if not is_cuda_oom_error(error):
                raise

            print()
            print("  ✗ BATCH OOM")
            print(f"    Failed batch size: {current_batch_size}")
            print("    → Cleaning CUDA memory")
            clean_cuda_memory()

            print("    → Retrying each sample with batch size = 1")

            for local_index, sample in enumerate(batch_samples):
                global_index = batch_start + local_index + 1
                print(
                    f"      [{global_index:>5}/{total_samples}] single sample"
                )

                sample_start_time = time.perf_counter()

                output_sample = generate_one_sample(
                    sample=sample,
                    model=model,
                    tokenizer=tokenizer,
                    demonstrations=demonstrations,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                )

                sample_generation_time = (
                    time.perf_counter() - sample_start_time
                )
                total_generation_time += sample_generation_time

                generated_samples.append(output_sample)

                if output_sample.get("teacher_generation_output") is None:
                    print("        → Failed / null")
                else:
                    print(
                        f"        ✓ success | {sample_generation_time:.2f}s"
                    )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ========================================================
    # VALIDATION
    # ========================================================

    assert len(generated_samples) == len(samples)

    # ========================================================
    # STATISTICS
    # ========================================================

    failed_count = sum(
        sample.get("teacher_generation_output") is None
        for sample in generated_samples
    )

    successful_count = (
        len(generated_samples) - failed_count
    )

    print()
    print("=" * 80)
    print(f"{hop}-HOP GENERATION COMPLETED (v2)")
    print("=" * 80)

    print(f"Total samples   : {len(generated_samples):,}")
    print(f"Successful      : {successful_count:,}")
    print(f"Failed / null   : {failed_count:,}")
    print(f"Generation time : {total_generation_time:.2f} sec")

    if total_generation_time > 0:
        print(
            f"Samples/sec     : "
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

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(f"✓ Saved {len(samples):,} samples")
    print(f"  {output_path}")


# ============================================================
# LOAD ONE GENERATED HOP AS HF DATASET
# ============================================================

def load_teacher_dataset(
    hop: int,
):
    """
    Load one generated MuSiQue training JSONL file
    as a Hugging Face Dataset.
    """

    if hop not in CONFIGS:
        raise ValueError(
            f"Unsupported hop: {hop}. Expected one of: {list(CONFIGS.keys())}"
        )

    input_path = (
        OUTPUT_DIR
        / f"musique_train_{hop}_hop_v2.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Teacher dataset not found:\n{input_path}"
        )

    print()
    print("=" * 80)
    print(f"LOADING {hop}-HOP DATASET FOR HUGGING FACE UPLOAD (v2)")
    print("=" * 80)

    print()
    print(f"Input file: {input_path}")

    # ========================================================
    # LOAD JSONL
    # ========================================================

    dataset = load_dataset(
        "json",
        data_files={
            HF_SPLIT: str(input_path),
        },
    )

    train_dataset = dataset[HF_SPLIT]

    print()
    print("✓ Dataset loaded")
    print(f"  Samples : {len(train_dataset):,}")
    print("  Features:")
    for feature in train_dataset.column_names:
        print(f"    - {feature}")

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
    teacher_generation_output is allowed to be None (null) for OOM.
    """

    print()
    print("=" * 80)
    print(f"VALIDATING {hop}-HOP DATASET (v2)")
    print("=" * 80)

    # 1. CHECK EMPTY
    if len(dataset) == 0:
        raise ValueError(f"{hop}-hop dataset is empty.")

    # 2. REQUIRED FIELDS
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
            f"{hop}-hop dataset is missing required fields: {missing_fields}"
        )

    # 3. VALIDATE IDS
    ids = dataset["id"]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{hop}-hop dataset contains duplicate IDs.")

    # 4. VALIDATE TEACHER OUTPUTS
    teacher_outputs = dataset["teacher_generation_output"]
    failed_count = 0
    invalid_count = 0

    for teacher_output in teacher_outputs:
        if teacher_output is None:
            failed_count += 1
            continue

        if not isinstance(teacher_output, str):
            invalid_count += 1
            continue

        if not teacher_output.strip():
            invalid_count += 1

    if invalid_count > 0:
        raise ValueError(
            f"{hop}-hop dataset contains {invalid_count:,} invalid teacher outputs."
        )

    # 5. PRINT SUMMARY
    print()
    print("✓ Validation passed")
    print(f"  Samples         : {len(dataset):,}")
    print(f"  Unique IDs      : {len(set(ids)):,}")
    print(f"  Features        : {len(dataset.column_names)}")
    print(f"  Successful      : {len(dataset) - failed_count:,}")
    print(f"  Sample-level OOM: {failed_count:,}")


# ============================================================
# UPLOAD ONE HOP DATASET
# ============================================================

def upload_hop_dataset(
    dataset,
    hop: int,
    repo_id: str = HF_REPO_ID,
):
    """
    Upload one hop as a separate Hugging Face
    Dataset configuration.
    """

    config_name = CONFIGS[hop]

    print()
    print("=" * 80)
    print(f"UPLOADING {hop}-HOP DATASET (v2)")
    print("=" * 80)

    print()
    print(f"Repository    : {repo_id}")
    print(f"Configuration : {config_name}")
    print(f"Split         : {HF_SPLIT}")
    print(f"Samples       : {len(dataset):,}")

    dataset.push_to_hub(
        repo_id,
        config_name=config_name,
        split=HF_SPLIT,
        commit_message=(
            f"Add MuSiQue {hop}-hop teacher-generated training data "
            f"(v2 with Qwen3-8B)"
        ),
    )

    print()
    print(f"✓ {hop}-hop dataset uploaded successfully to {repo_id}")


# ============================================================
# GENERATE + SAVE + UPLOAD ONE HOP
# ============================================================

def generate_save_upload_hop(
    samples,
    hop,
    model,
    tokenizer,
    batch_size,
    demonstrations,
    repo_id=HF_REPO_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    repetition_penalty=REPETITION_PENALTY,
):
    """
    Complete pipeline for one hop:

        generate (with adaptive halving on OOM)
            ↓
        save JSONL
            ↓
        load as HF Dataset
            ↓
        validate
            ↓
        upload to Hugging Face
    """

    # 1. GENERATE
    generated_samples = generate_hop(
        samples=samples,
        hop=hop,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        demonstrations=demonstrations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
    )

    # 2. SAVE LOCAL JSONL
    output_path = (
        OUTPUT_DIR
        / f"musique_train_{hop}_hop_v2.jsonl"
    )

    save_jsonl(
        samples=generated_samples,
        output_path=output_path,
    )

    # 3. LOAD AS HF DATASET
    dataset = load_teacher_dataset(hop=hop)

    # 4. VALIDATE
    validate_teacher_dataset(
        dataset=dataset,
        hop=hop,
    )

    # 5. UPLOAD
    upload_hop_dataset(
        dataset=dataset,
        hop=hop,
        repo_id=repo_id,
    )

    # 6. STATISTICS
    failed_count = sum(
        sample.get("teacher_generation_output") is None
        for sample in generated_samples
    )
    successful_count = len(generated_samples) - failed_count
    sample_count = len(generated_samples)

    # CLEAN MEMORY
    del dataset
    del generated_samples

    gc.collect()
    clean_cuda_memory()

    return {
        "hop": hop,
        "samples": sample_count,
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
    target_hops,
    demonstrations,
    repo_id=HF_REPO_ID,
    batch_size_by_hop=BATCH_SIZE_BY_HOP,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    repetition_penalty=REPETITION_PENALTY,
):
    """
    MuSiQue training-set teacher generation v2
    and Hugging Face upload.

    ALL valid samples belonging to target_hops are processed.
    """

    print("=" * 80)
    print("MUSIQUE TRAINING TEACHER GENERATION (v2)")
    print(f"MODEL : {MODEL_NAME}")
    print(f"REPO  : {repo_id}")
    print("FULL DATASET + OOM RECOVERY + HF UPLOAD")
    print("=" * 80)

    print()
    print(f"Target hops: {target_hops}")

    # VALIDATE BATCH CONFIG
    for hop in target_hops:
        if hop not in batch_size_by_hop:
            raise ValueError(f"Missing batch size for {hop}-hop.")
        if batch_size_by_hop[hop] <= 0:
            raise ValueError(
                f"Batch size for {hop}-hop must be greater than 0."
            )

    # LOAD DATASET
    print()
    print("=" * 80)
    print("LOADING MUSIQUE TRAINING SET")
    print("=" * 80)

    dataset = load_musique_ans()
    train_dataset = dataset["train"]

    print()
    print(f"✓ Training dataset loaded: {len(train_dataset):,} samples")

    # GROUP BY HOP
    samples_by_hop = group_training_samples_by_hop(
        dataset=train_dataset
    )

    # OUTPUT DIRECTORY
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"✓ Hugging Face repository: {repo_id}")

    results = {}

    for hop in target_hops:
        print()
        print("#" * 80)
        print(f"STARTING {hop}-HOP PIPELINE (v2)")
        print("#" * 80)

        hop_samples = samples_by_hop[hop]
        hop_batch_size = batch_size_by_hop[hop]

        print()
        print(f"Hop        : {hop}")
        print(f"Samples    : {len(hop_samples):,}")
        print(f"Batch size : {hop_batch_size}")

        result = generate_save_upload_hop(
            samples=hop_samples,
            hop=hop,
            model=model,
            tokenizer=tokenizer,
            batch_size=hop_batch_size,
            demonstrations=demonstrations,
            repo_id=repo_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )

        results[hop] = result

        print()
        print(f"✓ {hop}-hop COMPLETE")
        print(f"  Generated  : {result['samples']:,}")
        print(f"  Successful : {result['successful']:,}")
        print(f"  Failed     : {result['failed']:,}")
        print(
            f"  Uploaded   : {repo_id} / {CONFIGS[hop]} / {HF_SPLIT}"
        )

    # FINAL SUMMARY
    total_samples = sum(
        result["samples"] for result in results.values()
    )
    total_successful = sum(
        result["successful"] for result in results.values()
    )
    total_failed = sum(
        result["failed"] for result in results.values()
    )

    print()
    print("=" * 80)
    print("MUSIQUE TRAINING TEACHER GENERATION (v2) FINISHED")
    print("=" * 80)

    print()
    print(f"Repository   : {repo_id}")
    print(f"Target hops  : {target_hops}")
    print(f"Total samples: {total_samples:,}")
    print(f"Successful   : {total_successful:,}")
    print(f"Failed / null: {total_failed:,}")

    return results


# ============================================================
# OFFICIAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # 1. PARSE ARGUMENTS
    args = parse_args()
    target_hops = get_target_hops(args.mode)

    print()
    print("=" * 80)
    print("MuSiQue TEACHER GENERATION v2 (Qwen3-8B)")
    print(f"Mode        : {args.mode}")
    print(f"Target hops : {target_hops}")
    print(f"Repo ID     : {args.repo_id}")
    print("=" * 80)

    # 2. SET SEED
    set_seed()

    # 3. LOAD DEMONSTRATIONS FOR PROMPT V2
    demonstration_file = (
        LOCAL_DEMONSTRATION_FILE
        if LOCAL_DEMONSTRATION_FILE.exists()
        else DEFAULT_DEMONSTRATION_FILE
    )
    print(f"Loading demonstrations from: {demonstration_file}")
    demonstrations = load_teacher_demonstrations(demonstration_file)
    print(f"✓ Loaded {len(demonstrations)} teacher demonstrations")

    # 4. LOAD MODEL
    model, tokenizer = load_teacher_model()

    # 5. RUN PIPELINE
    run_training_teacher_generation(
        model=model,
        tokenizer=tokenizer,
        target_hops=target_hops,
        demonstrations=demonstrations,
        repo_id=args.repo_id,
        batch_size_by_hop=BATCH_SIZE_BY_HOP,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
        repetition_penalty=REPETITION_PENALTY,
    )

    # 6. CLEAN UP
    del model
    del tokenizer

    gc.collect()
    clean_cuda_memory()

    print()
    print("=" * 80)
    print("✓ PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)
