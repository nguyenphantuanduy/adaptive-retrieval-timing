import argparse
import csv
import gc
from pathlib import Path
from typing import Any

import torch

from shared.MuSiQue.load_hypothesis_validation import (
    load_teacher_generated_hypothesis_validation,
)

from .build_lm_loss_samples import (
    create_lm_loss_samples,
)

from .model_input import (
    create_model_input_for_lm_loss,
)

from shared.utils.compute_per_sample_lm_loss import (
    compute_per_sample_lm_loss,
)


# ============================================================
# CONFIG
# ============================================================

# ------------------------------------------------------------
# FORWARD BATCH SIZE
#
# This controls ONLY the model forward.
#
# One ratio contains:
#
#     33 retrieval-decision levels
#
# 2-hop -> forward batch 8
# 3-hop -> forward batch 4
# 4-hop -> forward batch 4
# ------------------------------------------------------------

FORWARD_BATCH_SIZE_BY_HOP = {
    "2-hop": 8,
    "3-hop": 4,
    "4-hop": 4,
}


# ------------------------------------------------------------
# LOSS BATCH SIZE
#
# CE is intentionally computed one sample at a time.
#
# This keeps the GPU memory usage of CrossEntropyLoss low.
# ------------------------------------------------------------

LOSS_BATCH_SIZE = 1


NUM_DECISION_LEVELS = 32


# ------------------------------------------------------------
# CSV CHECKPOINT
# ------------------------------------------------------------

SAVE_EVERY = 100


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "phase_1/stage_1_hypothesis_verification/"
    "H1_attention_modulation/lm_loss_results"
)


# ============================================================
# HELPERS
# ============================================================

def _move_tensors_to_device(
    model_input: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """
    Move tensor fields of a model-input dictionary to device.

    Non-tensor fields are kept unchanged.
    """

    return {
        key: value.to(device)
        if torch.is_tensor(value)
        else value
        for key, value in model_input.items()
    }


def _slice_model_input(
    model_input: dict[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    """
    Slice the batch dimension of a model-input dictionary.
    """

    return {
        key: value[start:end]
        if torch.is_tensor(value)
        else value
        for key, value in model_input.items()
    }


def _make_column_name(
    ratio: tuple[int, int],
    decision_level: int,
) -> str:
    """
    Flatten a ratio + retrieval decision level into one CSV column.

    Example:

        ratio = (2, 1)
        decision_level = 0

        -> 2:1/0.0000
    """

    decision_score = (
        decision_level / NUM_DECISION_LEVELS
    )

    return (
        f"{ratio[0]}:{ratio[1]}/"
        f"{decision_score:.4f}"
    )


def _get_forward_batch_size_for_hop(
    hop: str,
) -> int:
    """
    Return the GPU forward batch size for a given hop.
    """

    if hop not in FORWARD_BATCH_SIZE_BY_HOP:
        raise ValueError(
            f"Unknown hop '{hop}'. "
            f"Expected one of: "
            f"{list(FORWARD_BATCH_SIZE_BY_HOP.keys())}"
        )

    return FORWARD_BATCH_SIZE_BY_HOP[hop]


def _parse_mode(
    mode: str,
) -> list[str]:
    """
    Convert CLI mode into the list of hops to evaluate.

    Supported:

        --mode 2
        --mode 3
        --mode 4
        --mode full
    """

    if mode == "2":
        return ["2-hop"]

    if mode == "3":
        return ["3-hop"]

    if mode == "4":
        return ["4-hop"]

    if mode == "full":
        return [
            "2-hop",
            "3-hop",
            "4-hop",
        ]

    raise ValueError(
        f"Invalid mode '{mode}'. "
        "Expected: 2, 3, 4, or full."
    )


# ============================================================
# FORWARD PHASE
# ============================================================

def _forward_ratio_logits(
    model,
    model_input: dict[str, Any],
    device: torch.device,
    forward_batch_size: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    Forward one ratio across all 33 retrieval-decision levels.

    IMPORTANT
    ---------
    This function ONLY performs model forward.

    It does NOT calculate CrossEntropyLoss.

    Forward batches:

        2-hop -> 8
        3-hop -> 4
        4-hop -> 4

    Each forward batch produces logits:

        [batch, sequence_length, vocab_size]

    The logits are immediately moved to CPU.

    Therefore GPU memory does NOT retain all 33 decision-level
    logits simultaneously.

    Returns
    -------

    logits_cpu:
        List containing one CPU tensor per forward batch.

    labels_cpu:
        Full labels tensor on CPU.
    """

    num_samples = (
        model_input["input_ids"].shape[0]
    )

    assert (
        num_samples == NUM_DECISION_LEVELS + 1
    ), (
        "Expected "
        f"{NUM_DECISION_LEVELS + 1} decision levels, "
        f"got {num_samples}"
    )

    # --------------------------------------------------------
    # Labels never need to be on GPU during forward.
    #
    # Keep them on CPU so they can later be moved to GPU
    # one sample at a time during CE.
    # --------------------------------------------------------

    labels_cpu = (
        model_input["labels"]
        .detach()
        .cpu()
    )

    assert labels_cpu.shape == (
        num_samples,
        model_input["labels"].shape[1],
    )

    logits_cpu: list[torch.Tensor] = []

    # ========================================================
    # FORWARD ALL DECISION LEVELS
    # ========================================================

    for start in range(
        0,
        num_samples,
        forward_batch_size,
    ):

        end = min(
            start + forward_batch_size,
            num_samples,
        )

        # ----------------------------------------------------
        # Slice CPU input
        # ----------------------------------------------------

        batch_input = _slice_model_input(
            model_input,
            start,
            end,
        )

        # ----------------------------------------------------
        # Move only this forward batch to GPU
        # ----------------------------------------------------

        batch_input = _move_tensors_to_device(
            batch_input,
            device,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                input_ids=batch_input[
                    "input_ids"
                ],
                attention_mask=batch_input[
                    "attention_mask"
                ],
                document_attention_mask=batch_input[
                    "document_attention_mask"
                ],
                retrieval_decision=batch_input[
                    "retrieval_decision"
                ],
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Immediately move logits to CPU.
            #
            # This releases the GPU copy after this iteration.
            # ------------------------------------------------

            batch_logits_cpu = (
                outputs.logits
                .detach()
                .cpu()
            )

        # ----------------------------------------------------
        # Store CPU logits
        # ----------------------------------------------------

        logits_cpu.append(
            batch_logits_cpu
        )

        # ----------------------------------------------------
        # Cleanup GPU tensors
        # ----------------------------------------------------

        del outputs
        del batch_logits_cpu
        del batch_input

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # SANITY CHECK
    # ========================================================

    total_forwarded = sum(
        logits.shape[0]
        for logits in logits_cpu
    )

    assert total_forwarded == num_samples, (
        f"Expected {num_samples} forwarded samples, "
        f"got {total_forwarded}"
    )

    return (
        logits_cpu,
        labels_cpu,
    )


# ============================================================
# LOSS PHASE
# ============================================================

def _compute_losses_from_cpu_logits(
    logits_cpu: list[torch.Tensor],
    labels_cpu: torch.Tensor,
    device: torch.device,
) -> list[float]:
    """
    Compute per-sample LM loss from CPU logits.

    IMPORTANT
    ---------
    The forward phase has already completed.

    CE is now computed separately with LOSS_BATCH_SIZE = 1.

    For every decision level:

        CPU logits
            ↓
        GPU
            ↓
        CrossEntropyLoss
            ↓
        scalar CPU value
            ↓
        delete GPU tensors

    Therefore only ONE decision level's logits is on GPU
    during loss computation.
    """

    # --------------------------------------------------------
    # Flatten forward batches into individual samples.
    #
    # We intentionally do not concatenate them into one huge
    # tensor because that would create another large CPU tensor.
    # --------------------------------------------------------

    total_samples = sum(
        logits.shape[0]
        for logits in logits_cpu
    )

    assert (
        total_samples
        == NUM_DECISION_LEVELS + 1
    )

    assert labels_cpu.shape[0] == total_samples

    ratio_losses: list[float] = []

    current_index = 0

    # ========================================================
    # PROCESS ONE SAMPLE AT A TIME
    # ========================================================

    for batch_logits_cpu in logits_cpu:

        batch_size = (
            batch_logits_cpu.shape[0]
        )

        for local_index in range(
            batch_size
        ):

            global_index = (
                current_index
                + local_index
            )

            # ------------------------------------------------
            # LOSS BATCH = 1
            # ------------------------------------------------

            logits_single_cpu = (
                batch_logits_cpu[
                    local_index:local_index + 1
                ]
            )

            labels_single_cpu = (
                labels_cpu[
                    global_index:global_index + 1
                ]
            )

            # ------------------------------------------------
            # Move ONE sample to GPU
            # ------------------------------------------------

            logits_single = (
                logits_single_cpu.to(
                    device,
                    non_blocking=True,
                )
            )

            labels_single = (
                labels_single_cpu.to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # CE
            # ------------------------------------------------

            with torch.inference_mode():

                per_sample_loss = (
                    compute_per_sample_lm_loss(
                        logits=logits_single,
                        labels=labels_single,
                    )
                )

            # ------------------------------------------------
            # Extract scalar immediately
            # ------------------------------------------------

            ratio_losses.extend(
                per_sample_loss
                .detach()
                .cpu()
                .tolist()
            )

            # ------------------------------------------------
            # Cleanup GPU
            # ------------------------------------------------

            del logits_single
            del labels_single
            del per_sample_loss

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Cleanup CPU batch logits
        # ----------------------------------------------------

        del batch_logits_cpu

    # ========================================================
    # FINAL SANITY CHECK
    # ========================================================

    assert (
        len(ratio_losses)
        == NUM_DECISION_LEVELS + 1
    ), (
        f"Expected "
        f"{NUM_DECISION_LEVELS + 1} losses, "
        f"got {len(ratio_losses)}"
    )

    return ratio_losses


# ============================================================
# COMPLETE RATIO EVALUATION
# ============================================================

def _compute_ratio_losses(
    model,
    model_input: dict[str, Any],
    device: torch.device,
    forward_batch_size: int,
) -> list[float]:
    """
    Compute LM loss for one ratio.

    This is explicitly split into TWO phases:

        PHASE 1
        -------
        Forward

        2-hop -> batch 8
        3-hop -> batch 4
        4-hop -> batch 4

        logits are moved to CPU immediately.


        PHASE 2
        -------
        CrossEntropyLoss

        batch = 1

        One logits tensor is moved to GPU,
        CE is calculated,
        scalar is moved back to CPU,
        GPU tensor is deleted.

    This avoids the previous situation where the forward
    logits and CE intermediate tensors compete for GPU memory.
    """

    # ========================================================
    # PHASE 1 — FORWARD
    # ========================================================

    logits_cpu, labels_cpu = (
        _forward_ratio_logits(
            model=model,
            model_input=model_input,
            device=device,
            forward_batch_size=forward_batch_size,
        )
    )

    # ========================================================
    # PHASE 2 — CROSS ENTROPY
    # ========================================================

    ratio_losses = (
        _compute_losses_from_cpu_logits(
            logits_cpu=logits_cpu,
            labels_cpu=labels_cpu,
            device=device,
        )
    )

    # --------------------------------------------------------
    # Cleanup CPU tensors
    # --------------------------------------------------------

    del logits_cpu
    del labels_cpu

    gc.collect()

    return ratio_losses


# ============================================================
# CREATE GLOBAL CSV SCHEMA
# ============================================================

def _build_global_schema(
    datasets: dict[str, list[dict[str, Any]]],
) -> list[tuple[int, int]]:
    """
    Collect the union of all ratios appearing in the
    selected validation datasets.
    """

    print()
    print(
        "Building global CSV schema...",
        flush=True,
    )

    all_ratios: list[tuple[int, int]] = []

    for samples in datasets.values():

        for sample in samples:

            assert "id" in sample, (
                "Every sample must contain "
                "an 'id' field."
            )

            sample_ratios = (
                create_lm_loss_samples(
                    sample
                ).keys()
            )

            for ratio in sample_ratios:

                if ratio not in all_ratios:
                    all_ratios.append(
                        ratio
                    )

    # --------------------------------------------------------
    # Deterministic ordering
    # --------------------------------------------------------

    all_ratios = sorted(
        all_ratios,
        key=lambda x: (
            -x[0],
            x[1],
        ),
    )

    return all_ratios


# ============================================================
# CREATE CSV HEADER
# ============================================================

def _build_fieldnames(
    all_ratios: list[tuple[int, int]],
) -> list[str]:
    """
    Build the complete CSV schema.
    """

    all_columns = [
        _make_column_name(
            ratio,
            decision_level,
        )
        for ratio in all_ratios
        for decision_level in range(
            NUM_DECISION_LEVELS + 1
        )
    ]

    return [
        "sample_id",
        "sample_index",
    ] + all_columns


# ============================================================
# CREATE ONE CSV ROW
# ============================================================

def _evaluate_one_sample(
    model,
    tokenizer,
    sample: dict[str, Any],
    sample_index: int,
    hop: str,
    all_columns: list[str],
    device: torch.device,
    forward_batch_size: int,
) -> dict[str, Any]:
    """
    Evaluate one dataset sample across:

        ratio
            -> 33 retrieval decision levels

    Forward and CE are performed in separate phases.
    """

    assert "id" in sample, (
        f"{hop} sample {sample_index} "
        "does not contain an 'id' field."
    )

    sample_id = sample["id"]

    # --------------------------------------------------------
    # Create LM-loss samples
    # --------------------------------------------------------

    lm_loss_samples = (
        create_lm_loss_samples(
            sample
        )
    )

    # --------------------------------------------------------
    # Initialize CSV row
    # --------------------------------------------------------

    row = {
        column: ""
        for column in all_columns
    }

    row["sample_id"] = sample_id
    row["sample_index"] = sample_index

    # ========================================================
    # PROCESS EVERY RATIO
    # ========================================================

    for ratio, lm_sample in (
        lm_loss_samples.items()
    ):

        # ----------------------------------------------------
        # Create 33 decision-level inputs
        # ----------------------------------------------------

        model_input = (
            create_model_input_for_lm_loss(
                lm_sample,
                tokenizer,
            )
        )

        # ----------------------------------------------------
        # Sanity check
        # ----------------------------------------------------

        num_decisions = (
            model_input[
                "input_ids"
            ].shape[0]
        )

        assert (
            num_decisions
            == NUM_DECISION_LEVELS + 1
        ), (
            f"{hop} sample "
            f"{sample_index}: "
            f"ratio {ratio} produced "
            f"{num_decisions} decision levels "
            f"instead of "
            f"{NUM_DECISION_LEVELS + 1}"
        )

        # ====================================================
        # FORWARD + CE
        # ====================================================

        ratio_losses = (
            _compute_ratio_losses(
                model=model,
                model_input=model_input,
                device=device,
                forward_batch_size=forward_batch_size,
            )
        )

        # ----------------------------------------------------
        # Store losses
        # ----------------------------------------------------

        for decision_level, loss in enumerate(
            ratio_losses
        ):

            column = _make_column_name(
                ratio,
                decision_level,
            )

            row[column] = loss

        # ----------------------------------------------------
        # Cleanup ratio
        # ----------------------------------------------------

        del model_input
        del lm_sample
        del ratio_losses

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    del lm_loss_samples

    gc.collect()

    return row


# ============================================================
# SAVE CSV CHUNK
# ============================================================

def _append_rows_to_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """
    Append a chunk of rows to CSV.

    The file is assumed to already have its header.
    """

    if not rows:
        return

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writerows(rows)

        f.flush()


# ============================================================
# INITIALIZE CSV
# ============================================================

def _initialize_csv(
    output_path: Path,
    fieldnames: list[str],
) -> None:
    """
    Create / overwrite a CSV and write its header.
    """

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def run_lm_loss_evaluation(
    model,
    tokenizer,
    mode: str = "full",
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Compute per-sample LM loss across:

        hop
            -> sample
                -> document ratio
                    -> retrieval decision 0/32 ... 32/32

    Forward batch size:

        2-hop -> 8
        3-hop -> 4
        4-hop -> 4

    Loss batch size:

        1 for ALL hops

    Architecture of computation:

        CPU
          |
          v
        forward batch
          |
          v
        GPU model
          |
          v
        logits
          |
          v
        CPU
          |
          |
          +---- next forward batch
          |
          v
        all 33 logits on CPU
          |
          v
        one sample -> GPU
          |
          v
        CrossEntropyLoss
          |
          v
        scalar -> CPU
          |
          v
        repeat
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SELECT HOPS
    # ========================================================

    selected_hops = _parse_mode(
        mode
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = next(
        model.parameters()
    ).device

    # ========================================================
    # MODEL MODE
    # ========================================================

    model.eval()

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 100)
    print("ART — LM LOSS EVALUATION")
    print("=" * 100)

    print(
        f"Mode             : {mode}"
    )

    print(
        f"Device           : {device}"
    )

    print(
        "Selected hops     : "
        + ", ".join(selected_hops)
    )

    print(
        "Forward batch size:"
    )

    for hop in selected_hops:

        print(
            f"  {hop:<8}: "
            f"{_get_forward_batch_size_for_hop(hop)}"
        )

    print(
        f"Loss batch size  : "
        f"{LOSS_BATCH_SIZE}"
    )

    print(
        f"Decision levels  : "
        f"0/{NUM_DECISION_LEVELS} ... "
        f"{NUM_DECISION_LEVELS}/{NUM_DECISION_LEVELS}"
    )

    print(
        f"Save every       : "
        f"{SAVE_EVERY} samples"
    )

    print(
        f"Output directory : "
        f"{output_dir}"
    )

    print()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print(
        "Loading validation dataset...",
        flush=True,
    )

    all_datasets = (
        load_teacher_generated_hypothesis_validation()
    )

    # ========================================================
    # SELECT DATASETS
    # ========================================================

    datasets = {}

    for hop in selected_hops:

        if hop not in all_datasets:
            raise ValueError(
                f"Dataset does not contain '{hop}'. "
                f"Available hops: "
                f"{list(all_datasets.keys())}"
            )

        datasets[hop] = all_datasets[hop]

    # Free reference to unselected datasets.
    del all_datasets

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print()
    print("Dataset:")

    total_samples = 0

    for hop, samples in datasets.items():

        forward_batch_size = (
            _get_forward_batch_size_for_hop(
                hop
            )
        )

        print(
            f"  {hop:<8}: "
            f"{len(samples):,} samples "
            f"(forward batch={forward_batch_size}, "
            f"loss batch={LOSS_BATCH_SIZE})"
        )

        total_samples += len(samples)

    print(
        f"  {'TOTAL':<8}: "
        f"{total_samples:,} samples"
    )

    # ========================================================
    # GLOBAL CSV SCHEMA
    # ========================================================

    all_ratios = _build_global_schema(
        datasets
    )

    all_columns = [
        _make_column_name(
            ratio,
            decision_level,
        )
        for ratio in all_ratios
        for decision_level in range(
            NUM_DECISION_LEVELS + 1
        )
    ]

    fieldnames = [
        "sample_id",
        "sample_index",
    ] + all_columns

    print(
        "Ratios:",
        [
            f"{p}:{n}"
            for p, n in all_ratios
        ],
    )

    print(
        f"LM-loss columns: "
        f"{len(all_columns):,}"
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    output_paths: dict[str, Path] = {}

    for hop in selected_hops:

        output_path = (
            output_dir
            / f"lm_loss_"
            f"{hop.replace('-', '_')}.csv"
        )

        output_paths[hop] = output_path

    # ========================================================
    # GLOBAL PROGRESS
    # ========================================================

    processed_samples = 0

    # ========================================================
    # LOOP OVER HOPS
    # ========================================================

    for hop, samples in datasets.items():

        # ----------------------------------------------------
        # Forward batch size
        # ----------------------------------------------------

        forward_batch_size = (
            _get_forward_batch_size_for_hop(
                hop
            )
        )

        output_path = output_paths[hop]

        # ----------------------------------------------------
        # Initialize CSV
        # ----------------------------------------------------

        _initialize_csv(
            output_path,
            fieldnames,
        )

        print()
        print("=" * 100)
        print(
            f"{hop.upper()} — "
            f"{len(samples):,} SAMPLES "
            f"| FORWARD BATCH = "
            f"{forward_batch_size} "
            f"| LOSS BATCH = "
            f"{LOSS_BATCH_SIZE}"
        )
        print("=" * 100)

        print(
            f"Output: {output_path}"
        )

        print()

        # ----------------------------------------------------
        # Chunk buffer
        # ----------------------------------------------------

        rows_buffer: list[dict[str, Any]] = []

        # ====================================================
        # LOOP OVER SAMPLES
        # ====================================================

        for sample_index, sample in enumerate(
            samples
        ):

            # ------------------------------------------------
            # Sample identification
            # ------------------------------------------------

            assert "id" in sample, (
                f"{hop} sample {sample_index} "
                "does not contain an 'id' field."
            )

            sample_id = sample["id"]

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"[{processed_samples + 1:,}/"
                f"{total_samples:,}] "
                f"{hop} | "
                f"sample "
                f"{sample_index + 1:,}/"
                f"{len(samples):,} | "
                f"id={sample_id}",
                flush=True,
            )

            # ------------------------------------------------
            # Evaluate sample
            # ------------------------------------------------

            row = _evaluate_one_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                sample_index=sample_index,
                hop=hop,
                all_columns=all_columns,
                device=device,
                forward_batch_size=forward_batch_size,
            )

            # ------------------------------------------------
            # Add to checkpoint buffer
            # ------------------------------------------------

            rows_buffer.append(
                row
            )

            processed_samples += 1

            print(
                f"    ✓ complete | "
                f"id={sample_id}",
                flush=True,
            )

            # =================================================
            # CHECKPOINT
            # =================================================

            if (
                len(rows_buffer)
                >= SAVE_EVERY
            ):

                _append_rows_to_csv(
                    output_path=output_path,
                    fieldnames=fieldnames,
                    rows=rows_buffer,
                )

                print(
                    f"    💾 checkpoint saved | "
                    f"{len(rows_buffer)} samples | "
                    f"total processed={processed_samples:,}",
                    flush=True,
                )

                rows_buffer.clear()

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        # ====================================================
        # SAVE REMAINING SAMPLES
        # ====================================================

        if rows_buffer:

            _append_rows_to_csv(
                output_path=output_path,
                fieldnames=fieldnames,
                rows=rows_buffer,
            )

            print(
                f"    💾 final checkpoint saved | "
                f"{len(rows_buffer)} samples",
                flush=True,
            )

            rows_buffer.clear()

        # ====================================================
        # HOP SUMMARY
        # ====================================================

        print()
        print(
            f"✓ {hop.upper()} COMPLETE"
        )

        print(
            f"  Samples          : "
            f"{len(samples):,}"
        )

        print(
            f"  Forward batch    : "
            f"{forward_batch_size}"
        )

        print(
            f"  Loss batch       : "
            f"{LOSS_BATCH_SIZE}"
        )

        print(
            f"  Saved            : "
            f"{output_path}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 100)
    print("LM LOSS EVALUATION COMPLETE")
    print("=" * 100)

    print(
        f"Total samples processed: "
        f"{processed_samples:,}"
    )

    print()

    for hop in selected_hops:

        print(
            f"{hop:<8} → "
            f"{output_paths[hop]}"
        )

    print()
    print("=" * 100)

    return output_paths


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "ART H1 LM-loss evaluation"
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "2",
            "3",
            "4",
            "full",
        ],
        help=(
            "Evaluation mode: "
            "2 = 2-hop, "
            "3 = 3-hop, "
            "4 = 4-hop, "
            "full = all hops."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    # ========================================================
    # CONFIG
    # ========================================================

    MODEL_NAME = "Qwen/Qwen3-4B"

    HF_REPO = (
        "adaptive-retrieval-timing/"
        "ART-Attention-Mixing-Probe"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    print(
        f"Tokenizer: {MODEL_NAME}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    probe = AutoModelForCausalLM.from_pretrained(
        HF_REPO,

        torch_dtype=torch.bfloat16,

        device_map={"": 0},

        trust_remote_code=True,

        attn_implementation="eager",
    )

    # ========================================================
    # MODEL CHECK
    # ========================================================

    print()
    print("=" * 80)
    print("MODEL LOADED")
    print("=" * 80)

    print(
        f"Model           : {HF_REPO}"
    )

    print(
        f"Parameter device: "
        f"{next(probe.parameters()).device}"
    )

    # ========================================================
    # RUN
    # ========================================================

    print()
    print("=" * 80)
    print(
        f"STARTING LM-LOSS EVALUATION "
        f"(MODE = {args.mode})"
    )
    print("=" * 80)

    results = run_lm_loss_evaluation(
        model=probe,
        tokenizer=tokenizer,
        mode=args.mode,
        output_dir=OUTPUT_DIR,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("ALL DONE")
    print("=" * 80)

    for hop, output_path in results.items():

        print(
            f"{hop:<8} → {output_path}"
        )