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

# Batch size is selected by hop.
#
# This is the batch size used for the 33 retrieval-decision
# levels of ONE ratio.
#
# 2-hop -> 4
# 3-hop -> 1
# 4-hop -> 1
#
BATCH_SIZE_BY_HOP = {
    "2-hop": 4,
    "3-hop": 1,
    "4-hop": 1,
}

NUM_DECISION_LEVELS = 32

# Save CSV every N completed samples.
SAVE_EVERY = 100

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


def _get_batch_size_for_hop(
    hop: str,
) -> int:
    """
    Return the GPU forward batch size for a given hop.
    """

    if hop not in BATCH_SIZE_BY_HOP:
        raise ValueError(
            f"Unknown hop '{hop}'. "
            f"Expected one of: "
            f"{list(BATCH_SIZE_BY_HOP.keys())}"
        )

    return BATCH_SIZE_BY_HOP[hop]


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
# EVALUATE ONE LM-LOSS SAMPLE
# ============================================================

def _compute_ratio_losses(
    model,
    model_input: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    """
    Forward one ratio's 33 retrieval-decision samples.

    The full 33-sample model input stays on CPU.

    It is moved to GPU in smaller batches to avoid CUDA OOM.

    Parameters
    ----------
    model:
        ART model.

    model_input:
        Input containing exactly 33 decision levels.

    device:
        CUDA device.

    batch_size:
        Number of decision levels processed simultaneously.

    Returns
    -------
    list[float]
        33 per-sample LM losses corresponding to:

            0/32, 1/32, ..., 32/32
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

    ratio_losses: list[float] = []

    # ========================================================
    # PROCESS 33 DECISION LEVELS
    # ========================================================

    for start in range(
        0,
        num_samples,
        batch_size,
    ):

        end = min(
            start + batch_size,
            num_samples,
        )

        # ----------------------------------------------------
        # CPU -> GPU
        # ----------------------------------------------------

        batch_input = _slice_model_input(
            model_input,
            start,
            end,
        )

        batch_input = _move_tensors_to_device(
            batch_input,
            device,
        )

        # ----------------------------------------------------
        # Forward + LM loss
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

            per_sample_loss = (
                compute_per_sample_lm_loss(
                    logits=outputs.logits,
                    labels=batch_input["labels"],
                )
            )

        # ----------------------------------------------------
        # Move scalar losses back to CPU
        # ----------------------------------------------------

        ratio_losses.extend(
            per_sample_loss
            .detach()
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Cleanup GPU tensors
        # ----------------------------------------------------

        del outputs
        del per_sample_loss
        del batch_input

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # SANITY CHECK
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
    batch_size: int,
) -> dict[str, Any]:
    """
    Evaluate one dataset sample across:

        ratio
            -> 33 retrieval decision levels
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
    # Initialize one CSV row
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

        # ----------------------------------------------------
        # Forward 33 decision levels
        # ----------------------------------------------------

        ratio_losses = (
            _compute_ratio_losses(
                model=model,
                model_input=model_input,
                device=device,
                batch_size=batch_size,
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

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    del lm_loss_samples

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

    This intentionally starts a fresh evaluation for the
    selected mode.
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

    Modes
    -----

        mode="2"
            Run only 2-hop.

        mode="3"
            Run only 3-hop.

        mode="4"
            Run only 4-hop.

        mode="full"
            Run 2-hop, 3-hop, and 4-hop.

    Batch sizes
    -----------

        2-hop -> 4
        3-hop -> 1
        4-hop -> 1

    Checkpointing
    -------------

    Results are written every SAVE_EVERY samples.

    This means that the entire validation set is NOT kept
    in RAM before writing the CSV.
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
        f"Mode            : {mode}"
    )

    print(
        f"Device          : {device}"
    )

    print(
        "Selected hops    : "
        + ", ".join(selected_hops)
    )

    print(
        "Batch size by hop:"
    )

    for hop in selected_hops:

        print(
            f"  {hop:<8}: "
            f"{_get_batch_size_for_hop(hop)}"
        )

    print(
        f"Decision levels : "
        f"0/{NUM_DECISION_LEVELS} ... "
        f"{NUM_DECISION_LEVELS}/{NUM_DECISION_LEVELS}"
    )

    print(
        f"Save every      : "
        f"{SAVE_EVERY} samples"
    )

    print(
        f"Output directory: "
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

    # Free reference to the unselected datasets.
    del all_datasets

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print()
    print("Dataset:")

    total_samples = 0

    for hop, samples in datasets.items():

        batch_size = _get_batch_size_for_hop(
            hop
        )

        print(
            f"  {hop:<8}: "
            f"{len(samples):,} samples "
            f"(batch={batch_size})"
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
        # Batch size
        # ----------------------------------------------------

        batch_size = _get_batch_size_for_hop(
            hop
        )

        output_path = output_paths[hop]

        # ----------------------------------------------------
        # Initialize CSV
        #
        # This overwrites the previous file for this hop.
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
            f"| BATCH SIZE = {batch_size}"
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
                batch_size=batch_size,
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
            # CHECKPOINT EVERY 100 SAMPLES
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
            f"  Samples    : "
            f"{len(samples):,}"
        )

        print(
            f"  Batch size : "
            f"{batch_size}"
        )

        print(
            f"  Saved      : "
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