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
# ALL HOPS USE THE SAME FORWARD BATCH SIZE.
#
# One ratio contains:
#
#     33 retrieval-decision levels
#
# Forward:
#
#     batch = 4
# ------------------------------------------------------------

FORWARD_BATCH_SIZE = 4


# ------------------------------------------------------------
# LOSS BATCH SIZE
#
# CrossEntropyLoss is ALWAYS computed one sample at a time.
# ------------------------------------------------------------

LOSS_BATCH_SIZE = 1


# ------------------------------------------------------------
# NUMBER OF RETRIEVAL DECISION LEVELS
# ------------------------------------------------------------

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
# CUDA MEMORY CLEANUP
# ============================================================

def _cleanup_cuda() -> None:
    """
    Aggressively clean temporary CUDA memory after OOM.

    This does NOT reset the model.

    It only removes Python garbage and releases cached
    CUDA allocator blocks.
    """

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _is_cuda_oom(error: BaseException) -> bool:
    """
    Return True if an exception is a CUDA OOM.
    """

    if isinstance(
        error,
        torch.cuda.OutOfMemoryError,
    ):
        return True

    message = str(error).lower()

    return (
        "out of memory" in message
        and "cuda" in message
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


def _parse_mode(
    mode: str,
) -> list[str]:
    """
    Convert CLI mode into the list of hops to evaluate.
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
# COMPUTE ONE CE LOSS
# ============================================================

def _compute_single_loss(
    logits_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute CE loss for exactly ONE sample.

    CPU
      |
      v
    GPU
      |
      v
    CrossEntropyLoss
      |
      v
    scalar
      |
      v
    CPU

    Returns
    -------
    float
        Per-sample LM loss.

    Raises
    ------
    torch.cuda.OutOfMemoryError
        If CE itself cannot fit.
    """

    assert logits_cpu.shape[0] == 1
    assert labels_cpu.shape[0] == 1

    logits = None
    labels = None
    loss = None

    try:

        # ----------------------------------------------------
        # Move only one sample to GPU
        # ----------------------------------------------------

        logits = logits_cpu.to(
            device,
            non_blocking=True,
        )

        labels = labels_cpu.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # CE
        # ----------------------------------------------------

        with torch.inference_mode():

            loss = compute_per_sample_lm_loss(
                logits=logits,
                labels=labels,
            )

        # ----------------------------------------------------
        # Extract scalar immediately
        # ----------------------------------------------------

        result = (
            loss
            .detach()
            .cpu()
            .item()
        )

        return float(result)

    finally:

        # ----------------------------------------------------
        # ALWAYS cleanup GPU tensors
        # ----------------------------------------------------

        del logits
        del labels
        del loss

        _cleanup_cuda()


# ============================================================
# FORWARD ONE BATCH + IMMEDIATE CE
# ============================================================

def _forward_and_compute_loss_batch(
    model,
    model_input: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
) -> list[float]:
    """
    Process ONE forward batch.

    Flow:

        CPU input
            |
            v
        GPU
            |
            v
        model forward (batch=4)
            |
            v
        logits
            |
            +--------------------+
            |                    |
            v                    v
        sample 0              sample 1
            |                    |
            v                    v
           CE                   CE
         batch=1              batch=1
            |                    |
            +---------+----------+
                      |
                      v
                 CPU losses

    The important point is that the whole forward batch is
    never sent through CE together.

    Parameters
    ----------
    start, end:
        Decision-level range.

    Returns
    -------
    list[float]
        One loss for each decision level in [start, end).
    """

    batch_input = None
    outputs = None
    batch_logits = None

    try:

        # ====================================================
        # 1. SLICE CPU INPUT
        # ====================================================

        batch_input = _slice_model_input(
            model_input,
            start,
            end,
        )

        actual_batch_size = (
            end - start
        )

        assert (
            actual_batch_size
            <= FORWARD_BATCH_SIZE
        )

        # ====================================================
        # 2. MOVE INPUT TO GPU
        # ====================================================

        batch_input = _move_tensors_to_device(
            batch_input,
            device,
        )

        # ====================================================
        # 3. FORWARD
        # ====================================================

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
            # IMPORTANT
            #
            # Keep logits on GPU only while processing this
            # forward batch.
            # ------------------------------------------------

            batch_logits = (
                outputs.logits
            )

        # ====================================================
        # 4. CE ONE SAMPLE AT A TIME
        # ====================================================

        losses: list[float] = []

        for local_index in range(
            actual_batch_size
        ):

            # ------------------------------------------------
            # Slice ONE sample.
            #
            # This creates:
            #
            #     [1, sequence_length, vocab]
            #
            # ------------------------------------------------

            logits_single = (
                batch_logits[
                    local_index:
                    local_index + 1
                ]
                .detach()
                .cpu()
            )

            labels_single = (
                batch_input[
                    "labels"
                ][
                    local_index:
                    local_index + 1
                ]
                .detach()
                .cpu()
            )

            # ------------------------------------------------
            # Delete GPU batch logits reference before CE.
            #
            # The CPU copy is now independent.
            # ------------------------------------------------

            # CE receives only batch=1.
            #
            # Note:
            # batch_logits itself still exists here because
            # the remaining samples need it.
            #
            # The actual CE tensor is only one sample.
            # ------------------------------------------------

            try:

                loss_value = (
                    _compute_single_loss(
                        logits_cpu=logits_single,
                        labels_cpu=labels_single,
                        device=device,
                    )
                )

                losses.append(
                    loss_value
                )

            finally:

                del logits_single
                del labels_single

        return losses

    finally:

        # ====================================================
        # 5. CLEAN GPU FORWARD TENSORS
        # ====================================================

        del outputs
        del batch_logits
        del batch_input

        _cleanup_cuda()


# ============================================================
# SAFE BATCH PROCESSING WITH OOM RECOVERY
# ============================================================

def _process_forward_batch_safe(
    model,
    model_input: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
) -> list[float]:
    """
    Safely process one forward batch.

    Normal case:

        batch=4
            ↓
        forward
            ↓
        CE x4

    If batch=4 causes OOM:

        batch=4
            ↓
           OOM
            ↓
        cleanup
            ↓
        retry each sample individually

    This means a temporary OOM does NOT kill the entire
    evaluation.
    """

    try:

        return _forward_and_compute_loss_batch(
            model=model,
            model_input=model_input,
            start=start,
            end=end,
            device=device,
        )

    except torch.cuda.OutOfMemoryError as error:

        print(
            f"      ⚠ CUDA OOM "
            f"during forward/CE "
            f"[decision {start}:{end}]",
            flush=True,
        )

        print(
            "        Cleaning CUDA memory and "
            "retrying this batch with forward batch=1...",
            flush=True,
        )

        _cleanup_cuda()

        # ====================================================
        # RETRY EACH DECISION LEVEL INDIVIDUALLY
        # ====================================================

        recovered_losses: list[float] = []

        for index in range(
            start,
            end,
        ):

            try:

                single_losses = (
                    _forward_and_compute_loss_batch(
                        model=model,
                        model_input=model_input,
                        start=index,
                        end=index + 1,
                        device=device,
                    )
                )

                assert len(
                    single_losses
                ) == 1

                recovered_losses.extend(
                    single_losses
                )

            except torch.cuda.OutOfMemoryError:

                print(
                    f"        ✗ OOM even at "
                    f"forward batch=1 "
                    f"for decision {index}",
                    flush=True,
                )

                _cleanup_cuda()

                # ------------------------------------------------
                # We cannot calculate this sample.
                #
                # Use NaN rather than silently pretending the
                # result is valid.
                # ------------------------------------------------

                recovered_losses.append(
                    float("nan")
                )

        return recovered_losses

    except RuntimeError as error:

        if not _is_cuda_oom(error):
            raise

        print(
            f"      ⚠ CUDA OOM "
            f"[decision {start}:{end}]",
            flush=True,
        )

        _cleanup_cuda()

        recovered_losses: list[float] = []

        for index in range(
            start,
            end,
        ):

            try:

                single_losses = (
                    _forward_and_compute_loss_batch(
                        model=model,
                        model_input=model_input,
                        start=index,
                        end=index + 1,
                        device=device,
                    )
                )

                recovered_losses.extend(
                    single_losses
                )

            except RuntimeError as single_error:

                if not _is_cuda_oom(
                    single_error
                ):
                    raise

                print(
                    f"        ✗ OOM even at "
                    f"forward batch=1 "
                    f"for decision {index}",
                    flush=True,
                )

                _cleanup_cuda()

                recovered_losses.append(
                    float("nan")
                )

        return recovered_losses


# ============================================================
# COMPUTE ALL 33 DECISION LEVELS
# ============================================================

def _compute_ratio_losses(
    model,
    model_input: dict[str, Any],
    device: torch.device,
) -> list[float]:
    """
    Compute LM loss for one ratio.

    Forward:

        batch = 4

    CE:

        batch = 1

    Flow:

        0 1 2 3
        └── forward ──┐
                      ├─ CE(0)
                      ├─ CE(1)
                      ├─ CE(2)
                      └─ CE(3)

        4 5 6 7
        └── forward ──┐
                      ├─ CE(4)
                      ├─ CE(5)
                      ├─ CE(6)
                      └─ CE(7)

        ...

    OOM recovery:

        forward batch=4
              ↓
             OOM
              ↓
        cleanup
              ↓
        retry 4 samples as batch=1
    """

    num_samples = (
        model_input[
            "input_ids"
        ].shape[0]
    )

    assert (
        num_samples
        == NUM_DECISION_LEVELS + 1
    ), (
        "Expected "
        f"{NUM_DECISION_LEVELS + 1} "
        f"decision levels, "
        f"got {num_samples}"
    )

    ratio_losses: list[float] = []

    # ========================================================
    # FORWARD BATCHES
    # ========================================================

    for start in range(
        0,
        num_samples,
        FORWARD_BATCH_SIZE,
    ):

        end = min(
            start + FORWARD_BATCH_SIZE,
            num_samples,
        )

        print(
            f"      decisions "
            f"{start:02d}:{end:02d} "
            f"(forward batch={end - start})",
            flush=True,
        )

        batch_losses = (
            _process_forward_batch_safe(
                model=model,
                model_input=model_input,
                start=start,
                end=end,
                device=device,
            )
        )

        assert (
            len(batch_losses)
            == end - start
        ), (
            f"Expected "
            f"{end - start} losses, "
            f"got {len(batch_losses)}"
        )

        ratio_losses.extend(
            batch_losses
        )

        # ----------------------------------------------------
        # Cleanup between forward batches.
        # ----------------------------------------------------

        _cleanup_cuda()

    # ========================================================
    # FINAL CHECK
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
) -> dict[str, Any]:
    """
    Evaluate one dataset sample across:

        ratio
            -> 33 retrieval decision levels

    For every ratio:

        forward batch = 4
        CE batch      = 1
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

        print(
            f"    ratio={ratio[0]}:{ratio[1]}",
            flush=True,
        )

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
        # FORWARD BATCH=4 + IMMEDIATE CE BATCH=1
        # ====================================================

        ratio_losses = (
            _compute_ratio_losses(
                model=model,
                model_input=model_input,
                device=device,
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
        # Cleanup
        # ----------------------------------------------------

        del model_input
        del lm_sample
        del ratio_losses

        _cleanup_cuda()

    del lm_loss_samples

    _cleanup_cuda()

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

    Forward:

        batch = 4 for ALL hops

    Cross entropy:

        batch = 1

    OOM:

        forward batch=4
            ↓
           OOM
            ↓
        cleanup
            ↓
        retry each sample with batch=1
    """

    output_dir = Path(
        output_dir
    )

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
        f"Mode              : {mode}"
    )

    print(
        f"Device            : {device}"
    )

    print(
        "Selected hops      : "
        + ", ".join(
            selected_hops
        )
    )

    print(
        f"Forward batch size : "
        f"{FORWARD_BATCH_SIZE}"
    )

    print(
        f"Loss batch size    : "
        f"{LOSS_BATCH_SIZE}"
    )

    print(
        f"Decision levels    : "
        f"0/{NUM_DECISION_LEVELS} ... "
        f"{NUM_DECISION_LEVELS}/"
        f"{NUM_DECISION_LEVELS}"
    )

    print(
        f"Save every         : "
        f"{SAVE_EVERY} samples"
    )

    print(
        f"Output directory   : "
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
                f"Dataset does not contain "
                f"'{hop}'. "
                f"Available hops: "
                f"{list(all_datasets.keys())}"
            )

        datasets[hop] = (
            all_datasets[hop]
        )

    del all_datasets

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print()
    print("Dataset:")

    total_samples = 0

    for hop, samples in (
        datasets.items()
    ):

        print(
            f"  {hop:<8}: "
            f"{len(samples):,} samples "
            f"(forward batch="
            f"{FORWARD_BATCH_SIZE}, "
            f"loss batch="
            f"{LOSS_BATCH_SIZE})"
        )

        total_samples += len(
            samples
        )

    print(
        f"  {'TOTAL':<8}: "
        f"{total_samples:,} samples"
    )

    # ========================================================
    # GLOBAL CSV SCHEMA
    # ========================================================

    all_ratios = (
        _build_global_schema(
            datasets
        )
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

        output_paths[hop] = (
            output_path
        )

    # ========================================================
    # GLOBAL PROGRESS
    # ========================================================

    processed_samples = 0

    # ========================================================
    # LOOP OVER HOPS
    # ========================================================

    for hop, samples in (
        datasets.items()
    ):

        output_path = (
            output_paths[hop]
        )

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
            f"{FORWARD_BATCH_SIZE} "
            f"| LOSS BATCH = "
            f"{LOSS_BATCH_SIZE}"
        )
        print("=" * 100)

        print(
            f"Output: {output_path}"
        )

        print()

        # ----------------------------------------------------
        # Checkpoint buffer
        # ----------------------------------------------------

        rows_buffer: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # LOOP OVER SAMPLES
        # ====================================================

        for sample_index, sample in enumerate(
            samples
        ):

            assert "id" in sample, (
                f"{hop} sample "
                f"{sample_index} "
                "does not contain "
                "an 'id' field."
            )

            sample_id = sample[
                "id"
            ]

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"[{processed_samples + 1:,}/"
                f"{total_samples:,}] "
                f"{hop} | sample "
                f"{sample_index + 1:,}/"
                f"{len(samples):,} | "
                f"id={sample_id}",
                flush=True,
            )

            # ------------------------------------------------
            # Evaluate
            # ------------------------------------------------

            try:

                row = (
                    _evaluate_one_sample(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        sample_index=sample_index,
                        hop=hop,
                        all_columns=all_columns,
                        device=device,
                    )
                )

            except torch.cuda.OutOfMemoryError:

                # ------------------------------------------------
                # This should normally NOT happen because every
                # individual batch already has OOM recovery.
                #
                # But keep this outer safety net.
                # ------------------------------------------------

                print(
                    f"    ⚠ Unexpected OOM at "
                    f"{hop} sample "
                    f"{sample_index}. "
                    f"Cleaning CUDA and skipping sample.",
                    flush=True,
                )

                _cleanup_cuda()

                # Create an empty row so CSV alignment
                # remains valid.
                row = {
                    column: ""
                    for column in all_columns
                }

                row["sample_id"] = (
                    sample_id
                )

                row["sample_index"] = (
                    sample_index
                )

            # ------------------------------------------------
            # Store
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
                    f"total processed="
                    f"{processed_samples:,}",
                    flush=True,
                )

                rows_buffer.clear()

                _cleanup_cuda()

        # ====================================================
        # SAVE REMAINING
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
            f"  Samples       : "
            f"{len(samples):,}"
        )

        print(
            f"  Forward batch : "
            f"{FORWARD_BATCH_SIZE}"
        )

        print(
            f"  Loss batch    : "
            f"{LOSS_BATCH_SIZE}"
        )

        print(
            f"  Saved         : "
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

    MODEL_NAME = (
        "Qwen/Qwen3-4B"
    )

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

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
        )
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

    probe = (
        AutoModelForCausalLM.from_pretrained(
            HF_REPO,

            torch_dtype=torch.bfloat16,

            device_map={"": 0},

            trust_remote_code=True,

            attn_implementation="eager",
        )
    )

    # ========================================================
    # MODEL CHECK
    # ========================================================

    print()
    print("=" * 80)
    print("MODEL LOADED")
    print("=" * 80)

    print(
        f"Model           : "
        f"{HF_REPO}"
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

    results = (
        run_lm_loss_evaluation(
            model=probe,
            tokenizer=tokenizer,
            mode=args.mode,
            output_dir=OUTPUT_DIR,
        )
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("ALL DONE")
    print("=" * 80)

    for hop, output_path in (
        results.items()
    ):

        print(
            f"{hop:<8} → "
            f"{output_path}"
        )