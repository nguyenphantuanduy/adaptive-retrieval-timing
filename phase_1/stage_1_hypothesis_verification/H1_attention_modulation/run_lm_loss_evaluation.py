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

FORWARD_BATCH_SIZE = 4

LOSS_BATCH_SIZE = 1

NUM_DECISION_LEVELS = 32

SAVE_EVERY = 100

OUTPUT_DIR = Path(
    "phase_1/stage_1_hypothesis_verification/"
    "H1_attention_modulation/lm_loss_results"
)


# ============================================================
# CUDA OOM CLEANUP
# ============================================================

def _cleanup_cuda_after_oom() -> None:
    """
    Aggressive CUDA cleanup.

    IMPORTANT:
    This function is called ONLY after an OOM.
    """

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _is_cuda_oom(
    error: BaseException,
) -> bool:
    """
    Detect CUDA out-of-memory errors.
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
# COMPUTE CE FOR ONE SAMPLE
# ============================================================

def _compute_single_loss(
    logits_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    device: torch.device,
) -> float:

    assert logits_cpu.shape[0] == 1
    assert labels_cpu.shape[0] == 1

    logits = None
    labels = None
    loss = None

    try:

        logits = logits_cpu.to(
            device,
            non_blocking=True,
        )

        labels = labels_cpu.to(
            device,
            non_blocking=True,
        )

        with torch.inference_mode():

            loss = compute_per_sample_lm_loss(
                logits=logits,
                labels=labels,
            )

        result = (
            loss
            .detach()
            .cpu()
            .item()
        )

        return float(result)

    finally:

        del logits
        del labels
        del loss


# ============================================================
# FORWARD BATCH + IMMEDIATE CE
# ============================================================

def _forward_and_compute_loss_batch(
    model,
    model_input: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
) -> list[float]:
    """
    Forward a batch and immediately calculate CE.

    Normal flow:

        CPU input
            ↓
        GPU
            ↓
        forward batch=4
            ↓
        logits on GPU
            ↓
        sample 0 → CPU → CE batch=1
        sample 1 → CPU → CE batch=1
        sample 2 → CPU → CE batch=1
        sample 3 → CPU → CE batch=1
            ↓
        losses

    No CUDA cleanup is performed here during normal execution.
    """

    batch_input = None
    outputs = None
    batch_logits = None

    try:

        # ----------------------------------------------------
        # Slice input
        # ----------------------------------------------------

        batch_input = _slice_model_input(
            model_input,
            start,
            end,
        )

        actual_batch_size = end - start

        assert (
            actual_batch_size
            <= FORWARD_BATCH_SIZE
        )

        # ----------------------------------------------------
        # Move input to GPU
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

            batch_logits = outputs.logits

        # ----------------------------------------------------
        # CE one sample at a time
        # ----------------------------------------------------

        losses: list[float] = []

        for local_index in range(
            actual_batch_size
        ):

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

            except torch.cuda.OutOfMemoryError:

                # ------------------------------------------------
                # CE itself OOMed.
                #
                # Cleanup and retry the same sample.
                # ------------------------------------------------

                _cleanup_cuda_after_oom()

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

                except torch.cuda.OutOfMemoryError:

                    _cleanup_cuda_after_oom()

                    losses.append(
                        float("nan")
                    )

            finally:

                del logits_single
                del labels_single

        return losses

    finally:

        del outputs
        del batch_logits
        del batch_input


# ============================================================
# SAFE FORWARD BATCH
# ============================================================

def _process_forward_batch_safe(
    model,
    model_input: dict[str, Any],
    start: int,
    end: int,
    device: torch.device,
) -> list[float]:
    """
    Try forward batch=4.

    If OOM:

        batch=4
            ↓
           OOM
            ↓
        cleanup
            ↓
        retry each decision with batch=1

    No printing here.
    """

    try:

        return _forward_and_compute_loss_batch(
            model=model,
            model_input=model_input,
            start=start,
            end=end,
            device=device,
        )

    except torch.cuda.OutOfMemoryError:

        _cleanup_cuda_after_oom()

        recovered_losses: list[float] = []

        # ----------------------------------------------------
        # Retry every decision individually
        # ----------------------------------------------------

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

                _cleanup_cuda_after_oom()

                recovered_losses.append(
                    float("nan")
                )

            except RuntimeError as error:

                if not _is_cuda_oom(error):
                    raise

                _cleanup_cuda_after_oom()

                recovered_losses.append(
                    float("nan")
                )

        return recovered_losses

    except RuntimeError as error:

        if not _is_cuda_oom(error):
            raise

        _cleanup_cuda_after_oom()

        recovered_losses: list[float] = []

        # ----------------------------------------------------
        # Retry every decision individually
        # ----------------------------------------------------

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

                _cleanup_cuda_after_oom()

                recovered_losses.append(
                    float("nan")
                )

            except RuntimeError as single_error:

                if not _is_cuda_oom(
                    single_error
                ):
                    raise

                _cleanup_cuda_after_oom()

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
    Compute all 33 retrieval-decision losses.

    Forward:

        batch = 4

    CE:

        batch = 1

    Sequence:

        0 1 2 3
            forward
            CE(0)
            CE(1)
            CE(2)
            CE(3)

        4 5 6 7
            forward
            CE(4)
            CE(5)
            CE(6)
            CE(7)

        ...

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

    # --------------------------------------------------------
    # Process forward batches
    # --------------------------------------------------------

    for start in range(
        0,
        num_samples,
        FORWARD_BATCH_SIZE,
    ):

        end = min(
            start + FORWARD_BATCH_SIZE,
            num_samples,
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

    # --------------------------------------------------------
    # Final sanity check
    # --------------------------------------------------------

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
# GLOBAL CSV SCHEMA
# ============================================================

def _build_global_schema(
    datasets: dict[str, list[dict[str, Any]]],
) -> list[tuple[int, int]]:

    all_ratios: list[tuple[int, int]] = []

    for samples in datasets.values():

        for sample in samples:

            assert "id" in sample

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

    return sorted(
        all_ratios,
        key=lambda x: (
            -x[0],
            x[1],
        ),
    )


# ============================================================
# CSV HEADER
# ============================================================

def _build_fieldnames(
    all_ratios: list[tuple[int, int]],
) -> list[str]:

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
# EVALUATE ONE SAMPLE
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

    assert "id" in sample

    sample_id = sample["id"]

    lm_loss_samples = (
        create_lm_loss_samples(
            sample
        )
    )

    row = {
        column: ""
        for column in all_columns
    }

    row["sample_id"] = sample_id
    row["sample_index"] = sample_index

    # --------------------------------------------------------
    # Every document ratio
    # --------------------------------------------------------

    for ratio, lm_sample in (
        lm_loss_samples.items()
    ):

        model_input = (
            create_model_input_for_lm_loss(
                lm_sample,
                tokenizer,
            )
        )

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

        ratio_losses = (
            _compute_ratio_losses(
                model=model,
                model_input=model_input,
                device=device,
            )
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        for decision_level, loss in enumerate(
            ratio_losses
        ):

            column = _make_column_name(
                ratio,
                decision_level,
            )

            row[column] = loss

        del model_input
        del lm_sample
        del ratio_losses

    del lm_loss_samples

    return row


# ============================================================
# CSV
# ============================================================

def _append_rows_to_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:

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


def _initialize_csv(
    output_path: Path,
    fieldnames: list[str],
) -> None:

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
# MAIN EVALUATION
# ============================================================

def run_lm_loss_evaluation(
    model,
    tokenizer,
    mode: str = "full",
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Path]:

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

    model.eval()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    all_datasets = (
        load_teacher_generated_hypothesis_validation()
    )

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
    # TOTAL SAMPLES
    # ========================================================

    total_samples = sum(
        len(samples)
        for samples in datasets.values()
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

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    output_paths: dict[str, Path] = {}

    for hop in selected_hops:

        output_paths[hop] = (
            output_dir
            / f"lm_loss_"
            f"{hop.replace('-', '_')}.csv"
        )

    # ========================================================
    # PROGRESS
    # ========================================================

    processed_samples = 0

    # ========================================================
    # HOPS
    # ========================================================

    for hop, samples in (
        datasets.items()
    ):

        output_path = (
            output_paths[hop]
        )

        _initialize_csv(
            output_path,
            fieldnames,
        )

        rows_buffer: list[
            dict[str, Any]
        ] = []

        # ====================================================
        # SAMPLES
        # ====================================================

        for sample_index, sample in enumerate(
            samples
        ):

            assert "id" in sample

            sample_id = sample["id"]

            # ------------------------------------------------
            # ONLY SAMPLE-LEVEL PROGRESS
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
            # Evaluate
            # ------------------------------------------------

            row = _evaluate_one_sample(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                sample_index=sample_index,
                hop=hop,
                all_columns=all_columns,
                device=device,
            )

            rows_buffer.append(
                row
            )

            processed_samples += 1

            print(
                f"    ✓ complete",
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
                    f"    💾 checkpoint | "
                    f"{processed_samples:,}/"
                    f"{total_samples:,}",
                    flush=True,
                )

                rows_buffer.clear()

        # ====================================================
        # FINAL SAVE
        # ====================================================

        if rows_buffer:

            _append_rows_to_csv(
                output_path=output_path,
                fieldnames=fieldnames,
                rows=rows_buffer,
            )

            rows_buffer.clear()

        # ====================================================
        # HOP COMPLETE
        # ====================================================

        print(
            f"✓ {hop} complete | "
            f"{len(samples):,} samples",
            flush=True,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        f"✓ evaluation complete | "
        f"{processed_samples:,}/{total_samples:,} samples",
        flush=True,
    )

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

    print(
        "Loading tokenizer...",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        "Loading model...",
        flush=True,
    )

    probe = (
        AutoModelForCausalLM.from_pretrained(
            HF_REPO,

            torch_dtype=torch.bfloat16,

            device_map={"": 0},

            trust_remote_code=True,

            attn_implementation="eager",
        )
    )

    print(
        "Model loaded.",
        flush=True,
    )

    # ========================================================
    # RUN
    # ========================================================

    results = (
        run_lm_loss_evaluation(
            model=probe,
            tokenizer=tokenizer,
            mode=args.mode,
            output_dir=OUTPUT_DIR,
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    for hop, output_path in (
        results.items()
    ):

        print(
            f"{hop:<8} → {output_path}",
            flush=True,
        )