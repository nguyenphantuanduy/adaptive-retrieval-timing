"""
ART H2 Gradient Direction Evaluation
====================================

For every MuSiQue validation sample:

    sample
      ↓
    create_lm_loss_samples(sample)
      ↓
    each document ratio
      ↓
    33 decision levels:
        d = 0/32, 1/32, ..., 32/32
      ↓
    create ONE HF Dataset with 33 samples
      ↓
    HF Trainer
      ↓
    batch_size = 1
      ↓
    capture ∂L/∂d with Tensor.register_hook()
      ↓
    store gradient for each ratio / decision level
      ↓
    save to CSV

Important:
- d is an ordinary Tensor, NOT nn.Parameter.
- d comes directly from the HF Dataset batch.
- d.requires_grad_(True) is enabled inside forward().
- Hook is registered on the actual Tensor used by ART forward().
- The actual ART model is used directly.
- No GradientProbeModel wrapper.
- No custom Trainer.
- No custom Dataset.
- HF Dataset is used directly.
- ART model parameters are frozen.
- learning_rate = 0.
- Trainer uses gradient checkpointing.
- batch size = 1.
- The gradient is mapped back to its decision level using
  the actual d value, so Dataset sampling order does not matter.
- If an OOM occurs:
      cleanup CUDA
      retry missing decisions individually
      persistent OOM -> NaN
"""


import argparse
import csv
import gc
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from shared.MuSiQue.load_hypothesis_validation import (
    load_teacher_generated_hypothesis_validation,
)

from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.build_lm_loss_samples import (
    create_lm_loss_samples,
)

from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.model_input import (
    create_model_input_for_lm_loss,
)


# ============================================================
# CONFIG
# ============================================================

NUM_DECISION_LEVELS = 32

NUM_DECISIONS = (
    NUM_DECISION_LEVELS + 1
)

MODEL_NAME = (
    "Qwen/Qwen2.5-3B-Instruct"
)

HF_REPO = (
    "adaptive-retrieval-timing/"
    "ART-Attention-Mixing-Probe"
)

OUTPUT_DIR = Path(
    "phase_1/stage_2_hypothesis_verification/"
    "H2_gradient_direction/"
    "gradient_results"
)

SAVE_EVERY = 100

TRAINER_OUTPUT_DIR = (
    "./art_gradient_evaluation_tmp"
)


# ============================================================
# CUDA CLEANUP
# ============================================================

def _cleanup_cuda_after_oom() -> None:
    """
    CUDA cleanup after an out-of-memory error.
    """

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _cleanup_cuda() -> None:
    """
    General CUDA cleanup.
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
# GRADIENT CAPTURE
# ============================================================

class GradientCapture:
    """
    Store gradients by their actual decision index.

    We do NOT assume that Trainer returns Dataset samples
    in Dataset order.

    Instead:

        actual d
          ↓
        decision_index
          ↓
        gradients[decision_index]
    """

    def __init__(self):

        self.gradients: dict[
            int,
            float,
        ] = {}

    def hook(
        self,
        decision_index: int,
        grad: torch.Tensor,
    ) -> None:
        """
        Capture ∂L/∂d.

        The gradient itself is not modified.
        """

        gradient_value = (
            grad.detach()
            .float()
            .mean()
            .item()
        )

        self.gradients[
            decision_index
        ] = gradient_value


# ============================================================
# DECISION INDEX
# ============================================================

def _decision_index_from_tensor(
    d: torch.Tensor,
) -> int:
    """
    Convert the actual retrieval_decision Tensor
    into its corresponding decision index.

    Expected:

        d = 0/32  -> 0
        d = 1/32  -> 1
        ...
        d = 32/32 -> 32
    """

    d_value = (
        d.detach()
        .float()
        .reshape(-1)
        .item()
    )

    decision_index = int(
        round(
            d_value
            * NUM_DECISION_LEVELS
        )
    )

    if not (
        0
        <= decision_index
        <= NUM_DECISION_LEVELS
    ):
        raise ValueError(
            "Invalid retrieval decision: "
            f"d={d_value}, "
            f"decision_index="
            f"{decision_index}"
        )

    return decision_index


# ============================================================
# BUILD HF DATASET
# ============================================================

def _build_dataset(
    model_input: dict[str, Any],
) -> Dataset:
    """
    Convert the existing 33-row model_input
    directly into an HF Dataset.

    No new decision values are generated.

    Expected:

        input_ids:
            (33, seq_len)

        attention_mask:
            (33, seq_len)

        document_attention_mask:
            (33, seq_len)

        retrieval_decision:
            (33,)

        labels:
            (33, seq_len)
    """

    num_samples = (
        model_input[
            "input_ids"
        ].shape[0]
    )

    assert (
        num_samples
        == NUM_DECISIONS
    ), (
        "Expected "
        f"{NUM_DECISIONS} decisions, "
        f"got {num_samples}"
    )

    # --------------------------------------------------------
    # Verify first dimension
    # --------------------------------------------------------

    for key, value in (
        model_input.items()
    ):

        if torch.is_tensor(value):

            assert (
                value.shape[0]
                == num_samples
            ), (
                f"{key}: expected first "
                f"dimension {num_samples}, "
                f"got {value.shape[0]}"
            )

    # --------------------------------------------------------
    # Directly copy existing values
    # --------------------------------------------------------

    dataset_dict = {}

    for key, value in (
        model_input.items()
    ):

        if not torch.is_tensor(value):
            continue

        dataset_dict[key] = (
            value.detach()
            .cpu()
            .tolist()
        )

    dataset = Dataset.from_dict(
        dataset_dict
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert (
        len(dataset)
        == NUM_DECISIONS
    )

    assert (
        dataset[0][
            "retrieval_decision"
        ]
        == 0.0
    )

    assert (
        abs(
            dataset[
                NUM_DECISION_LEVELS
            ][
                "retrieval_decision"
            ]
            - 1.0
        )
        < 1e-7
    )

    return dataset


# ============================================================
# PATCH ART FORWARD
# ============================================================

def _install_gradient_hook(
    model,
    capture: GradientCapture,
):
    """
    Patch model.forward().

    The exact retrieval_decision Tensor received from Trainer
    is used directly:

        d.requires_grad_(True)

        d.register_hook(...)

    No replacement d is created.
    """

    original_forward = (
        model.forward
    )

    def forward_with_gradient_hook(
        *args,
        **kwargs,
    ):

        # ----------------------------------------------------
        # Actual Tensor coming from Trainer
        # ----------------------------------------------------

        d = kwargs[
            "retrieval_decision"
        ]

        assert torch.is_tensor(d)

        # ----------------------------------------------------
        # Determine actual decision index
        # ----------------------------------------------------

        decision_index = (
            _decision_index_from_tensor(
                d
            )
        )

        # ----------------------------------------------------
        # Enable gradient on actual Tensor
        # ----------------------------------------------------

        d.requires_grad_(True)

        # ----------------------------------------------------
        # Register hook on exact Tensor
        # ----------------------------------------------------

        d.register_hook(
            lambda grad,
            idx=decision_index:
                capture.hook(
                    idx,
                    grad,
                )
        )

        # ----------------------------------------------------
        # Original ART forward
        # ----------------------------------------------------

        return original_forward(
            *args,
            **kwargs,
        )

    model.forward = (
        forward_with_gradient_hook
    )

    return original_forward


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

def _build_training_args(
    max_steps: int,
) -> TrainingArguments:
    """
    Build Trainer arguments.

    Batch size is always 1.
    """

    return TrainingArguments(

        output_dir=(
            TRAINER_OUTPUT_DIR
        ),

        # ----------------------------------------------------
        # ONE decision per batch
        # ----------------------------------------------------

        per_device_train_batch_size=1,

        # ----------------------------------------------------
        # Number of requested Trainer steps
        # ----------------------------------------------------

        max_steps=max_steps,

        # ----------------------------------------------------
        # NEVER update model parameters
        # ----------------------------------------------------

        learning_rate=0.0,

        # ----------------------------------------------------
        # Disable Trainer logging
        # ----------------------------------------------------

        logging_strategy="no",

        report_to="none",

        disable_tqdm=True,

        # ----------------------------------------------------
        # Keep all dataset columns
        # ----------------------------------------------------

        remove_unused_columns=False,

        # ----------------------------------------------------
        # No checkpoint files
        # ----------------------------------------------------

        save_strategy="no",

        eval_strategy="no",

        # ----------------------------------------------------
        # No gradient accumulation
        # ----------------------------------------------------

        gradient_accumulation_steps=1,

        # ----------------------------------------------------
        # Gradient checkpointing
        # ----------------------------------------------------

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        

        # ----------------------------------------------------
        # Dataloader
        # ----------------------------------------------------

        dataloader_num_workers=0,
    )


# ============================================================
# RUN ONE TRAINER
# ============================================================

def _run_trainer(
    model,
    dataset: Dataset,
    capture: GradientCapture,
) -> None:
    """
    Run one Trainer.

    Returns normally if all requested samples finish.

    Raises CUDA OOM if Trainer encounters OOM.
    """

    trainer = Trainer(

        model=model,

        args=_build_training_args(
            max_steps=len(dataset)
        ),

        train_dataset=dataset,
    )

    try:

        trainer.train()

    finally:

        del trainer

        _cleanup_cuda()


# ============================================================
# RUN ONE DECISION
# ============================================================

def _run_single_decision(
    model,
    dataset: Dataset,
    decision_index: int,
    capture: GradientCapture,
) -> bool:
    """
    Run one decision with batch_size=1.

    Returns:

        True
            gradient successfully captured

        False
            OOM -> NaN
    """

    single_dataset = dataset.select(
        [decision_index]
    )

    try:

        _run_trainer(
            model=model,
            dataset=single_dataset,
            capture=capture,
        )

        return (
            decision_index
            in capture.gradients
        )

    except torch.cuda.OutOfMemoryError:

        _cleanup_cuda_after_oom()

        return False

    except RuntimeError as error:

        if not _is_cuda_oom(error):
            raise

        _cleanup_cuda_after_oom()

        return False

    finally:

        del single_dataset

        _cleanup_cuda()


# ============================================================
# MEASURE ONE RATIO
# ============================================================

def _measure_ratio_gradients(
    model,
    model_input: dict[str, Any],
    ratio: tuple[int, int],
) -> list[float]:
    """
    Measure all 33 ∂L/∂d values for one ratio.

    Normal path:

        Dataset(33)
            ↓
        Trainer
            ↓
        batch=1
            ↓
        33 gradients

    OOM path:

        Trainer
            ↓
           OOM
            ↓
         cleanup
            ↓
        retry missing decisions individually
            ↓
        persistent OOM
            ↓
           NaN
    """

    dataset = _build_dataset(
        model_input
    )

    capture = GradientCapture()

    original_forward = (
        _install_gradient_hook(
            model=model,
            capture=capture,
        )
    )

    try:

        # ====================================================
        # FIRST ATTEMPT
        # ====================================================

        try:

            _run_trainer(
                model=model,
                dataset=dataset,
                capture=capture,
            )

        except torch.cuda.OutOfMemoryError:

            _cleanup_cuda_after_oom()

        except RuntimeError as error:

            if not _is_cuda_oom(error):
                raise

            _cleanup_cuda_after_oom()

        # ====================================================
        # RETRY MISSING DECISIONS
        # ====================================================

        missing_decisions = [
            decision_index
            for decision_index
            in range(NUM_DECISIONS)
            if decision_index
            not in capture.gradients
        ]

        failed_decisions = []

        for decision_index in (
            missing_decisions
        ):

            success = (
                _run_single_decision(
                    model=model,
                    dataset=dataset,
                    decision_index=decision_index,
                    capture=capture,
                )
            )

            if not success:

                failed_decisions.append(
                    decision_index
                )

        # ====================================================
        # BUILD ORDERED RESULTS
        # ====================================================

        gradients = []

        for decision_index in range(
            NUM_DECISIONS
        ):

            if (
                decision_index
                in capture.gradients
            ):

                gradient = (
                    capture.gradients[
                        decision_index
                    ]
                )

            else:

                gradient = float("nan")

            gradients.append(
                gradient
            )

        # ====================================================
        # SANITY CHECK
        # ====================================================

        assert (
            len(gradients)
            == NUM_DECISIONS
        )

        return gradients

    finally:

        # ----------------------------------------------------
        # Restore original forward
        # ----------------------------------------------------

        model.forward = (
            original_forward
        )

        del dataset
        del capture

        _cleanup_cuda()


# ============================================================
# COLUMN NAME
# ============================================================

def _make_column_name(
    ratio: tuple[int, int],
    decision_level: int,
) -> str:
    """
    Example:

        2:3/0.0000
        2:3/0.0312
        ...
        2:3/1.0000
    """

    decision_score = (
        decision_level
        / NUM_DECISION_LEVELS
    )

    return (
        f"{ratio[0]}:{ratio[1]}/"
        f"{decision_score:.4f}"
    )


# ============================================================
# PARSE MODE
# ============================================================

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
# GLOBAL RATIO SCHEMA
# ============================================================

def _build_global_schema(
    datasets: dict[
        str,
        list[dict[str, Any]],
    ],
) -> list[tuple[int, int]]:
    """
    Collect all ratios appearing in all selected samples.
    """

    all_ratios: list[
        tuple[int, int]
    ] = []

    for samples in (
        datasets.values()
    ):

        for sample in samples:

            assert "id" in sample

            sample_ratios = (
                create_lm_loss_samples(
                    sample
                ).keys()
            )

            for ratio in (
                sample_ratios
            ):

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
# CSV FIELDNAMES
# ============================================================

def _build_fieldnames(
    all_ratios: list[
        tuple[int, int]
    ],
) -> list[str]:

    all_columns = [

        _make_column_name(
            ratio,
            decision_level,
        )

        for ratio in all_ratios

        for decision_level in range(
            NUM_DECISIONS
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
) -> dict[str, Any]:
    """
    Evaluate every ratio for one sample.

    Output:

        {
            sample_id,
            sample_index,
            ratio/decision: gradient,
            ...
        }
    """

    assert "id" in sample

    sample_id = sample[
        "id"
    ]

    # --------------------------------------------------------
    # Create all ratio variants
    # --------------------------------------------------------

    lm_loss_samples = (
        create_lm_loss_samples(
            sample
        )
    )

    # --------------------------------------------------------
    # Initialize empty row
    # --------------------------------------------------------

    row = {
        column: ""
        for column in all_columns
    }

    row[
        "sample_id"
    ] = sample_id

    row[
        "sample_index"
    ] = sample_index

    # ========================================================
    # EVERY DOCUMENT RATIO
    # ========================================================

    for ratio, lm_sample in (
        lm_loss_samples.items()
    ):

        # ----------------------------------------------------
        # Same H1 input construction
        # ----------------------------------------------------

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
            == NUM_DECISIONS
        ), (
            f"{hop} sample "
            f"{sample_index}: "
            f"ratio {ratio} produced "
            f"{num_decisions} decision levels "
            f"instead of "
            f"{NUM_DECISIONS}"
        )

        # ----------------------------------------------------
        # Gradient evaluation
        # ----------------------------------------------------

        ratio_gradients = (
            _measure_ratio_gradients(
                model=model,
                model_input=model_input,
                ratio=ratio,
            )
        )

        # ----------------------------------------------------
        # Store gradient values
        # ----------------------------------------------------

        for decision_level, gradient in (
            enumerate(
                ratio_gradients
            )
        ):

            column = (
                _make_column_name(
                    ratio,
                    decision_level,
                )
            )

            row[column] = gradient

        del model_input
        del lm_sample
        del ratio_gradients

        _cleanup_cuda()

    del lm_loss_samples

    return row


# ============================================================
# APPEND CSV
# ============================================================

def _append_rows_to_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[
        dict[str, Any]
    ],
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

        writer.writerows(
            rows
        )

        f.flush()


# ============================================================
# INITIALIZE CSV
# ============================================================

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

def run_gradient_evaluation(
    model,
    tokenizer,
    mode: str = "full",
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Run H2 gradient evaluation.

    One CSV is produced for each hop.
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

    selected_hops = (
        _parse_mode(
            mode
        )
    )

    # ========================================================
    # DEVICE
    # ========================================================

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
        for samples
        in datasets.values()
    )

    # ========================================================
    # GLOBAL RATIO SCHEMA
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
            NUM_DECISIONS
        )
    ]

    fieldnames = [
        "sample_id",
        "sample_index",
    ] + all_columns

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    output_paths: dict[
        str,
        Path,
    ] = {}

    for hop in selected_hops:

        output_paths[hop] = (
            output_dir
            / (
                "gradient_"
                f"{hop.replace('-', '_')}.csv"
            )
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

            sample_id = sample[
                "id"
            ]

            # ------------------------------------------------
            # Minimal progress log
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

            row = (
                _evaluate_one_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    sample_index=sample_index,
                    hop=hop,
                    all_columns=all_columns,
                )
            )

            rows_buffer.append(
                row
            )

            processed_samples += 1

            # ------------------------------------------------
            # Sample complete
            # ------------------------------------------------

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
            f"✓ {hop} complete",
            flush=True,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        f"✓ gradient evaluation complete | "
        f"{processed_samples:,}/"
        f"{total_samples:,} samples",
        flush=True,
    )

    return output_paths


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "ART H2 gradient-direction "
            "evaluation"
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

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = (
        AutoModelForCausalLM.from_pretrained(

            HF_REPO,

            torch_dtype=torch.bfloat16,

            device_map={"": 0},

            trust_remote_code=True,

            attn_implementation="eager",
        )
    )

    # ========================================================
    # FREEZE MODEL
    # ========================================================

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad_(False)

    model.eval()

    # ========================================================
    # RUN
    # ========================================================

    results = (
        run_gradient_evaluation(
            model=model,
            tokenizer=tokenizer,
            mode=args.mode,
            output_dir=OUTPUT_DIR,
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("Done.", flush=True)