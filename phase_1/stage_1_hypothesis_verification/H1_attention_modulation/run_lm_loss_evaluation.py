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

BATCH_SIZE = 8

NUM_DECISION_LEVELS = 32

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
        2:1 / 0/32
        -> 2:1/0.0000
    """

    decision_score = (
        decision_level / NUM_DECISION_LEVELS
    )

    return (
        f"{ratio[0]}:{ratio[1]}/"
        f"{decision_score:.4f}"
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

    ratio_losses = []

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
            # Per-sample LM loss
            # ------------------------------------------------

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
        # Cleanup
        # ----------------------------------------------------

        del outputs
        del per_sample_loss
        del batch_input

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert (
        len(ratio_losses)
        == NUM_DECISION_LEVELS + 1
    )

    return ratio_losses


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def run_lm_loss_evaluation(
    model,
    tokenizer,
    batch_size: int = BATCH_SIZE,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, list[dict[str, Any]]]:
    """
    Compute per-sample LM loss across:

        hop
            -> sample
                -> document ratio
                    -> retrieval decision 0/32 ... 32/32

    Each CSV row stores:

        sample_id
        sample_index
        LM loss for every ratio / retrieval decision

    Notes
    -----
    Terminal output is intentionally kept compact so that
    large-scale evaluation does not flood the terminal.
    """

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1"
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        f"Device          : {device}"
    )

    print(
        f"Batch size      : {batch_size}"
    )

    print(
        f"Decision levels : "
        f"0/{NUM_DECISION_LEVELS} ... "
        f"{NUM_DECISION_LEVELS}/{NUM_DECISION_LEVELS}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print()

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print(
        "Loading validation dataset...",
        flush=True,
    )

    datasets = (
        load_teacher_generated_hypothesis_validation()
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print()
    print("Dataset:")
    
    total_samples = 0

    for hop, samples in datasets.items():

        print(
            f"  {hop:<8}: "
            f"{len(samples):,} samples"
        )

        total_samples += len(samples)

    print(
        f"  {'TOTAL':<8}: "
        f"{total_samples:,} samples"
    )

    # ========================================================
    # GLOBAL CSV SCHEMA
    # ========================================================
    #
    # Collect the union of all ratios appearing anywhere in
    # the validation set.
    #
    # ========================================================

    print()
    print(
        "Building global CSV schema...",
        flush=True,
    )

    all_ratios = []

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
    # STORE RESULTS
    # ========================================================

    all_hop_results = {}

    # ========================================================
    # GLOBAL PROGRESS
    # ========================================================

    processed_samples = 0

    # ========================================================
    # LOOP OVER HOPS
    # ========================================================

    for hop, samples in datasets.items():

        print()
        print("=" * 100)
        print(
            f"{hop.upper()} — "
            f"{len(samples):,} SAMPLES"
        )
        print("=" * 100)

        hop_results = []

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
            # Compact progress line
            # ------------------------------------------------

            print(
                f"[{processed_samples + 1:,}/"
                f"{total_samples:,}] "
                f"{hop} | "
                f"sample {sample_index + 1:,}/"
                f"{len(samples):,} | "
                f"id={sample_id}",
                flush=True,
            )

            # ------------------------------------------------
            # Create LM-loss samples
            # ------------------------------------------------

            lm_loss_samples = (
                create_lm_loss_samples(
                    sample
                )
            )

            # ------------------------------------------------
            # One CSV row
            # ------------------------------------------------

            row = {
                column: ""
                for column in all_columns
            }

            row["sample_id"] = sample_id

            row["sample_index"] = (
                sample_index
            )

            # =================================================
            # PROCESS EVERY RATIO
            # =================================================

            for ratio, lm_sample in (
                lm_loss_samples.items()
            ):

                # --------------------------------------------
                # Create 33 decision-level inputs
                # --------------------------------------------

                model_input = (
                    create_model_input_for_lm_loss(
                        lm_sample,
                        tokenizer,
                    )
                )

                # --------------------------------------------
                # Sanity check
                # --------------------------------------------

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

                # --------------------------------------------
                # Forward 33 levels
                # --------------------------------------------

                ratio_losses = (
                    _compute_ratio_losses(
                        model=model,
                        model_input=model_input,
                        device=device,
                        batch_size=batch_size,
                    )
                )

                # --------------------------------------------
                # Store losses
                # --------------------------------------------

                for decision_level, loss in enumerate(
                    ratio_losses
                ):

                    column = _make_column_name(
                        ratio,
                        decision_level,
                    )

                    row[column] = loss

                # --------------------------------------------
                # Cleanup ratio
                # --------------------------------------------

                del model_input
                del lm_sample

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

                    torch.cuda.ipc_collect()

            # =================================================
            # STORE ROW
            # =================================================

            hop_results.append(
                row
            )

            processed_samples += 1

            print(
                f"    ✓ complete | "
                f"id={sample_id}",
                flush=True,
            )

        # ====================================================
        # SAVE HOP CSV
        # ====================================================

        fieldnames = [
            "sample_id",
            "sample_index",
        ] + all_columns

        output_path = (
            output_dir
            / f"lm_loss_"
            f"{hop.replace('-', '_')}.csv"
        )

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

            writer.writerows(
                hop_results
            )

        all_hop_results[hop] = (
            hop_results
        )

        # ====================================================
        # HOP SUMMARY
        # ====================================================

        print()
        print(
            f"✓ {hop.upper()} COMPLETE"
        )

        print(
            f"  Samples : {len(hop_results):,}"
        )

        print(
            f"  Saved   : {output_path}"
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

    for hop in datasets:

        output_path = (
            output_dir
            / f"lm_loss_"
            f"{hop.replace('-', '_')}.csv"
        )

        print(
            f"{hop:<8} → {output_path}"
        )

    print()
    print("=" * 100)

    return all_hop_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

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
    print("STARTING LM-LOSS EVALUATION")
    print("=" * 80)

    results = run_lm_loss_evaluation(
        model=probe,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("ALL DONE")
    print("=" * 80)

    for hop in results:

        print(
            f"{hop:<8} → "
            f"{OUTPUT_DIR / f'lm_loss_{hop.replace('-', '_')}.csv'}"
        )