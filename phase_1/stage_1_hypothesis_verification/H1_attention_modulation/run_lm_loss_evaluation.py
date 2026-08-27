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

    The column contains the LM loss for that condition.
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

    The full 33-sample model input stays on CPU. It is moved to
    GPU in smaller batches to avoid CUDA OOM.

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
        # CPU -> small GPU batch
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
        # Move only scalar results back to CPU
        # ----------------------------------------------------

        ratio_losses.extend(
            per_sample_loss
            .detach()
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Explicit cleanup
        # ----------------------------------------------------

        del outputs
        del per_sample_loss
        del batch_input

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert (
        len(ratio_losses) == NUM_DECISION_LEVELS + 1
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

    Parameters
    ----------
    model:
        ART/Qwen model already loaded on GPU.

    tokenizer:
        Tokenizer corresponding to the model.

    batch_size:
        GPU forward batch size.
        Defaults to BATCH_SIZE.

    output_dir:
        Directory where the three CSV files are written.

    Returns
    -------
    dict
        Results grouped by hop.
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
    # LOAD DATASET
    # ========================================================

    print("=" * 100)
    print("LM LOSS EVALUATION")
    print("=" * 100)

    print()
    print(
        f"Device      : {device}"
    )

    print(
        f"Batch size  : {batch_size}"
    )

    print(
        f"Decision levels: "
        f"0/{NUM_DECISION_LEVELS} ... "
        f"{NUM_DECISION_LEVELS}/{NUM_DECISION_LEVELS}"
    )

    print(
        f"Output dir  : {output_dir}"
    )

    print()

    datasets = (
        load_teacher_generated_hypothesis_validation()
    )

    # ========================================================
    # GLOBAL CSV SCHEMA
    # ========================================================
    #
    # 2-hop may not contain (3, 0), while 3-hop / 4-hop may.
    #
    # We therefore use the union of all ratios so every CSV has
    # the same flattened column structure.
    #
    # ========================================================

    all_ratios = []

    for samples in datasets.values():

        for sample in samples:

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
    # Deterministic ratio ordering
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
        f"CSV loss columns: "
        f"{len(all_columns)}"
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_hop_results = {}

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

            print()
            print(
                "-" * 100
            )
            print(
                f"{hop} — "
                f"Sample "
                f"{sample_index + 1:,}/"
                f"{len(samples):,}"
            )
            print(
                f"Question: "
                f"{sample['question']}"
            )
            print(
                "-" * 100
            )

            # ------------------------------------------------
            # Create LM-loss samples for all ratios
            # ------------------------------------------------

            lm_loss_samples = (
                create_lm_loss_samples(
                    sample
                )
            )

            # ------------------------------------------------
            # One flat row for this sample
            # ------------------------------------------------

            row = {
                column: ""
                for column in all_columns
            }

            row["sample_index"] = (
                sample_index
            )

            # ------------------------------------------------
            # Process every ratio
            # ------------------------------------------------

            for ratio, lm_sample in (
                lm_loss_samples.items()
            ):

                print(
                    f"  Ratio "
                    f"{ratio[0]}:{ratio[1]}"
                )

                # --------------------------------------------
                # Create 33 retrieval-decision inputs
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
                    f"{num_decisions} decision "
                    f"levels instead of "
                    f"{NUM_DECISION_LEVELS + 1}"
                )

                # --------------------------------------------
                # Forward 33 levels in small GPU batches
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
                # Flatten into CSV columns
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
                # Free ratio input
                # --------------------------------------------

                del model_input
                del lm_sample

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # ------------------------------------------------
            # Store sample row
            # ------------------------------------------------

            hop_results.append(
                row
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"  ✓ Sample "
                f"{sample_index + 1:,} complete"
            )

        # ====================================================
        # SAVE HOP CSV
        # ====================================================

        fieldnames = [
            "sample_index"
        ] + all_columns

        output_path = (
            output_dir
            / f"lm_loss_{hop.replace('-', '_')}.csv"
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

        print()
        print(
            f"✓ Saved: "
            f"{output_path}"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 100)
    print("LM LOSS EVALUATION COMPLETE")
    print("=" * 100)

    for hop in datasets:

        output_path = (
            output_dir
            / f"lm_loss_{hop.replace('-', '_')}.csv"
        )

        print(
            f"{hop:<8} → "
            f"{output_path}"
        )

    return all_hop_results


# ============================================================
# OPTIONAL MAIN
# ============================================================
#
# The function expects `model` and `tokenizer` to already exist.
#
# If this file is imported from a notebook:
#
#     from run_lm_loss_evaluation import (
#         run_lm_loss_evaluation,
#     )
#
#     results = run_lm_loss_evaluation(
#         model=probe,
#         tokenizer=tokenizer,
#         batch_size=8,
#     )
#
# ============================================================

if __name__ == "__main__":

    import torch
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
    # LOAD TOKENIZER
    # ========================================================

    print("=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    print()
    print(f"Tokenizer  : {MODEL_NAME}")

    # ========================================================
    # LOAD MODEL — SINGLE GPU
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
    # CHECK
    # ========================================================

    print()
    print("=" * 80)
    print("MODEL LOADED")
    print("=" * 80)

    print()
    print(f"Model           : {HF_REPO}")
    print(
        f"Parameter device : "
        f"{next(probe.parameters()).device}"
    )

    # ========================================================
    # RUN LM-LOSS EVALUATION
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