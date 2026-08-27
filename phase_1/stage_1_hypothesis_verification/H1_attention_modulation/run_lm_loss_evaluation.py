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
BATCH_SIZE_BY_HOP = {
    "2-hop": 4,
    "3-hop": 1,
    "4-hop": 1,
}

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
# MAIN EVALUATION FUNCTION
# ============================================================

def run_lm_loss_evaluation(
    model,
    tokenizer,
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

    Batch size is selected automatically by hop:

        2-hop -> 4
        3-hop -> 1
        4-hop -> 1

    Notes
    -----
    Terminal output is intentionally compact for large-scale
    evaluation.
    """

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
        "Batch size by hop:"
    )

    for hop, batch_size in BATCH_SIZE_BY_HOP.items():

        print(
            f"  {hop:<8}: "
            f"{batch_size}"
        )

    print(
        f"Decision levels : "
        f"0/{NUM_DECISION_LEVELS} ... "
        f"{NUM_DECISION_LEVELS}/{NUM_DECISION_LEVELS}"
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

            # ------------------------------------------------
            # Verify sample ID
            # ------------------------------------------------

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
    # Deterministic ratio ordering
    # --------------------------------------------------------

    all_ratios = sorted(
        all_ratios,
        key=lambda x: (
            -x[0],
            x[1],
        ),
    )

    # --------------------------------------------------------
    # CSV loss columns
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Select batch size for this hop
        # ----------------------------------------------------

        batch_size = _get_batch_size_for_hop(
            hop
        )

        print()
        print("=" * 100)
        print(
            f"{hop.upper()} — "
            f"{len(samples):,} SAMPLES "
            f"| BATCH SIZE = {batch_size}"
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
            # Compact progress
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

            # ------------------------------------------------
            # Sample identification
            # ------------------------------------------------

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
                # Forward all 33 decision levels
                #
                # Batch size depends on hop:
                #
                # 2-hop -> 4
                # 3-hop -> 1
                # 4-hop -> 1
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

            # ------------------------------------------------
            # Completion progress
            # ------------------------------------------------

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
            f"  Samples    : "
            f"{len(hop_results):,}"
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

    for hop in datasets:

        output_path = (
            output_dir
            / f"lm_loss_"
            f"{hop.replace('-', '_')}.csv"
        )

        print(
            f"{hop:<8} → "
            f"{output_path}"
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