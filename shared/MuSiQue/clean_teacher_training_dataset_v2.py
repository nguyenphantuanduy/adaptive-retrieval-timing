
"""
Clean teacher-generated MuSiQue training dataset (v2).

Pipeline
--------
1. Load each hop configuration from Hugging Face.
2. Validate teacher_generation_output.
3. Detect missing / empty teacher outputs.
4. Detect missing final <answer>...</answer>.
5. Extract the LAST <answer>...</answer> block.
6. Compare the extracted answer with the ground-truth answer.
7. Count samples whose extracted answer is exactly:
       Insufficient evidence in the provided documents.
8. Collect ALL removal reasons for each invalid sample.
9. Export ALL removed samples (including original sample data)
   into a centralized JSON file.

Normal mode:
    - remove invalid samples
    - upload cleaned datasets to Hugging Face
    - export removed samples to JSON
    - export statistics to CSV

Dry-run mode:
    - perform all checks
    - export removed samples to JSON
    - export statistics to CSV
    - do NOT upload or modify the remote dataset

Run from repository root:

    .\\.venv\\Scripts\\python.exe ^
        shared\\MuSiQue\\data\\musique\\clean_teacher_training_dataset_v2.py

Dry run:

    .\\.venv\\Scripts\\python.exe ^
        shared\\MuSiQue\\data\\musique\\clean_teacher_training_dataset_v2.py ^
        --dry-run

Dataset
-------
    adaptive-retrieval-timing/MuSiQue-ART-Training-v2

Configs
-------
    2-hop
    3-hop
    4-hop

Output files
------------
    shared/MuSiQue/data/teacher_training_removed_samples_v2.json
    teacher_training_cleaning_statistics_v2.csv

Notes
-----
- All splits returned by each config are processed.
- Answer comparison uses lightweight string normalization only.
- This script does not evaluate reasoning/evidence quality.
- A sample is exported only ONCE, even if it has multiple removal reasons.
- Original sample data is preserved in the JSON export.
- JSON is overwritten on each execution to reflect the latest analysis.
- Normal mode uploads directly to the target v2 repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import re

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASET_REPO = "adaptive-retrieval-timing/MuSiQue-ART-Training-v2"

CONFIGS = (
    "2-hop",
    "3-hop",
    "4-hop",
)

# Relative to the repository root / current working directory.
STATS_CSV = Path("shared/MuSiQue/data/teacher_training_cleaning_statistics_v2.csv")

# Centralized JSON containing all removed samples.
REMOVED_SAMPLES_JSON = Path(
    "shared/MuSiQue/data/teacher_training_removed_samples_v2.json"
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "Insufficient evidence in the provided documents."
)


# ============================================================
# REMOVAL REASONS
# ============================================================

REMOVAL_REASONS = (
    "missing_teacher_output",
    "empty_teacher_output",
    "missing_final_answer",
    "answer_mismatch",
)


# ============================================================
# ANSWER NORMALIZATION
# ============================================================

def normalize_answer(answer: object) -> str:
    """
    Normalize an answer for robust comparison.

    This intentionally performs only lightweight normalization.
    It does NOT perform semantic matching.
    """
    if answer is None:
        return ""

    answer = str(answer)

    # Normalize whitespace.
    answer = re.sub(r"\s+", " ", answer).strip()

    # Lowercase.
    answer = answer.lower()

    # Remove surrounding common punctuation.
    answer = answer.strip(" \t\n\r.,;:!?\"'`")

    return answer


# ============================================================
# EXTRACT FINAL ANSWER
# ============================================================

def extract_final_answer(teacher_output: object) -> str | None:
    """
    Extract the final answer from teacher_generation_output.

    Expected format:

        <answer>
        something
        </answer>

    The function searches for the LAST occurrence of
    <answer>...</answer>.
    """
    if teacher_output is None:
        return None

    teacher_output = str(teacher_output).strip()

    if not teacher_output:
        return None

    matches = list(
        re.finditer(
            r"(?is)<answer>\s*(.*?)\s*</answer>",
            teacher_output,
        )
    )

    if not matches:
        return None

    answer = matches[-1].group(1).strip()

    # Remove common markdown formatting.
    answer = answer.strip("*_` ")

    if not answer:
        return None

    return answer


# ============================================================
# CHECK ONE SAMPLE
# ============================================================

def check_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Validate one training sample.

    Collects ALL applicable removal reasons instead of
    returning immediately after encountering the first issue.

    Returns
    -------
    dict
        Contains:
            - removal_reasons
            - missing_teacher_output
            - empty_teacher_output
            - missing_final_answer
            - answer_match
            - insufficient_evidence_answer
            - expected_answer
            - predicted_answer
    """

    expected_answer = sample.get("answer")
    teacher_output = sample.get("teacher_generation_output")

    removal_reasons: list[str] = []

    missing_teacher_output = False
    empty_teacher_output = False
    missing_final_answer = False

    answer_match = False
    insufficient_evidence_answer = False

    predicted_answer = None

    # --------------------------------------------------------
    # 1. Missing teacher output
    # --------------------------------------------------------

    if teacher_output is None:

        missing_teacher_output = True

        removal_reasons.append(
            "missing_teacher_output"
        )

    # --------------------------------------------------------
    # 2. Empty teacher output
    # --------------------------------------------------------

    elif not str(teacher_output).strip():

        empty_teacher_output = True

        removal_reasons.append(
            "empty_teacher_output"
        )

    # --------------------------------------------------------
    # 3. Extract final answer
    # --------------------------------------------------------

    if (
        missing_teacher_output
        or empty_teacher_output
    ):

        # No usable output means there cannot be a final answer.
        missing_final_answer = True

    else:

        predicted_answer = extract_final_answer(
            teacher_output
        )

        if predicted_answer is None:

            missing_final_answer = True

    # --------------------------------------------------------
    # 4. Missing final answer
    # --------------------------------------------------------

    if missing_final_answer:

        removal_reasons.append(
            "missing_final_answer"
        )

    # --------------------------------------------------------
    # 5. Answer validation
    # --------------------------------------------------------

    if predicted_answer is not None:

        # ----------------------------------------------------
        # Insufficient-evidence answer
        # ----------------------------------------------------

        insufficient_evidence_answer = (
            normalize_answer(predicted_answer)
            == normalize_answer(
                INSUFFICIENT_EVIDENCE_ANSWER
            )
        )

        # ----------------------------------------------------
        # Compare normalized answers
        # ----------------------------------------------------

        expected_normalized = normalize_answer(
            expected_answer
        )

        predicted_normalized = normalize_answer(
            predicted_answer
        )

        answer_match = (
            expected_normalized == predicted_normalized
        )

        if not answer_match:

            removal_reasons.append(
                "answer_mismatch"
            )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "removal_reasons": removal_reasons,

        "missing_teacher_output": missing_teacher_output,
        "empty_teacher_output": empty_teacher_output,
        "missing_final_answer": missing_final_answer,

        "answer_match": answer_match,

        "insufficient_evidence_answer": (
            insufficient_evidence_answer
        ),

        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
    }


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(
    dataset: Dataset,
    hop: str,
    config_name: str,
    split_name: str,
) -> dict[str, Any]:
    """
    Analyze every sample in one hop/split dataset.

    No samples are modified or removed here.

    Every invalid sample is stored with:
        - config
        - split
        - index
        - id
        - removal_reasons
        - expected_answer
        - predicted_answer
        - sample (original data)
    """

    print()
    print("=" * 80)
    print(f"ANALYZING {hop}")
    print("=" * 80)

    total = len(dataset)

    print(f"Total samples: {total:,}")

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    reason_counts = {
        reason: 0
        for reason in REMOVAL_REASONS
    }

    answer_match_count = 0
    answer_mismatch_count = 0

    insufficient_evidence_answer_count = 0

    # --------------------------------------------------------
    # Invalid samples
    # --------------------------------------------------------

    invalid_samples: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Iterate
    # --------------------------------------------------------

    for index, sample in enumerate(dataset):

        result = check_sample(sample)

        sample_id = sample.get(
            "id",
            f"<index={index}>"
        )

        reasons = result["removal_reasons"]

        # ----------------------------------------------------
        # Count insufficient-evidence answers
        # ----------------------------------------------------

        if result["insufficient_evidence_answer"]:

            insufficient_evidence_answer_count += 1

        # ----------------------------------------------------
        # Count answer matches / mismatches
        # ----------------------------------------------------

        if result["predicted_answer"] is not None:

            if result["answer_match"]:

                answer_match_count += 1

            else:

                answer_mismatch_count += 1

        # ----------------------------------------------------
        # Count ALL removal reasons
        # ----------------------------------------------------

        for reason in reasons:

            reason_counts[reason] += 1

        # ----------------------------------------------------
        # Store invalid sample
        # ----------------------------------------------------

        if reasons:

            invalid_samples.append(
                {
                    "config": config_name,
                    "split": split_name,
                    "index": index,
                    "id": sample_id,

                    "removal_reasons": reasons,

                    "expected_answer": (
                        result["expected_answer"]
                    ),

                    "predicted_answer": (
                        result["predicted_answer"]
                    ),

                    # Preserve the complete original sample.
                    "sample": dict(sample),
                }
            )

    # --------------------------------------------------------
    # Unique removed / valid samples
    # --------------------------------------------------------

    removed_samples = len(invalid_samples)

    valid_samples = total - removed_samples

    # --------------------------------------------------------
    # Match rate
    # --------------------------------------------------------

    valid_answer_samples = (
        answer_match_count
        + answer_mismatch_count
    )

    if valid_answer_samples > 0:

        match_rate = (
            answer_match_count
            / valid_answer_samples
            * 100
        )

    else:

        match_rate = 0.0

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print(f"{hop} SUMMARY")
    print("-" * 80)

    print(f"Total samples                       : {total:,}")

    print(
        f"Missing teacher output              : "
        f"{reason_counts['missing_teacher_output']:,}"
    )

    print(
        f"Empty teacher output                : "
        f"{reason_counts['empty_teacher_output']:,}"
    )

    print(
        f"Missing final <answer>              : "
        f"{reason_counts['missing_final_answer']:,}"
    )

    print(
        f"Answer matches                      : "
        f"{answer_match_count:,}"
    )

    print(
        f"Answer mismatches                   : "
        f"{reason_counts['answer_mismatch']:,}"
    )

    print(
        f"Insufficient-evidence answers       : "
        f"{insufficient_evidence_answer_count:,}"
    )

    print(
        f"Unique samples to remove            : "
        f"{removed_samples:,}"
    )

    print(
        f"Valid samples to keep                : "
        f"{valid_samples:,}"
    )

    print(
        f"Answer match rate                    : "
        f"{match_rate:.2f}%"
    )

    return {
        "hop": hop,

        "total": total,

        **reason_counts,

        "answer_matches": answer_match_count,
        "answer_mismatches": answer_mismatch_count,

        "insufficient_evidence_answers": (
            insufficient_evidence_answer_count
        ),

        "valid_samples": valid_samples,
        "removed_samples": removed_samples,

        "invalid_samples": invalid_samples,

        "match_rate": match_rate,
    }


# ============================================================
# CLEAN ONE CONFIG
# ============================================================

def clean_config(
    config_name: str,
    dry_run: bool,
) -> dict[str, Any]:

    print()
    print("=" * 80)
    print(f"LOADING CONFIG: {config_name}")
    print("=" * 80)

    # Load this config ONCE.
    source = load_dataset(
        DATASET_REPO,
        name=config_name,
    )

    cleaned = DatasetDict()

    split_results: dict[str, Any] = {}

    # ========================================================
    # PROCESS EVERY SPLIT
    # ========================================================

    for split_name, split_dataset in source.items():

        print()
        print(f"Processing split: {split_name}")

        # ----------------------------------------------------
        # Analyze original dataset
        # ----------------------------------------------------

        result = analyze_dataset(
            dataset=split_dataset,
            hop=f"{config_name}/{split_name}",
            config_name=config_name,
            split_name=split_name,
        )

        split_results[split_name] = result

        # ----------------------------------------------------
        # Dry run
        # ----------------------------------------------------

        if dry_run:

            print()
            print(
                f"DRY RUN: {config_name}/{split_name} "
                "was not modified."
            )

            print(
                f"Would remove samples                : "
                f"{result['removed_samples']:,}"
            )

            for reason in REMOVAL_REASONS:

                print(
                    f"  - {reason:<32}: "
                    f"{result[reason]:,}"
                )

            print(
                f"Insufficient-evidence answers       : "
                f"{result['insufficient_evidence_answers']:,}"
            )

            continue

        # ====================================================
        # REMOVE INVALID SAMPLES
        # ====================================================

        invalid_indices = {
            item["index"]
            for item in result["invalid_samples"]
        }

        valid_indices = [
            index
            for index in range(len(split_dataset))
            if index not in invalid_indices
        ]

        cleaned_split = split_dataset.select(
            valid_indices
        )

        removed_count = (
            len(split_dataset)
            - len(cleaned_split)
        )

        print()
        print(f"Removed invalid samples: {removed_count:,}")

        for reason in REMOVAL_REASONS:

            print(
                f"  - {reason:<32}: "
                f"{result[reason]:,}"
            )

        # ====================================================
        # VERIFY CLEANED DATASET
        # ====================================================

        for sample in cleaned_split:

            teacher_output = sample.get(
                "teacher_generation_output"
            )

            if (
                teacher_output is None
                or not str(teacher_output).strip()
            ):

                raise RuntimeError(
                    f"{config_name}/{split_name}: "
                    "cleaned dataset contains an empty "
                    "teacher_generation_output."
                )

            predicted_answer = extract_final_answer(
                teacher_output
            )

            if predicted_answer is None:

                raise RuntimeError(
                    f"{config_name}/{split_name}: "
                    "cleaned dataset contains a sample "
                    "without a final <answer>."
                )

            expected_answer = sample.get("answer")

            if normalize_answer(
                predicted_answer
            ) != normalize_answer(
                expected_answer
            ):

                raise RuntimeError(
                    f"{config_name}/{split_name}: "
                    "cleaned dataset contains an answer mismatch."
                )

        # ----------------------------------------------------
        # Store cleaned split
        # ----------------------------------------------------

        cleaned[split_name] = cleaned_split

        print()
        print(
            f"Cleaned samples remaining: "
            f"{len(cleaned_split):,}"
        )

    # ========================================================
    # DRY RUN / UPLOAD
    # ========================================================

    if dry_run:

        print()
        print(
            f"DRY RUN: {config_name} was not uploaded."
        )

    else:

        print()
        print(
            f"Uploading cleaned {config_name} "
            f"back to {DATASET_REPO}..."
        )

        cleaned.push_to_hub(
            DATASET_REPO,
            config_name=config_name,
            commit_message=(
                f"Clean teacher-generated training data "
                f"for {config_name} (v2)"
            ),
        )

        print(
            f"Uploaded cleaned {config_name}."
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "config": config_name,
        "split_results": split_results,
    }


# ============================================================
# SAVE REMOVED SAMPLES TO JSON
# ============================================================

def save_removed_samples_json(
    all_results: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    """
    Export all removed samples from all configs/splits.

    The JSON contains:
        - Metadata
        - All applicable removal reasons
        - Expected answer
        - Predicted answer
        - Complete original sample

    The file is overwritten on every execution.
    """

    removed_samples: list[dict[str, Any]] = []

    # ========================================================
    # COLLECT ALL REMOVED SAMPLES
    # ========================================================

    for config_result in all_results:

        for split_result in config_result[
            "split_results"
        ].values():

            removed_samples.extend(
                split_result["invalid_samples"]
            )

    # ========================================================
    # SORT FOR EASY REVIEW
    # ========================================================

    removed_samples.sort(
        key=lambda item: (
            item["config"],
            item["split"],
            item["index"],
        )
    )

    # ========================================================
    # GLOBAL REASON STATISTICS
    # ========================================================

    reason_counts = {
        reason: 0
        for reason in REMOVAL_REASONS
    }

    for item in removed_samples:

        for reason in item["removal_reasons"]:

            reason_counts[reason] += 1

    # ========================================================
    # BUILD JSON PAYLOAD
    # ========================================================

    payload = {
        "metadata": {
            "dataset": DATASET_REPO,

            "mode": (
                "dry-run"
                if dry_run
                else "clean-and-upload"
            ),

            "configs": list(CONFIGS),

            "total_removed_samples": len(
                removed_samples
            ),

            "removal_reason_counts": reason_counts,

            "note": (
                "Each sample is stored only once. "
                "removal_reasons contains all applicable "
                "removal reasons. The sample field preserves "
                "the original dataset record."
            ),
        },

        "removed_samples": removed_samples,
    }

    # ========================================================
    # WRITE JSON
    # ========================================================

    REMOVED_SAMPLES_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REMOVED_SAMPLES_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("REMOVED SAMPLES JSON")
    print("=" * 80)

    print(
        f"Total removed samples : "
        f"{len(removed_samples):,}"
    )

    print()
    print("Removal reason counts:")

    for reason, count in reason_counts.items():

        print(
            f"  - {reason:<32}: {count:,}"
        )

    print()
    print(
        f"Saved JSON to: {REMOVED_SAMPLES_JSON}"
    )


# ============================================================
# SAVE STATISTICS TO CSV
# ============================================================

def save_statistics_csv(
    all_results: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    """
    Save cleaning statistics to CSV.

    One row is written for every config/split,
    followed by one TOTAL row.

    Reason counts can overlap because a sample
    may have multiple removal reasons.

    removed_samples always counts unique samples.
    """

    csv_file = STATS_CSV

    csv_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "mode",
        "config",
        "split",
        "total_samples",
        "missing_teacher_output",
        "empty_teacher_output",
        "missing_final_answer",
        "answer_matches",
        "answer_mismatches",
        "answer_match_rate_percent",
        "insufficient_evidence_answers",
        "valid_samples",
        "removed_samples",
    ]

    rows: list[dict[str, Any]] = []

    # ========================================================
    # PER-CONFIG / PER-SPLIT ROWS
    # ========================================================

    for result in all_results:

        config_name = str(result["config"])

        for split_name, split_result in result[
            "split_results"
        ].items():

            rows.append(
                {
                    "mode": (
                        "dry-run"
                        if dry_run
                        else "clean-and-upload"
                    ),

                    "config": config_name,
                    "split": split_name,

                    "total_samples": int(
                        split_result["total"]
                    ),

                    "missing_teacher_output": int(
                        split_result["missing_teacher_output"]
                    ),

                    "empty_teacher_output": int(
                        split_result["empty_teacher_output"]
                    ),

                    "missing_final_answer": int(
                        split_result["missing_final_answer"]
                    ),

                    "answer_matches": int(
                        split_result["answer_matches"]
                    ),

                    "answer_mismatches": int(
                        split_result["answer_mismatches"]
                    ),

                    "answer_match_rate_percent": (
                        f"{float(split_result['match_rate']):.2f}"
                    ),

                    "insufficient_evidence_answers": int(
                        split_result[
                            "insufficient_evidence_answers"
                        ]
                    ),

                    "valid_samples": int(
                        split_result["valid_samples"]
                    ),

                    "removed_samples": int(
                        split_result["removed_samples"]
                    ),
                }
            )

    # ========================================================
    # TOTAL ROW
    # ========================================================

    sum_fields = (
        "total_samples",
        "missing_teacher_output",
        "empty_teacher_output",
        "missing_final_answer",
        "answer_matches",
        "answer_mismatches",
        "insufficient_evidence_answers",
        "valid_samples",
        "removed_samples",
    )

    totals = {
        field: sum(
            int(row[field])
            for row in rows
        )
        for field in sum_fields
    }

    total_valid_answer_samples = (
        totals["answer_matches"]
        + totals["answer_mismatches"]
    )

    if total_valid_answer_samples > 0:

        total_match_rate = (
            totals["answer_matches"]
            / total_valid_answer_samples
            * 100
        )

    else:

        total_match_rate = 0.0

    rows.append(
        {
            "mode": (
                "dry-run"
                if dry_run
                else "clean-and-upload"
            ),

            "config": "TOTAL",
            "split": "ALL",

            **totals,

            "answer_match_rate_percent": (
                f"{total_match_rate:.2f}"
            ),
        }
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    with csv_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("STATISTICS CSV")
    print("=" * 80)

    print(f"Saved statistics to: {csv_file}")
    print(f"Rows written      : {len(rows):,}")


# ============================================================
# PRINT PER-HOP TABLE
# ============================================================

def print_per_hop_table(
    all_results: list[dict[str, Any]],
) -> None:
    """Print the per-hop cleaning summary."""

    print()
    print("=" * 112)
    print("PER-HOP RESULT")
    print("=" * 112)
    print()

    print(
        f"{'Hop':<12}"
        f"{'Total':>10}"
        f"{'No Output':>12}"
        f"{'Empty':>10}"
        f"{'No Answer':>12}"
        f"{'Match':>10}"
        f"{'Mismatch':>12}"
        f"{'Insuff. Ev.':>14}"
        f"{'Keep':>10}"
    )

    print("-" * 112)

    for result in all_results:

        config_name = str(result["config"])

        for split_name, split_result in result[
            "split_results"
        ].items():

            if len(result["split_results"]) == 1:

                hop_label = config_name

            else:

                hop_label = (
                    f"{config_name}/{split_name}"
                )

            print(
                f"{hop_label:<12}"
                f"{int(split_result['total']):>10,}"
                f"{int(split_result['missing_teacher_output']):>12,}"
                f"{int(split_result['empty_teacher_output']):>10,}"
                f"{int(split_result['missing_final_answer']):>12,}"
                f"{int(split_result['answer_matches']):>10,}"
                f"{int(split_result['answer_mismatches']):>12,}"
                f"{int(split_result['insufficient_evidence_answers']):>14,}"
                f"{int(split_result['valid_samples']):>10,}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate and clean MuSiQue teacher-generated "
            "training data (v2)."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Analyze and report dataset statistics "
            "without modifying or uploading anything."
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CLEAN TEACHER TRAINING DATASET (V2)")
    print("=" * 80)

    print()
    print(f"Dataset : {DATASET_REPO}")

    print(
        f"Mode    : "
        f"{'DRY RUN' if args.dry_run else 'CLEAN + UPLOAD'}"
    )

    # ========================================================
    # PROCESS CONFIGS
    # ========================================================

    all_results: list[dict[str, Any]] = []

    for config_name in CONFIGS:

        result = clean_config(
            config_name=config_name,
            dry_run=args.dry_run,
        )

        all_results.append(result)

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    total_samples = 0

    total_answer_matches = 0
    total_answer_mismatches = 0

    total_insufficient_evidence_answers = 0

    total_valid_samples = 0

    total_reason_counts = {
        reason: 0
        for reason in REMOVAL_REASONS
    }

    for config_result in all_results:

        for split_result in config_result[
            "split_results"
        ].values():

            total_samples += int(
                split_result["total"]
            )

            total_answer_matches += int(
                split_result["answer_matches"]
            )

            total_answer_mismatches += int(
                split_result["answer_mismatches"]
            )

            total_insufficient_evidence_answers += int(
                split_result[
                    "insufficient_evidence_answers"
                ]
            )

            total_valid_samples += int(
                split_result["valid_samples"]
            )

            for reason in REMOVAL_REASONS:

                total_reason_counts[reason] += int(
                    split_result[reason]
                )

    # Unique removed samples.
    total_removed = (
        total_samples
        - total_valid_samples
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)

    if args.dry_run:

        print("DRY RUN — GLOBAL SUMMARY")

    else:

        print("CLEANING — GLOBAL SUMMARY")

    print("=" * 80)

    print()
    print(
        f"Total samples                       : "
        f"{total_samples:,}"
    )

    # --------------------------------------------------------
    # Removal statistics
    # --------------------------------------------------------

    print()
    print("REMOVED SAMPLES")
    print("-" * 80)

    print(
        f"Total unique samples removed        : "
        f"{total_removed:,}"
    )

    for reason in REMOVAL_REASONS:

        print(
            f"  - {reason:<32}: "
            f"{total_reason_counts[reason]:,}"
        )

    print()
    print(
        "Note: Removal reason counts may overlap "
        "because one sample can have multiple reasons."
    )

    # --------------------------------------------------------
    # Answer statistics
    # --------------------------------------------------------

    print()
    print("ANSWER VALIDATION")
    print("-" * 80)

    print(
        f"Answer matches                      : "
        f"{total_answer_matches:,}"
    )

    print(
        f"Answer mismatches                   : "
        f"{total_answer_mismatches:,}"
    )

    print(
        f"Insufficient-evidence answers       : "
        f"{total_insufficient_evidence_answers:,}"
    )

    valid_answer_samples = (
        total_answer_matches
        + total_answer_mismatches
    )

    if valid_answer_samples > 0:

        global_match_rate = (
            total_answer_matches
            / valid_answer_samples
            * 100
        )

    else:

        global_match_rate = 0.0

    print(
        f"Answer match rate                   : "
        f"{global_match_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Final dataset size
    # --------------------------------------------------------

    print()
    print("FINAL DATASET")
    print("-" * 80)

    print(
        f"Valid samples to keep                : "
        f"{total_valid_samples:,}"
    )

    # ========================================================
    # PER-HOP TABLE
    # ========================================================

    print_per_hop_table(
        all_results
    )

    # ========================================================
    # SAVE REMOVED SAMPLES JSON
    # ========================================================

    save_removed_samples_json(
        all_results=all_results,
        dry_run=args.dry_run,
    )

    # ========================================================
    # SAVE STATISTICS CSV
    # ========================================================

    save_statistics_csv(
        all_results=all_results,
        dry_run=args.dry_run,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 80)

    if args.dry_run:

        print("✓ DRY RUN COMPLETE")
        print()

        print("No remote dataset was modified.")

        print()
        print("Generated files:")

        print(
            f"  - Removed samples JSON: "
            f"{REMOVED_SAMPLES_JSON}"
        )

        print(
            f"  - Statistics CSV: "
            f"{STATS_CSV}"
        )

        print()
        print("The JSON contains:")
        print("  - All samples that would be removed")
        print("  - All applicable removal reasons")
        print("  - Original sample data")
        print("  - Config, split and sample index")

    else:

        print("✓ DATASET CLEANING COMPLETE")
        print()

        print(
            f"Removed unique samples              : "
            f"{total_removed:,}"
        )

        print(
            f"Remaining valid samples             : "
            f"{total_valid_samples:,}"
        )

        print()
        print("Uploaded cleaned dataset to:")
        print(f"  {DATASET_REPO}")

        print()
        print("Exported removed samples to:")
        print(f"  {REMOVED_SAMPLES_JSON}")

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
