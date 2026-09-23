"""
Clean teacher-generated MuSiQue training dataset.

Pipeline
--------
1. Load each hop configuration from Hugging Face.
2. Validate teacher_generation_output.
3. Detect missing / empty teacher outputs.
4. Detect missing final <answer>...</answer>.
5. Compare extracted answer with ground-truth answer.
6. Detect verbatim question prefixes.
7. In normal mode:
      - remove samples without a valid final answer
      - remove samples with answer mismatch
      - remove verbatim question prefixes from retained samples
      - upload cleaned dataset back to Hugging Face
8. In --dry-run mode:
      - perform all checks
      - print statistics
      - do NOT upload or modify the remote dataset.

Run from repository root:

    .\\.venv\\Scripts\\python.exe ^
        shared\\MuSiQue\\data\\musique\\clean_teacher_training_dataset.py

Dry run:

    .\\.venv\\Scripts\\python.exe ^
        shared\\MuSiQue\\data\\musique\\clean_teacher_training_dataset.py ^
        --dry-run

Dataset
-------
    adaptive-retrieval-timing/MuSiQue-ART-Training

Configs
-------
    2-hop
    3-hop
    4-hop

Split
-----
    train

Statistics
----------
    data/musique/teacher_training_cleaning_statistics.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Mapping
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASET_REPO = (
    "adaptive-retrieval-timing/"
    "MuSiQue-ART-Training"
)

CONFIGS = (
    "2-hop",
    "3-hop",
    "4-hop",
)

# Statistics CSV output.
# Relative to the repository root / current working directory.
STATS_CSV = Path(
    "teacher_training_cleaning_statistics.csv"
)


# ============================================================
# ANSWER NORMALIZATION
# ============================================================

def normalize_answer(
    answer: object,
) -> str:
    """
    Normalize an answer for robust comparison.

    This intentionally performs only lightweight normalization.
    It does NOT perform semantic matching.
    """

    if answer is None:
        return ""

    answer = str(answer)

    # Normalize whitespace.
    answer = re.sub(
        r"\s+",
        " ",
        answer,
    ).strip()

    # Lowercase.
    answer = answer.lower()

    # Remove surrounding common punctuation.
    answer = answer.strip(
        " \t\n\r.,;:!?\"'`"
    )

    return answer


# ============================================================
# EXTRACT FINAL ANSWER
# ============================================================

def extract_final_answer(
    teacher_output: object,
) -> str | None:
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

    teacher_output = str(
        teacher_output
    ).strip()

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
    answer = answer.strip(
        "*_` "
    )

    if not answer:
        return None

    return answer


# ============================================================
# QUESTION PREFIX DETECTION
# ============================================================

def has_verbatim_question_prefix(
    sample: Mapping[str, object],
) -> bool:
    """
    Return whether teacher_generation_output starts with
    the exact question.

    Leading whitespace before the question is ignored.
    """

    question = sample.get(
        "question"
    )

    teacher_output = sample.get(
        "teacher_generation_output"
    )

    return (
        isinstance(question, str)
        and bool(question)
        and isinstance(teacher_output, str)
        and teacher_output.lstrip().startswith(
            question
        )
    )


# ============================================================
# REMOVE QUESTION PREFIX
# ============================================================

def remove_verbatim_question_prefix(
    sample: Mapping[str, object],
) -> dict[str, object]:
    """
    Remove only an exact question prefix.

    No other part of teacher_generation_output
    is modified.
    """

    teacher_output = sample.get(
        "teacher_generation_output"
    )

    if not has_verbatim_question_prefix(
        sample
    ):
        return {
            "teacher_generation_output": (
                teacher_output
            )
        }

    question = sample["question"]

    assert isinstance(
        question,
        str,
    )

    assert isinstance(
        teacher_output,
        str,
    )

    normalized_output = (
        teacher_output
        .lstrip()
        [len(question):]
        .lstrip()
    )

    return {
        "teacher_generation_output": (
            normalized_output
        )
    }


# ============================================================
# CHECK ONE SAMPLE
# ============================================================

def check_sample(
    sample: Mapping[str, object],
) -> dict[str, object]:
    """
    Validate one training sample.

    Returns
    -------
    dict
        Contains:
            - missing_teacher_output
            - empty_teacher_output
            - missing_final_answer
            - answer_match
            - question_prefix
            - expected_answer
            - predicted_answer
    """

    expected_answer = sample.get(
        "answer"
    )

    teacher_output = sample.get(
        "teacher_generation_output"
    )

    # --------------------------------------------------------
    # Missing teacher output
    # --------------------------------------------------------

    if teacher_output is None:

        return {
            "missing_teacher_output": True,
            "empty_teacher_output": False,
            "missing_final_answer": True,
            "answer_match": False,
            "question_prefix": False,
            "expected_answer": expected_answer,
            "predicted_answer": None,
        }

    # --------------------------------------------------------
    # Empty teacher output
    # --------------------------------------------------------

    if not str(
        teacher_output
    ).strip():

        return {
            "missing_teacher_output": False,
            "empty_teacher_output": True,
            "missing_final_answer": True,
            "answer_match": False,
            "question_prefix": False,
            "expected_answer": expected_answer,
            "predicted_answer": None,
        }

    # --------------------------------------------------------
    # Question prefix
    # --------------------------------------------------------

    question_prefix = (
        has_verbatim_question_prefix(
            sample
        )
    )

    # --------------------------------------------------------
    # Extract final answer
    # --------------------------------------------------------

    predicted_answer = (
        extract_final_answer(
            teacher_output
        )
    )

    if predicted_answer is None:

        return {
            "missing_teacher_output": False,
            "empty_teacher_output": False,
            "missing_final_answer": True,
            "answer_match": False,
            "question_prefix": question_prefix,
            "expected_answer": expected_answer,
            "predicted_answer": None,
        }

    # --------------------------------------------------------
    # Compare normalized answers
    # --------------------------------------------------------

    expected_normalized = (
        normalize_answer(
            expected_answer
        )
    )

    predicted_normalized = (
        normalize_answer(
            predicted_answer
        )
    )

    answer_match = (
        expected_normalized
        == predicted_normalized
    )

    return {
        "missing_teacher_output": False,
        "empty_teacher_output": False,
        "missing_final_answer": False,
        "answer_match": answer_match,
        "question_prefix": question_prefix,
        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
    }


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(
    dataset: Dataset,
    hop: str,
) -> dict[str, object]:
    """
    Analyze every sample in one hop/split dataset.

    No samples are modified or removed here.
    """

    print()
    print("=" * 80)
    print(
        f"ANALYZING {hop}"
    )
    print("=" * 80)

    total = len(dataset)

    print(
        f"Total samples: {total:,}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    missing_teacher_output_count = 0
    empty_teacher_output_count = 0
    missing_final_answer_count = 0
    answer_match_count = 0
    answer_mismatch_count = 0
    question_prefix_count = 0

    # --------------------------------------------------------
    # Invalid samples
    # --------------------------------------------------------

    invalid_samples = []

    question_prefix_samples = []

    # Number of prefixes that belong to samples which
    # survive validation and therefore WOULD be cleaned.
    prefixes_to_clean = 0

    # --------------------------------------------------------
    # Iterate
    # --------------------------------------------------------

    for index, sample in enumerate(
        dataset
    ):

        result = check_sample(
            sample
        )

        sample_id = sample.get(
            "id",
            f"<index={index}>",
        )

        # ----------------------------------------------------
        # Question prefix
        # ----------------------------------------------------

        if result[
            "question_prefix"
        ]:

            question_prefix_count += 1

            question_prefix_samples.append(
                {
                    "index": index,
                    "id": sample_id,
                }
            )

        # ----------------------------------------------------
        # Missing teacher output
        # ----------------------------------------------------

        if result[
            "missing_teacher_output"
        ]:

            missing_teacher_output_count += 1

            invalid_samples.append(
                {
                    "index": index,
                    "id": sample_id,
                    "reason": "missing_teacher_output",
                }
            )

            continue

        # ----------------------------------------------------
        # Empty teacher output
        # ----------------------------------------------------

        if result[
            "empty_teacher_output"
        ]:

            empty_teacher_output_count += 1

            invalid_samples.append(
                {
                    "index": index,
                    "id": sample_id,
                    "reason": "empty_teacher_output",
                }
            )

            continue

        # ----------------------------------------------------
        # Missing final answer
        # ----------------------------------------------------

        if result[
            "missing_final_answer"
        ]:

            missing_final_answer_count += 1

            invalid_samples.append(
                {
                    "index": index,
                    "id": sample_id,
                    "reason": "missing_final_answer",
                    "expected_answer": result[
                        "expected_answer"
                    ],
                    "teacher_output": sample.get(
                        "teacher_generation_output",
                        "",
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Answer match / mismatch
        # ----------------------------------------------------

        if result[
            "answer_match"
        ]:

            answer_match_count += 1

            # Prefix is cleaned only if the sample survives.
            if result[
                "question_prefix"
            ]:
                prefixes_to_clean += 1

        else:

            answer_mismatch_count += 1

            invalid_samples.append(
                {
                    "index": index,
                    "id": sample_id,
                    "reason": "answer_mismatch",
                    "expected_answer": result[
                        "expected_answer"
                    ],
                    "predicted_answer": result[
                        "predicted_answer"
                    ],
                    "teacher_output": sample.get(
                        "teacher_generation_output",
                        "",
                    ),
                }
            )

    # --------------------------------------------------------
    # Valid samples
    # --------------------------------------------------------

    valid_samples = (
        total
        - missing_teacher_output_count
        - empty_teacher_output_count
        - missing_final_answer_count
        - answer_mismatch_count
    )

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
    print(
        f"{hop} SUMMARY"
    )
    print("-" * 80)

    print(
        f"Total samples              : "
        f"{total:,}"
    )

    print(
        f"Missing teacher output     : "
        f"{missing_teacher_output_count:,}"
    )

    print(
        f"Empty teacher output       : "
        f"{empty_teacher_output_count:,}"
    )

    print(
        f"Missing final <answer>     : "
        f"{missing_final_answer_count:,}"
    )

    print(
        f"Answer matches             : "
        f"{answer_match_count:,}"
    )

    print(
        f"Answer mismatches          : "
        f"{answer_mismatch_count:,}"
    )

    print(
        f"Question prefixes found    : "
        f"{question_prefix_count:,}"
    )

    print(
        f"Question prefixes to clean : "
        f"{prefixes_to_clean:,}"
    )

    print(
        f"Valid samples to keep      : "
        f"{valid_samples:,}"
    )

    print(
        f"Answer match rate          : "
        f"{match_rate:.2f}%"
    )

    return {
        "hop": hop,
        "total": total,

        "missing_teacher_output": (
            missing_teacher_output_count
        ),
        "empty_teacher_output": (
            empty_teacher_output_count
        ),
        "missing_final_answer": (
            missing_final_answer_count
        ),

        "answer_matches": (
            answer_match_count
        ),
        "answer_mismatches": (
            answer_mismatch_count
        ),

        "question_prefixes": (
            question_prefix_count
        ),
        "prefixes_to_clean": (
            prefixes_to_clean
        ),

        "valid_samples": (
            valid_samples
        ),

        "invalid_samples": (
            invalid_samples
        ),

        "question_prefix_samples": (
            question_prefix_samples
        ),

        "match_rate": match_rate,
    }


# ============================================================
# CLEAN ONE CONFIG
# ============================================================

def clean_config(
    config_name: str,
    dry_run: bool,
) -> dict[str, object]:

    print()
    print("=" * 80)
    print(
        f"LOADING CONFIG: {config_name}"
    )
    print("=" * 80)

    # Load this config ONCE.
    source = load_dataset(
        DATASET_REPO,
        name=config_name,
    )

    cleaned = DatasetDict()

    # --------------------------------------------------------
    # Config-level statistics
    # --------------------------------------------------------

    total_removed = 0

    removed_missing_teacher_output = 0
    removed_empty_teacher_output = 0
    removed_missing_final_answer = 0
    removed_answer_mismatch = 0

    total_prefixes_before = 0
    total_prefixes_cleaned = 0
    total_prefixes_after = 0

    split_results = {}

    # ========================================================
    # PROCESS EVERY SPLIT
    # ========================================================

    for split_name, split_dataset in (
        source.items()
    ):

        print()
        print(
            f"Processing split: {split_name}"
        )

        # ----------------------------------------------------
        # Analyze original dataset.
        #
        # This is the ONLY full scan required for validation.
        # ----------------------------------------------------

        result = analyze_dataset(
            dataset=split_dataset,
            hop=f"{config_name}/{split_name}",
        )

        split_results[split_name] = result

        total_prefixes_before += int(
            result["question_prefixes"]
        )

        # Number of prefixes on samples that survive.
        prefixes_to_clean = int(
            result["prefixes_to_clean"]
        )

        total_prefixes_cleaned += (
            prefixes_to_clean
        )

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
                f"Would remove samples          : "
                f"{len(result['invalid_samples']):,}"
            )

            print(
                f"  - Missing teacher output    : "
                f"{result['missing_teacher_output']:,}"
            )

            print(
                f"  - Empty teacher output      : "
                f"{result['empty_teacher_output']:,}"
            )

            print(
                f"  - Missing final <answer>    : "
                f"{result['missing_final_answer']:,}"
            )

            print(
                f"  - Answer mismatch           : "
                f"{result['answer_mismatches']:,}"
            )

            print(
                f"Question prefixes found      : "
                f"{result['question_prefixes']:,}"
            )

            print(
                f"Question prefixes to clean   : "
                f"{prefixes_to_clean:,}"
            )

            # If cleaning were performed, all prefixes on
            # retained samples would be removed.
            #
            # Prefixes on invalid samples disappear together
            # with those samples, so the projected final
            # prefix count is also zero.
            continue

        # ====================================================
        # REMOVE INVALID SAMPLES
        # ====================================================

        invalid_indices = {
            item["index"]
            for item in result[
                "invalid_samples"
            ]
        }

        valid_indices = [
            index
            for index in range(
                len(split_dataset)
            )
            if index not in invalid_indices
        ]

        cleaned_split = (
            split_dataset.select(
                valid_indices
            )
        )

        removed_count = (
            len(split_dataset)
            - len(cleaned_split)
        )

        total_removed += removed_count

        # ----------------------------------------------------
        # Count removal reasons
        # ----------------------------------------------------

        removed_missing_teacher_output += int(
            result[
                "missing_teacher_output"
            ]
        )

        removed_empty_teacher_output += int(
            result[
                "empty_teacher_output"
            ]
        )

        removed_missing_final_answer += int(
            result[
                "missing_final_answer"
            ]
        )

        removed_answer_mismatch += int(
            result[
                "answer_mismatches"
            ]
        )

        print()
        print(
            f"Removed invalid samples: "
            f"{removed_count:,}"
        )

        print(
            f"  - Missing teacher output: "
            f"{result['missing_teacher_output']:,}"
        )

        print(
            f"  - Empty teacher output: "
            f"{result['empty_teacher_output']:,}"
        )

        print(
            f"  - Missing final <answer>: "
            f"{result['missing_final_answer']:,}"
        )

        print(
            f"  - Answer mismatch: "
            f"{result['answer_mismatches']:,}"
        )

        # ====================================================
        # REMOVE QUESTION PREFIX
        # ====================================================

        print()
        print(
            f"Question prefixes found: "
            f"{result['question_prefixes']:,}"
        )

        print(
            f"Question prefixes cleaned: "
            f"{prefixes_to_clean:,}"
        )

        cleaned_split = cleaned_split.map(
            remove_verbatim_question_prefix
        )

        # ====================================================
        # VERIFY NO PREFIX REMAINS
        # ====================================================

        prefixes_after = sum(
            has_verbatim_question_prefix(
                sample
            )
            for sample in cleaned_split
        )

        total_prefixes_after += (
            prefixes_after
        )

        if prefixes_after:

            raise RuntimeError(
                f"{config_name}/{split_name}: "
                f"{prefixes_after} repeated-question "
                "prefixes remain after normalization."
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
                or not str(
                    teacher_output
                ).strip()
            ):

                raise RuntimeError(
                    f"{config_name}/{split_name}: "
                    "cleaned dataset contains an empty "
                    "teacher_generation_output."
                )

            predicted_answer = (
                extract_final_answer(
                    teacher_output
                )
            )

            if predicted_answer is None:

                raise RuntimeError(
                    f"{config_name}/{split_name}: "
                    "cleaned dataset contains a sample "
                    "without a final <answer>."
                )

            expected_answer = sample.get(
                "answer"
            )

            if (
                normalize_answer(
                    predicted_answer
                )
                != normalize_answer(
                    expected_answer
                )
            ):

                raise RuntimeError(
                    f"{config_name}/{split_name}: "
                    "cleaned dataset contains an "
                    "answer mismatch."
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

        print(
            f"Question prefixes remaining: "
            f"{prefixes_after:,}"
        )

    # ========================================================
    # DRY RUN / UPLOAD
    # ========================================================

    if dry_run:

        print()
        print(
            f"DRY RUN: {config_name} was not uploaded."
        )

        # In a hypothetical cleaned dataset:
        # every prefix would have been removed.
        total_prefixes_after = 0

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
                f"Clean teacher-generated "
                f"training data for {config_name}"
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

        # Removed samples.
        "removed": total_removed,

        "removed_missing_teacher_output": (
            removed_missing_teacher_output
        ),

        "removed_empty_teacher_output": (
            removed_empty_teacher_output
        ),

        "removed_missing_final_answer": (
            removed_missing_final_answer
        ),

        "removed_answer_mismatch": (
            removed_answer_mismatch
        ),

        # Prefix statistics.
        "prefixes_before": (
            total_prefixes_before
        ),

        "prefixes_cleaned": (
            total_prefixes_cleaned
        ),

        "prefixes_after": (
            total_prefixes_after
        ),
    }


# ============================================================
# SAVE STATISTICS TO CSV
# ============================================================

def save_statistics_csv(
    all_results: list[dict[str, object]],
    dry_run: bool,
) -> None:
    """
    Save cleaning statistics to CSV.

    One row is written for every config/split, followed by
    one TOTAL row.

    The function does NOT reload the Hugging Face dataset.
    All statistics come from the analysis already performed.
    """

    csv_file = STATS_CSV

    # Create parent directory if necessary.
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
        "question_prefixes_found",
        "question_prefixes_cleaned",
        "question_prefixes_remaining",
        "valid_samples",
        "removed_samples",
    ]

    rows = []

    # ========================================================
    # PER-CONFIG / PER-SPLIT ROWS
    # ========================================================

    for result in all_results:

        config_name = str(
            result["config"]
        )

        for split_name, split_result in (
            result["split_results"].items()
        ):

            total = int(
                split_result["total"]
            )

            valid = int(
                split_result["valid_samples"]
            )

            removed = (
                total
                - valid
            )

            prefixes_found = int(
                split_result[
                    "question_prefixes"
                ]
            )

            prefixes_cleaned = int(
                split_result[
                    "prefixes_to_clean"
                ]
            )

            # After cleaning, all prefixes on retained
            # samples should be gone.
            prefixes_remaining = 0

            rows.append(
                {
                    "mode": (
                        "dry-run"
                        if dry_run
                        else "clean-and-upload"
                    ),
                    "config": config_name,
                    "split": split_name,

                    "total_samples": total,

                    "missing_teacher_output": int(
                        split_result[
                            "missing_teacher_output"
                        ]
                    ),

                    "empty_teacher_output": int(
                        split_result[
                            "empty_teacher_output"
                        ]
                    ),

                    "missing_final_answer": int(
                        split_result[
                            "missing_final_answer"
                        ]
                    ),

                    "answer_matches": int(
                        split_result[
                            "answer_matches"
                        ]
                    ),

                    "answer_mismatches": int(
                        split_result[
                            "answer_mismatches"
                        ]
                    ),

                    "answer_match_rate_percent": (
                        f"{float(split_result['match_rate']):.2f}"
                    ),

                    "question_prefixes_found": (
                        prefixes_found
                    ),

                    "question_prefixes_cleaned": (
                        prefixes_cleaned
                    ),

                    "question_prefixes_remaining": (
                        prefixes_remaining
                    ),

                    "valid_samples": valid,

                    "removed_samples": removed,
                }
            )

    # ========================================================
    # TOTAL ROW
    # ========================================================

    total_samples = sum(
        int(row["total_samples"])
        for row in rows
    )

    total_missing_teacher_output = sum(
        int(row["missing_teacher_output"])
        for row in rows
    )

    total_empty_teacher_output = sum(
        int(row["empty_teacher_output"])
        for row in rows
    )

    total_missing_final_answer = sum(
        int(row["missing_final_answer"])
        for row in rows
    )

    total_answer_matches = sum(
        int(row["answer_matches"])
        for row in rows
    )

    total_answer_mismatches = sum(
        int(row["answer_mismatches"])
        for row in rows
    )

    total_prefixes_found = sum(
        int(row["question_prefixes_found"])
        for row in rows
    )

    total_prefixes_cleaned = sum(
        int(row["question_prefixes_cleaned"])
        for row in rows
    )

    total_valid_samples = sum(
        int(row["valid_samples"])
        for row in rows
    )

    total_removed_samples = sum(
        int(row["removed_samples"])
        for row in rows
    )

    total_valid_answer_samples = (
        total_answer_matches
        + total_answer_mismatches
    )

    if total_valid_answer_samples > 0:

        total_match_rate = (
            total_answer_matches
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

            "total_samples": total_samples,

            "missing_teacher_output": (
                total_missing_teacher_output
            ),

            "empty_teacher_output": (
                total_empty_teacher_output
            ),

            "missing_final_answer": (
                total_missing_final_answer
            ),

            "answer_matches": (
                total_answer_matches
            ),

            "answer_mismatches": (
                total_answer_mismatches
            ),

            "answer_match_rate_percent": (
                f"{total_match_rate:.2f}"
            ),

            "question_prefixes_found": (
                total_prefixes_found
            ),

            "question_prefixes_cleaned": (
                total_prefixes_cleaned
            ),

            "question_prefixes_remaining": 0,

            "valid_samples": (
                total_valid_samples
            ),

            "removed_samples": (
                total_removed_samples
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

    print(
        f"Saved statistics to: {csv_file}"
    )

    print(
        f"Rows written      : {len(rows):,}"
    )


# ============================================================
# PRINT PER-HOP TABLE
# ============================================================

def print_per_hop_table(
    all_results: list[dict[str, object]],
) -> None:
    """
    Print the per-hop cleaning summary.

    No dataset is reloaded here.
    """

    print()
    print("=" * 80)
    print(
        "PER-HOP RESULT"
    )
    print("=" * 80)

    print()

    print(
        f"{'Hop':<8}"
        f"{'Total':>10}"
        f"{'No Output':>12}"
        f"{'Empty':>10}"
        f"{'No Answer':>12}"
        f"{'Match':>10}"
        f"{'Mismatch':>12}"
        f"{'Prefix':>10}"
        f"{'Cleaned':>10}"
        f"{'Keep':>10}"
    )

    print("-" * 114)

    for result in all_results:

        config_name = str(
            result["config"]
        )

        for split_name, split_result in (
            result["split_results"].items()
        ):

            if len(
                result["split_results"]
            ) == 1:

                hop_label = config_name

            else:

                hop_label = (
                    f"{config_name}/"
                    f"{split_name}"
                )

            print(
                f"{hop_label:<8}"
                f"{int(split_result['total']):>10,}"
                f"{int(split_result['missing_teacher_output']):>12,}"
                f"{int(split_result['empty_teacher_output']):>10,}"
                f"{int(split_result['missing_final_answer']):>12,}"
                f"{int(split_result['answer_matches']):>10,}"
                f"{int(split_result['answer_mismatches']):>12,}"
                f"{int(split_result['question_prefixes']):>10,}"
                f"{int(split_result['prefixes_to_clean']):>10,}"
                f"{int(split_result['valid_samples']):>10,}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate and clean MuSiQue teacher-generated "
            "training data."
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
    print(
        "CLEAN TEACHER TRAINING DATASET"
    )
    print("=" * 80)

    print()
    print(
        f"Dataset : {DATASET_REPO}"
    )

    print(
        f"Mode    : "
        f"{'DRY RUN' if args.dry_run else 'CLEAN + UPLOAD'}"
    )

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    total_samples = 0

    total_missing_teacher_output = 0
    total_empty_teacher_output = 0
    total_missing_final_answer = 0

    total_answer_matches = 0
    total_answer_mismatches = 0

    total_question_prefixes = 0
    total_prefixes_cleaned = 0
    total_prefixes_after = 0

    total_valid_samples = 0

    all_results = []

    # ========================================================
    # PROCESS CONFIGS
    # ========================================================

    for config_name in CONFIGS:

        result = clean_config(
            config_name=config_name,
            dry_run=args.dry_run,
        )

        all_results.append(
            result
        )

        # ----------------------------------------------------
        # Accumulate statistics
        # ----------------------------------------------------

        for split_result in result[
            "split_results"
        ].values():

            total_samples += int(
                split_result["total"]
            )

            total_missing_teacher_output += int(
                split_result[
                    "missing_teacher_output"
                ]
            )

            total_empty_teacher_output += int(
                split_result[
                    "empty_teacher_output"
                ]
            )

            total_missing_final_answer += int(
                split_result[
                    "missing_final_answer"
                ]
            )

            total_answer_matches += int(
                split_result[
                    "answer_matches"
                ]
            )

            total_answer_mismatches += int(
                split_result[
                    "answer_mismatches"
                ]
            )

            total_question_prefixes += int(
                split_result[
                    "question_prefixes"
                ]
            )

            total_valid_samples += int(
                split_result[
                    "valid_samples"
                ]
            )

        total_prefixes_cleaned += int(
            result[
                "prefixes_cleaned"
            ]
        )

        total_prefixes_after += int(
            result[
                "prefixes_after"
            ]
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)

    if args.dry_run:

        print(
            "DRY RUN — GLOBAL SUMMARY"
        )

    else:

        print(
            "CLEANING — GLOBAL SUMMARY"
        )

    print("=" * 80)

    print()
    print(
        f"Total samples               : "
        f"{total_samples:,}"
    )

    # --------------------------------------------------------
    # Removal statistics
    # --------------------------------------------------------

    total_removed = (
        total_samples
        - total_valid_samples
    )

    print()
    print(
        "REMOVED SAMPLES"
    )

    print("-" * 80)

    print(
        f"Total removed                : "
        f"{total_removed:,}"
    )

    print(
        f"  - Missing teacher output  : "
        f"{total_missing_teacher_output:,}"
    )

    print(
        f"  - Empty teacher output    : "
        f"{total_empty_teacher_output:,}"
    )

    print(
        f"  - Missing final <answer>  : "
        f"{total_missing_final_answer:,}"
    )

    print(
        f"  - Answer mismatch         : "
        f"{total_answer_mismatches:,}"
    )

    # --------------------------------------------------------
    # Answer statistics
    # --------------------------------------------------------

    print()
    print(
        "ANSWER VALIDATION"
    )

    print("-" * 80)

    print(
        f"Answer matches               : "
        f"{total_answer_matches:,}"
    )

    print(
        f"Answer mismatches            : "
        f"{total_answer_mismatches:,}"
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
        f"Answer match rate            : "
        f"{global_match_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Prefix statistics
    # --------------------------------------------------------

    print()
    print(
        "QUESTION PREFIX CLEANING"
    )

    print("-" * 80)

    print(
        f"Prefixes found in raw data  : "
        f"{total_question_prefixes:,}"
    )

    print(
        f"Prefixes actually cleaned   : "
        f"{total_prefixes_cleaned:,}"
    )

    print(
        f"Prefixes remaining           : "
        f"{total_prefixes_after:,}"
    )

    # --------------------------------------------------------
    # Final dataset size
    # --------------------------------------------------------

    print()
    print(
        "FINAL DATASET"
    )

    print("-" * 80)

    print(
        f"Valid samples to keep        : "
        f"{total_valid_samples:,}"
    )

    # ========================================================
    # PER-HOP TABLE
    # ========================================================

    print_per_hop_table(
        all_results
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

        print(
            "✓ DRY RUN COMPLETE"
        )

        print()
        print(
            "No remote dataset was modified."
        )

        print(
            "The statistics above show:"
        )

        print(
            "  - how many samples would be removed"
        )

        print(
            "  - why each sample would be removed"
        )

        print(
            "  - how many question prefixes would be cleaned"
        )

        print(
            "  - how many valid samples would remain"
        )

    else:

        print(
            "✓ DATASET CLEANING COMPLETE"
        )

        print()
        print(
            f"Removed invalid samples       : "
            f"{total_removed:,}"
        )

        print(
            f"Remaining valid samples       : "
            f"{total_valid_samples:,}"
        )

        print(
            f"Question prefixes found       : "
            f"{total_question_prefixes:,}"
        )

        print(
            f"Question prefixes cleaned     : "
            f"{total_prefixes_cleaned:,}"
        )

        print(
            f"Question prefixes remaining   : "
            f"{total_prefixes_after:,}"
        )

        print()
        print(
            "Uploaded cleaned dataset to:"
        )

        print(
            f"  {DATASET_REPO}"
        )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
