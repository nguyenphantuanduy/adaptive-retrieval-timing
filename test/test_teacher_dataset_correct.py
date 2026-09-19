
# ============================================================
# TEST TEACHER GENERATED HUGGING FACE DATASET
#
# Check:
#   1. teacher_generation_output exists
#   2. Teacher output contains final "<answer>...</answer>"
#   3. Extracted final answer matches ground-truth answer
#
# Dataset:
#   adaptive-retrieval-timing/teacher-generated-hypothesis-validation
#
# Configs:
#   2-hop
#   3-hop
#   4-hop
#
# Split:
#   validation
# ============================================================

import re

from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = (
    "adaptive-retrieval-timing/"
    "teacher-generated-hypothesis-validation"
)

CONFIGS = (
    "2-hop",
    "3-hop",
    "4-hop",
)

SPLIT = "validation"


# ============================================================
# ANSWER NORMALIZATION
# ============================================================

def normalize_answer(answer):
    """
    Normalize an answer for robust comparison.

    This intentionally performs only lightweight normalization.
    It does NOT perform semantic matching.
    """

    if answer is None:
        return ""

    answer = str(answer)

    # Normalize whitespace
    answer = re.sub(
        r"\s+",
        " ",
        answer,
    ).strip()

    # Lowercase
    answer = answer.lower()

    # Remove surrounding common punctuation
    answer = answer.strip(
        " \t\n\r.,;:!?\"'`"
    )

    return answer


# ============================================================
# EXTRACT FINAL ANSWER
# ============================================================

def extract_final_answer(
    teacher_output,
):
    """
    Extract the final answer from teacher_generation_output.

    Expected format:

        <answer>
        something
        </answer>

    The function searches for the LAST occurrence of
    "<answer>...</answer>" because the teacher may theoretically
    mention the answer format earlier in its reasoning.
    """

    if teacher_output is None:
        return None

    teacher_output = str(
        teacher_output
    ).strip()

    if not teacher_output:
        return None

    # --------------------------------------------------------
    # Find all occurrences of <answer>...</answer>
    # --------------------------------------------------------

    matches = list(
        re.finditer(
            r"(?is)<answer>\s*(.*?)\s*</answer>",
            teacher_output,
        )
    )

    if not matches:
        return None

    # --------------------------------------------------------
    # Use the LAST <answer>...</answer> occurrence
    # --------------------------------------------------------

    answer = matches[-1].group(1).strip()

    # --------------------------------------------------------
    # Remove common markdown formatting
    # --------------------------------------------------------

    answer = answer.strip(
        "*_` "
    )

    return answer


# ============================================================
# CHECK ONE SAMPLE
# ============================================================

def check_sample(
    sample,
):
    """
    Check one dataset sample.

    Returns a dictionary containing:
        - missing_teacher_output
        - missing_final_answer
        - answer_match
        - expected_answer
        - predicted_answer
    """

    # --------------------------------------------------------
    # Ground-truth answer
    # --------------------------------------------------------

    expected_answer = sample.get(
        "answer"
    )

    # --------------------------------------------------------
    # Teacher output
    # --------------------------------------------------------

    teacher_output = sample.get(
        "teacher_generation_output"
    )

    if (
        teacher_output is None
        or not str(teacher_output).strip()
    ):

        return {
            "missing_teacher_output": True,
            "missing_final_answer": True,
            "answer_match": False,
            "expected_answer": expected_answer,
            "predicted_answer": None,
        }

    # --------------------------------------------------------
    # Extract final <answer>...</answer>
    # --------------------------------------------------------

    predicted_answer = (
        extract_final_answer(
            teacher_output
        )
    )

    if predicted_answer is None:

        return {
            "missing_teacher_output": False,
            "missing_final_answer": True,
            "answer_match": False,
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
        "missing_final_answer": False,
        "answer_match": answer_match,
        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
    }


# ============================================================
# CHECK ONE HOP
# ============================================================

def check_hop_dataset(
    dataset,
    hop,
):
    """
    Validate every sample in one hop dataset.
    """

    print()
    print("=" * 80)
    print(
        f"CHECKING {hop}"
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
    missing_final_answer_count = 0
    answer_mismatch_count = 0
    answer_match_count = 0

    # --------------------------------------------------------
    # Error containers
    # --------------------------------------------------------

    missing_teacher_output_samples = []
    missing_final_answer_samples = []
    answer_mismatch_samples = []

    # --------------------------------------------------------
    # Iterate over all samples
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

        # ====================================================
        # MISSING TEACHER OUTPUT
        # ====================================================

        if result[
            "missing_teacher_output"
        ]:

            missing_teacher_output_count += 1

            missing_teacher_output_samples.append(
                {
                    "index": index,
                    "id": sample_id,
                }
            )

            continue

        # ====================================================
        # MISSING FINAL ANSWER
        # ====================================================

        if result[
            "missing_final_answer"
        ]:

            missing_final_answer_count += 1

            missing_final_answer_samples.append(
                {
                    "index": index,
                    "id": sample_id,
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

        # ====================================================
        # ANSWER MATCH / MISMATCH
        # ====================================================

        if result[
            "answer_match"
        ]:

            answer_match_count += 1

        else:

            answer_mismatch_count += 1

            answer_mismatch_samples.append(
                {
                    "index": index,
                    "id": sample_id,
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

    # ========================================================
    # SUMMARY
    # ========================================================

    valid_final_answers = (
        total
        - missing_teacher_output_count
        - missing_final_answer_count
    )

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
        f"Missing final <answer>     : "
        f"{missing_final_answer_count:,}"
    )

    print(
        f"Valid final answers        : "
        f"{valid_final_answers:,}"
    )

    print(
        f"Answer matches             : "
        f"{answer_match_count:,}"
    )

    print(
        f"Answer mismatches          : "
        f"{answer_mismatch_count:,}"
    )

    # --------------------------------------------------------
    # Match rate
    # --------------------------------------------------------

    if valid_final_answers > 0:

        match_rate = (
            answer_match_count
            / valid_final_answers
            * 100
        )

    else:

        match_rate = 0.0

    print(
        f"Answer match rate          : "
        f"{match_rate:.2f}%"
    )

    # ========================================================
    # PRINT MISSING FINAL ANSWERS
    # ========================================================

    if missing_final_answer_samples:

        print()
        print("-" * 80)
        print(
            "SAMPLES WITHOUT FINAL ANSWER"
        )
        print("-" * 80)

        for item in (
            missing_final_answer_samples
        ):

            print()
            print(
                f"[index] "
                f"{item['index']}"
            )

            print(
                f"[id] "
                f"{item['id']}"
            )

            print(
                f"[expected answer] "
                f"{item['expected_answer']}"
            )

            print(
                "[teacher output]"
            )

            print(
                item["teacher_output"]
            )

    # ========================================================
    # PRINT ANSWER MISMATCHES
    # ========================================================

    if answer_mismatch_samples:

        print()
        print("-" * 80)
        print(
            "ANSWER MISMATCHES"
        )
        print("-" * 80)

        for item in (
            answer_mismatch_samples
        ):

            print()
            print(
                f"[index] "
                f"{item['index']}"
            )

            print(
                f"[id] "
                f"{item['id']}"
            )

            print(
                f"[expected] "
                f"{item['expected_answer']}"
            )

            print(
                f"[predicted] "
                f"{item['predicted_answer']}"
            )

            print()
            print(
                "[teacher output]"
            )

            print(
                item["teacher_output"]
            )

            print()
            print(
                "-" * 80
            )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "hop": hop,
        "total": total,
        "missing_teacher_output": (
            missing_teacher_output_count
        ),
        "missing_final_answer": (
            missing_final_answer_count
        ),
        "valid_final_answers": (
            valid_final_answers
        ),
        "answer_matches": (
            answer_match_count
        ),
        "answer_mismatches": (
            answer_mismatch_count
        ),
        "match_rate": match_rate,
        "missing_teacher_output_samples": (
            missing_teacher_output_samples
        ),
        "missing_final_answer_samples": (
            missing_final_answer_samples
        ),
        "answer_mismatch_samples": (
            answer_mismatch_samples
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "TEST HUGGING FACE TEACHER DATASET"
    )
    print("=" * 80)

    print()
    print(
        f"Dataset : {DATASET_NAME}"
    )

    print(
        f"Split   : {SPLIT}"
    )

    # ========================================================
    # GLOBAL RESULTS
    # ========================================================

    all_results = []

    total_samples = 0
    total_missing_teacher_output = 0
    total_missing_final_answer = 0
    total_valid_final_answers = 0
    total_answer_matches = 0
    total_answer_mismatches = 0

    # ========================================================
    # LOAD + CHECK EACH HOP
    # ========================================================

    for hop in CONFIGS:

        print()
        print("=" * 80)
        print(
            f"LOADING {hop}"
        )
        print("=" * 80)

        dataset = load_dataset(
            DATASET_NAME,
            name=hop,
            split=SPLIT,
            download_mode="force_redownload",
        )

        print()
        print(
            f"Config      : {hop}"
        )

        print(
            f"Split       : {SPLIT}"
        )

        print(
            f"Num samples : {len(dataset):,}"
        )

        result = check_hop_dataset(
            dataset=dataset,
            hop=hop,
        )

        all_results.append(
            result
        )

        # ----------------------------------------------------
        # Accumulate global statistics
        # ----------------------------------------------------

        total_samples += (
            result["total"]
        )

        total_missing_teacher_output += (
            result[
                "missing_teacher_output"
            ]
        )

        total_missing_final_answer += (
            result[
                "missing_final_answer"
            ]
        )

        total_valid_final_answers += (
            result[
                "valid_final_answers"
            ]
        )

        total_answer_matches += (
            result[
                "answer_matches"
            ]
        )

        total_answer_mismatches += (
            result[
                "answer_mismatches"
            ]
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "GLOBAL SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"Total samples               : "
        f"{total_samples:,}"
    )

    print(
        f"Missing teacher output      : "
        f"{total_missing_teacher_output:,}"
    )

    print(
        f"Missing final <answer>      : "
        f"{total_missing_final_answer:,}"
    )

    print(
        f"Valid final answers         : "
        f"{total_valid_final_answers:,}"
    )

    print(
        f"Answer matches              : "
        f"{total_answer_matches:,}"
    )

    print(
        f"Answer mismatches           : "
        f"{total_answer_mismatches:,}"
    )

    if total_valid_final_answers > 0:

        global_match_rate = (
            total_answer_matches
            / total_valid_final_answers
            * 100
        )

    else:

        global_match_rate = 0.0

    print(
        f"Overall answer match rate   : "
        f"{global_match_rate:.2f}%"
    )

    # ========================================================
    # HOP TABLE
    # ========================================================

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
        f"{'No Answer':>12}"
        f"{'Match':>10}"
        f"{'Mismatch':>12}"
        f"{'Rate':>10}"
    )

    print("-" * 80)

    for result in all_results:

        print(
            f"{result['hop']:<8}"
            f"{result['total']:>10,}"
            f"{result['missing_teacher_output']:>12,}"
            f"{result['missing_final_answer']:>12,}"
            f"{result['answer_matches']:>10,}"
            f"{result['answer_mismatches']:>12,}"
            f"{result['match_rate']:>9.2f}%"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 80)

    if (
        total_missing_teacher_output == 0
        and total_missing_final_answer == 0
        and total_answer_mismatches == 0
    ):

        print(
            "✓ ALL SAMPLES PASSED"
        )

        print(
            "✓ Every sample has teacher output"
        )

        print(
            "✓ Every teacher output has final <answer>"
        )

        print(
            "✓ Every final answer matches ground truth"
        )

    else:

        print(
            "✗ DATASET VALIDATION FAILED"
        )

        if total_missing_teacher_output > 0:

            print(
                f"  - "
                f"{total_missing_teacher_output:,} "
                f"samples missing teacher output"
            )

        if total_missing_final_answer > 0:

            print(
                f"  - "
                f"{total_missing_final_answer:,} "
                f"samples missing final <answer>"
            )

        if total_answer_mismatches > 0:

            print(
                f"  - "
                f"{total_answer_mismatches:,} "
                f"samples with answer mismatch"
            )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()