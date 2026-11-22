from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from REVIVE_training.utils import (
    acc_score,
    f1_scorer,
    compute_exact,
)


# ============================================================
# Configuration
# ============================================================

MAX_DEBUG_SAMPLES = 10


# ============================================================
# Argument parser
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ART-as-a-tool results on MuSiQue "
            "using Acc, F1, and EM."
        )
    )

    parser.add_argument(
        "--filename",
        type=Path,
        required=True,
        help="Input ART-as-a-tool evaluation JSONL file.",
    )

    return parser.parse_args()


# ============================================================
# Build output path
# ============================================================

def get_output_file(
    input_file: Path,
) -> Path:

    # --------------------------------------------------------
    # Replace "result" directory with "metrics"
    # --------------------------------------------------------

    parts = list(input_file.parts)

    if "result" in parts:

        result_index = (
            len(parts)
            - 1
            - parts[::-1].index("result")
        )

        parts[result_index] = "metrics"

        output_file = Path(*parts)

    else:

        output_file = (
            input_file.parent.parent
            / "metrics"
            / input_file.name
        )

    # --------------------------------------------------------
    # Replace filename suffix
    # --------------------------------------------------------

    if output_file.name.endswith(
        "_results.jsonl"
    ):

        output_name = (
            output_file.name[
                :-len("_results.jsonl")
            ]
            + "_metrics.json"
        )

    elif output_file.suffix == ".jsonl":

        output_name = (
            output_file.stem
            + "_metrics.json"
        )

    else:

        output_name = (
            output_file.stem
            + "_metrics.json"
        )

    return output_file.parent / output_name


# ============================================================
# Extract answer
# ============================================================

def extract_answer(
    text: str,
) -> tuple[str, bool]:

    matches = list(
        re.finditer(
            r"<answer>(.*?)</answer>",
            text,
            re.DOTALL,
        )
    )

    if matches:

        # Use the final <answer>...</answer>
        answer = matches[-1].group(1).strip()

        return answer, True

    # Same fallback behavior as R-Search
    return text.strip(), False


# ============================================================
# Load results
# ============================================================

def load_results(
    path: Path,
) -> list[dict]:

    results = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                data = json.loads(line)

            except json.JSONDecodeError:

                print(
                    f"WARNING: Could not parse line "
                    f"{line_number}"
                )

                continue

            # Ignore files that do not contain
            # ART-as-a-tool generated answers.
            if "art_as_tool_answer" not in data:

                continue

            results.append(data)

    return results


# ============================================================
# Main evaluation
# ============================================================

def main():

    args = parse_args()

    input_file = args.filename

    output_file = get_output_file(
        input_file
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{input_file}"
        )

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------

    results = load_results(
        input_file
    )

    if not results:

        raise ValueError(
            "No valid ART-as-a-tool evaluation samples found."
        )

    total_samples = len(results)

    print("=" * 80)
    print("ART-AS-A-TOOL R-SEARCH MUSIQUE EVALUATION")
    print("=" * 80)

    print(
        f"Input file : {input_file}"
    )

    print(
        f"Output file: {output_file}"
    )

    print(
        f"Samples    : {total_samples}"
    )

    print()

    # --------------------------------------------------------
    # Extract predictions and gold answers
    # --------------------------------------------------------

    predictions = []
    answers = []

    answer_tag_count = 0
    no_answer_tag_count = 0

    debug_no_answer = []

    for result in results:

        generated_text = result[
            "art_as_tool_answer"
        ]

        # ----------------------------------------------------
        # Extract answer
        # ----------------------------------------------------

        prediction, has_answer_tag = extract_answer(
            generated_text
        )

        if has_answer_tag:

            answer_tag_count += 1

        else:

            no_answer_tag_count += 1

            # Keep a few examples for debugging
            if len(debug_no_answer) < MAX_DEBUG_SAMPLES:

                debug_no_answer.append(
                    {
                        "id": result.get(
                            "id"
                        ),
                        "question": result.get(
                            "question",
                            "",
                        ),
                        "golden_answers": result.get(
                            "golden_answers",
                            [],
                        ),
                        "generated": generated_text,
                    }
                )

        # ----------------------------------------------------
        # Gold answers
        # ----------------------------------------------------

        ground_truths = result[
            "golden_answers"
        ]

        predictions.append(
            prediction
        )

        answers.append(
            ground_truths
        )

    # ========================================================
    # Answer extraction statistics
    # ========================================================

    answer_tag_percentage = (
        100.0
        * answer_tag_count
        / total_samples
    )

    no_answer_tag_percentage = (
        100.0
        * no_answer_tag_count
        / total_samples
    )

    print("-" * 80)
    print("ANSWER EXTRACTION")
    print("-" * 80)

    print(
        f"Has <answer> tag : "
        f"{answer_tag_count}/{total_samples} "
        f"({answer_tag_percentage:.2f}%)"
    )

    print(
        f"No <answer> tag  : "
        f"{no_answer_tag_count}/{total_samples} "
        f"({no_answer_tag_percentage:.2f}%)"
    )

    # ========================================================
    # Debug samples without answer tag
    # ========================================================

    if debug_no_answer:

        print()
        print("-" * 80)

        print(
            f"SAMPLES WITHOUT <answer> TAG "
            f"(showing up to {MAX_DEBUG_SAMPLES})"
        )

        print("-" * 80)

        for i, sample in enumerate(
            debug_no_answer,
            start=1,
        ):

            print()

            print(
                f"[{i}] ID={sample['id']}"
            )

            print(
                f"Question: {sample['question']}"
            )

            print(
                f"Gold: {sample['golden_answers']}"
            )

            print("Generated:")

            print(
                sample["generated"]
            )

            print("-" * 80)

    # ========================================================
    # Calculate metrics
    # ========================================================

    acc = acc_score(
        predictions,
        answers,
    )

    f1 = f1_scorer(
        predictions,
        answers,
    )

    em = compute_exact(
        predictions,
        answers,
    )

    # ========================================================
    # Print metric results
    # ========================================================

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Samples : {total_samples}"
    )

    print(
        f"Acc     : {acc:.2f}"
    )

    print(
        f"F1      : {f1:.2f}"
    )

    print(
        f"EM      : {em:.2f}"
    )

    print("=" * 80)

    # ========================================================
    # Save metrics
    # ========================================================

    metrics = {
        "num_samples": total_samples,

        "answer_extraction": {
            "has_answer_tag": answer_tag_count,
            "has_answer_tag_percent": round(
                answer_tag_percentage,
                2,
            ),
            "no_answer_tag": no_answer_tag_count,
            "no_answer_tag_percent": round(
                no_answer_tag_percentage,
                2,
            ),
        },

        "metrics": {
            "Acc": acc,
            "F1": f1,
            "EM": em,
        },
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved metrics to: {output_file}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()