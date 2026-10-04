from __future__ import annotations

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

INPUT_FILE = Path(
    "REVIVE_training/result/rsearch_musique_art_results.jsonl"
)

OUTPUT_FILE = Path(
    "REVIVE_training/result/rsearch_musique_art_metrics.json"
)

# Number of samples to print when <answer> tag is missing
MAX_DEBUG_SAMPLES = 10


# ============================================================
# Extract answer
# ============================================================

def extract_answer(text: str) -> tuple[str, bool]:
    """
    Extract the final answer from <answer>...</answer>.

    Same logic as R-Search:

        - Find all <answer>...</answer>
        - Take the last match
        - Strip whitespace
        - If no match, use the whole response

    Returns:
        answer:
            Extracted answer string.

        has_answer_tag:
            True if <answer>...</answer> was found.
    """

    matches = list(
        re.finditer(
            r"<answer>(.*?)</answer>",
            text,
            re.DOTALL,
        )
    )

    if matches:
        answer = matches[-1].group(1).strip()
        return answer, True

    # Same fallback behavior as R-Search
    return text.strip(), False


# ============================================================
# Load results
# ============================================================

def load_results(path: Path) -> list[dict]:
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

            # Ignore metric/result lines if the file
            # already contains one.
            if "pure_art_answer" not in data:
                continue

            results.append(data)

    return results


# ============================================================
# Main evaluation
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------

    results = load_results(INPUT_FILE)

    if not results:
        raise ValueError(
            "No valid ART evaluation samples found."
        )

    total_samples = len(results)

    print("=" * 80)
    print("ART R-SEARCH MUSIQUE EVALUATION")
    print("=" * 80)

    print(
        f"Input file : {INPUT_FILE}"
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

        generated_text = result["pure_art_answer"]

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
                        "id": result.get("id"),
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

        ground_truths = result["golden_answers"]

        predictions.append(prediction)
        answers.append(ground_truths)

    # ========================================================
    # Answer extraction statistics
    # ========================================================

    answer_tag_percentage = (
        100.0 * answer_tag_count / total_samples
    )

    no_answer_tag_percentage = (
        100.0 * no_answer_tag_count / total_samples
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

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
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
        f"Saved metrics to: {OUTPUT_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
