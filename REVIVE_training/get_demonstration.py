from __future__ import annotations

import json
from pathlib import Path

from shared.MuSiQue.rsearch_musique_test_dataset import (
    load_rsearch_musique,
)

from shared.MuSiQue.load_hypothesis_validation import (
    load_teacher_generated_hypothesis_validation,
)


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FILE = Path(
    "REVIVE_training/result/"
    "teacher_hypothesis_demonstrations.jsonl"
)


# ============================================================
# HELPERS
# ============================================================

def get_question(sample: dict) -> str:
    """
    Extract question from a dataset sample.
    """

    return sample["question"].strip()


def get_teacher_generation_output(sample: dict):
    """
    Extract teacher generation output.

    Expected field:
        teacher_generation_output
    """

    return sample["teacher_generation_output"]


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD R-SEARCH TEST SET
    # ========================================================

    print("=" * 80)
    print("LOADING R-SEARCH MuSiQue TEST SET")
    print("=" * 80)

    test_set = load_rsearch_musique()

    test_questions = {
        get_question(sample)
        for sample in test_set
    }

    print(
        f"Unique test questions: "
        f"{len(test_questions)}"
    )


    # ========================================================
    # LOAD TEACHER HYPOTHESIS VALIDATION SET
    # ========================================================

    print("\n" + "=" * 80)
    print("LOADING TEACHER HYPOTHESIS VALIDATION SET")
    print("=" * 80)

    validation_sets = (
        load_teacher_generated_hypothesis_validation()
    )


    # ========================================================
    # SELECT DEMONSTRATIONS
    # ========================================================

    demonstrations = []

    print("\n" + "=" * 80)
    print("SELECTING DEMONSTRATIONS")
    print("=" * 80)

    for hop in (2, 3, 4):

        config_name = f"{hop}-hop"

        validation_set = validation_sets[config_name]

        selected_sample = None

        for sample in validation_set:

            question = get_question(sample)

            # Skip if this question already exists
            # in the R-Search test set.
            if question in test_questions:
                continue

            selected_sample = sample
            break

        if selected_sample is None:

            raise RuntimeError(
                f"Could not find a demonstration "
                f"sample for {config_name} that is "
                f"not present in the R-Search test set."
            )

        demonstration = {
            "hop": hop,
            "question": get_question(
                selected_sample
            ),
            "teacher_generation_output": (
                get_teacher_generation_output(
                    selected_sample
                )
            ),
        }

        demonstrations.append(
            demonstration
        )

        print()
        print(f"[{config_name}]")
        print(
            f"Question: "
            f"{demonstration['question']}"
        )


    # ========================================================
    # SAVE JSONL
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for demonstration in demonstrations:

            fout.write(
                json.dumps(
                    demonstration,
                    ensure_ascii=False,
                )
                + "\n"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 80)
    print("DEMONSTRATIONS SAVED")
    print("=" * 80)

    print(
        f"Output file:\n"
        f"  {OUTPUT_FILE}"
    )

    print(
        f"Number of demonstrations: "
        f"{len(demonstrations)}"
    )

    print("=" * 80)


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    main()