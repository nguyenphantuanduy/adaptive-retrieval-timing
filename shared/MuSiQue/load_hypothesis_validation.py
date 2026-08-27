from datasets import load_dataset
from typing import Dict, Any


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = (
    "adaptive-retrieval-timing/"
    "teacher-generated-hypothesis-validation"
)


# ============================================================
# LOAD TEACHER-GENERATED HYPOTHESIS VALIDATION DATASET
# ============================================================

def load_teacher_generated_hypothesis_validation() -> Dict[str, Any]:
    """
    Load teacher-generated hypothesis validation datasets.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping each hop configuration directly
        to its validation split.

    Example
    -------
    {
        "2-hop": validation_samples,
        "3-hop": validation_samples,
        "4-hop": validation_samples,
    }

    Notes
    -----
    The returned values are the validation splits themselves.
    There is no intermediate "validation" key.
    """

    datasets = {}

    for hop in (2, 3, 4):

        config_name = f"{hop}-hop"

        dataset = load_dataset(
            DATASET_NAME,
            config_name,
        )

        datasets[config_name] = dataset["validation"]

    return datasets


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    datasets = (
        load_teacher_generated_hypothesis_validation()
    )

    print("=" * 80)
    print("2-HOP — FIRST VALIDATION SAMPLE")
    print("=" * 80)

    sample = datasets["2-hop"][0]

    for key, value in sample.items():

        print()
        print(f"[{key}]")
        print(value)

