from datasets import load_dataset
from typing import Dict, Any


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = (
    "adaptive-retrieval-timing/"
    "MuSiQue-ART-Training"
)


# ============================================================
# LOAD TEACHER-GENERATED TRAINING DATASET
# ============================================================

def load_teacher_generated_training() -> Dict[str, Any]:
    """
    Load teacher-generated training datasets.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping each hop configuration directly
        to its training split.

    Example
    -------
    {
        "2-hop": train_samples,
        "3-hop": train_samples,
        "4-hop": train_samples,
    }

    Notes
    -----
    The returned values are the training splits themselves.
    There is no intermediate "train" key.
    """

    datasets = {}

    for hop in (2, 3, 4):

        config_name = f"{hop}-hop"

        dataset = load_dataset(
            DATASET_NAME,
            config_name,
        )

        datasets[config_name] = dataset["train"]

    return datasets


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    datasets = (
        load_teacher_generated_training()
    )

    print("=" * 80)
    print("2-HOP — FIRST TRAINING SAMPLE")
    print("=" * 80)

    sample = datasets["2-hop"][0]

    for key, value in sample.items():

        print()
        print(f"[{key}]")
        print(value)
