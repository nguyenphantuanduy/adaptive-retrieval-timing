# ============================================================
# MuSiQue-Ans Dataset Loader
# ============================================================

from pathlib import Path
import json

from .dataset import download_musique


# ============================================================
# LOAD MuSiQue-Ans
# ============================================================

def load_musique_ans():
    """
    Download MuSiQue v1.0 if necessary and load
    the MuSiQue-Ans train/dev/test splits.

    Returns
    -------
    dict
        {
            "train": [...],
            "dev": [...],
            "test": [...]
        }
    """

    # ========================================================
    # ENSURE DATASET EXISTS
    # ========================================================

    dataset_dir = download_musique()


    # ========================================================
    # FILE PATHS
    # ========================================================

    train_path = dataset_dir / "musique_ans_v1.0_train.jsonl"
    dev_path = dataset_dir / "musique_ans_v1.0_dev.jsonl"
    test_path = dataset_dir / "musique_ans_v1.0_test.jsonl"


    # ========================================================
    # LOAD JSONL
    # ========================================================

    def load_jsonl(path):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return [
                json.loads(line)
                for line in f
                if line.strip()
            ]


    print("=" * 80)
    print("LOADING MuSiQue-Ans v1.0")
    print("=" * 80)

    print(f"Train: {train_path}")
    print(f"Dev  : {dev_path}")
    print(f"Test : {test_path}")


    train = load_jsonl(train_path)
    dev = load_jsonl(dev_path)
    test = load_jsonl(test_path)


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("DATASET LOADED")
    print("=" * 80)

    print(f"Train : {len(train):,} samples")
    print(f"Dev   : {len(dev):,} samples")
    print(f"Test  : {len(test):,} samples")

    print("-" * 80)
    print(f"Total : {len(train) + len(dev) + len(test):,} samples")

    print("=" * 80)


    return {
        "train": train,
        "dev": dev,
        "test": test,
    }


# ============================================================
# MAIN — TEST
# ============================================================

if __name__ == "__main__":

    dataset = load_musique_ans()

    # --------------------------------------------------------
    # Print first sample from DEV
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FIRST SAMPLE — DEV")
    print("=" * 80)

    print(dataset["dev"][0])