# ============================================================
# R-Search MuSiQue Evaluation Dataset Utility
# ============================================================

from pathlib import Path
import json

from huggingface_hub import hf_hub_download


# ============================================================
# CONFIG
# ============================================================

HF_REPO_ID = "qingfei1/R-Search_datasets"
HF_REPO_TYPE = "dataset"

HF_TEST_FILE = "data/eval/musique/test.jsonl"

TEST_FILENAME = "rsearch_musique_test.jsonl"


# ============================================================
# PATH
# ============================================================

def get_rsearch_musique_path():
    """
    Return the local directory containing the R-Search
    MuSiQue evaluation dataset.

    Dataset location is always relative to this file:

        <this_file_directory>/
        ├── dataset.py
        └── data/
            └── rsearch_musique/
                └── rsearch_musique_test.jsonl

    This function does NOT download anything.
    """

    this_dir = Path(__file__).resolve().parent

    return (
        this_dir
        / "data"
        / "musique_rsearch"
    )


# ============================================================
# CHECK DATASET
# ============================================================

def _dataset_exists(dataset_dir):
    """
    Check whether the R-Search MuSiQue test file exists.
    """

    return (
        dataset_dir / TEST_FILENAME
    ).is_file()


# ============================================================
# DOWNLOAD R-SEARCH MuSiQue TEST SET
# ============================================================

def download_rsearch_musique():
    """
    Check and download the R-Search MuSiQue evaluation
    test set if necessary.

    Returns
    -------
    Path
        Path to the R-Search MuSiQue test JSONL file.
    """

    # ========================================================
    # DIRECTORY
    # ========================================================

    dataset_dir = get_rsearch_musique_path()

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_path = dataset_dir / TEST_FILENAME


    # ========================================================
    # CHECK EXISTING DATASET
    # ========================================================

    if _dataset_exists(dataset_dir):

        print("=" * 80)
        print("R-SEARCH MuSiQue TEST SET — ALREADY AVAILABLE")
        print("=" * 80)

        print(
            "Dataset file:\n"
            f"  {test_path}"
        )

        return test_path


    # ========================================================
    # DATASET NOT FOUND
    # ========================================================

    print("=" * 80)
    print("R-SEARCH MuSiQue TEST SET — DATASET NOT FOUND")
    print("=" * 80)

    print(
        "Target file:\n"
        f"  {test_path}"
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    print("\n" + "=" * 80)
    print("DOWNLOADING R-SEARCH MuSiQue TEST SET")
    print("=" * 80)

    print(
        "Hugging Face repository:\n"
        f"  {HF_REPO_ID}"
    )

    print(
        "Remote file:\n"
        f"  {HF_TEST_FILE}"
    )

    cached_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_TEST_FILE,
        repo_type=HF_REPO_TYPE,
    )

    # ========================================================
    # COPY FROM HF CACHE TO PROJECT DATA DIRECTORY
    # ========================================================

    source_path = Path(cached_path)

    test_path.write_bytes(
        source_path.read_bytes()
    )

    if not test_path.exists():

        raise RuntimeError(
            "R-Search MuSiQue test set download failed."
        )


    # ========================================================
    # VERIFY JSONL
    # ========================================================

    print("\n" + "=" * 80)
    print("VERIFYING DATASET")
    print("=" * 80)

    num_samples = 0

    with open(
        test_path,
        "r",
        encoding="utf-8",
    ) as fin:

        for line in fin:

            line = line.strip()

            if not line:
                continue

            json.loads(line)

            num_samples += 1


    # ========================================================
    # COMPLETE
    # ========================================================

    size_mb = (
        test_path.stat().st_size
        / 1024**2
    )

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(
        f"Dataset file:\n"
        f"  {test_path}"
    )

    print(
        f"Samples:\n"
        f"  {num_samples}"
    )

    print(
        f"Size:\n"
        f"  {size_mb:.2f} MB"
    )

    print("=" * 80)

    return test_path


# ============================================================
# LOAD DATASET
# ============================================================

def load_rsearch_musique():
    """
    Download the R-Search MuSiQue test set if necessary,
    then load and return it as a list of dictionaries.

    Returns
    -------
    list[dict]
        R-Search MuSiQue evaluation samples.
    """

    test_path = download_rsearch_musique()

    print("\n" + "=" * 80)
    print("LOADING R-SEARCH MuSiQue TEST SET")
    print("=" * 80)

    dataset = []

    with open(
        test_path,
        "r",
        encoding="utf-8",
    ) as fin:

        for line in fin:

            line = line.strip()

            if not line:
                continue

            dataset.append(
                json.loads(line)
            )

    print(
        f"Loaded samples: {len(dataset)}"
    )

    print("=" * 80)

    return dataset


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    dataset = load_rsearch_musique()

    print(
        "\nFirst sample:"
    )

    print(
        json.dumps(
            dataset[0],
            ensure_ascii=False,
            indent=2,
        )
    )