# ============================================================
# MuSiQue Dataset Utility
# ============================================================

from pathlib import Path
import zipfile
import shutil

import gdown


# ============================================================
# CONFIG
# ============================================================

MUSIQUE_FILE_ID = (
    "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
)

ZIP_NAME = "musique_v1.0.zip"


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FILES = [
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_test.jsonl",
    "musique_full_v1.0_train.jsonl",
    "musique_full_v1.0_dev.jsonl",
    "musique_full_v1.0_test.jsonl",
]


# ============================================================
# PATH
# ============================================================

def get_musique_path():
    """
    Return the directory containing the MuSiQue dataset.

    Dataset location is always relative to this file:

        <this_file_directory>/
        ├── dataset.py
        └── data/
            └── musique/
                ├── musique_v1.0.zip
                ├── musique_ans_v1.0_train.jsonl
                ├── musique_ans_v1.0_dev.jsonl
                ├── musique_ans_v1.0_test.jsonl
                ├── musique_full_v1.0_train.jsonl
                ├── musique_full_v1.0_dev.jsonl
                └── musique_full_v1.0_test.jsonl

    This function does NOT download anything.
    """

    this_dir = Path(__file__).resolve().parent

    return (
        this_dir
        / "data"
        / "musique"
    )


# ============================================================
# CHECK DATASET
# ============================================================

def _dataset_exists(dataset_dir):
    """
    Check whether all required MuSiQue files exist.
    """

    return all(
        (dataset_dir / filename).is_file()
        for filename in REQUIRED_FILES
    )


# ============================================================
# DOWNLOAD MuSiQue
# ============================================================

def download_musique():
    """
    Check and download MuSiQue v1.0 if necessary.

    The dataset is always stored relative to this file.

    Returns
    -------
    Path
        Path to the MuSiQue dataset directory.
    """

    # ========================================================
    # DIRECTORY OF THIS FILE
    # ========================================================

    dataset_dir = get_musique_path()

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = dataset_dir / ZIP_NAME


    # ========================================================
    # CHECK EXISTING DATASET
    # ========================================================

    if _dataset_exists(dataset_dir):

        print("=" * 80)
        print("MuSiQue v1.0 — ALREADY AVAILABLE")
        print("=" * 80)

        print(
            "Dataset directory:\n"
            f"  {dataset_dir}"
        )

        return dataset_dir


    # ========================================================
    # DATASET NOT FOUND
    # ========================================================

    print("=" * 80)
    print("MuSiQue v1.0 — DATASET NOT FOUND")
    print("=" * 80)

    print(
        "Target directory:\n"
        f"  {dataset_dir}"
    )


    # ========================================================
    # DOWNLOAD ZIP
    # ========================================================

    if not zip_path.exists():

        print("\n" + "=" * 80)
        print("DOWNLOADING MuSiQue v1.0")
        print("=" * 80)

        print(
            "Output:\n"
            f"  {zip_path}"
        )

        gdown.download(
            id=MUSIQUE_FILE_ID,
            output=str(zip_path),
            quiet=False,
        )

        if not zip_path.exists():

            raise RuntimeError(
                "MuSiQue download failed."
            )

    else:

        print("\n" + "=" * 80)
        print("ZIP ALREADY EXISTS")
        print("=" * 80)

        print(
            "Using:\n"
            f"  {zip_path}"
        )


    # ========================================================
    # EXTRACT
    #
    # IMPORTANT:
    #
    # The official ZIP has this structure:
    #
    # data/
    #   musique_ans_v1.0_train.jsonl
    #   ...
    #
    # We flatten the ZIP structure and place the files
    # directly into:
    #
    # <this_file>/data/musique/
    # ========================================================

    print("\n" + "=" * 80)
    print("EXTRACTING MuSiQue v1.0")
    print("=" * 80)

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zip_file:

        members = zip_file.namelist()

        for filename in REQUIRED_FILES:

            # ------------------------------------------------
            # Find the file anywhere inside the ZIP
            # ------------------------------------------------

            matches = [
                member
                for member in members
                if Path(member).name == filename
            ]

            if not matches:
                continue

            source_member = matches[0]

            destination = dataset_dir / filename

            print(
                f"  Extracting: {filename}"
            )

            with zip_file.open(source_member) as source:
                with open(destination, "wb") as target:
                    shutil.copyfileobj(
                        source,
                        target,
                    )


    # ========================================================
    # VERIFY
    # ========================================================

    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not (dataset_dir / filename).is_file()
    ]

    if missing_files:

        raise RuntimeError(
            "MuSiQue extraction completed, "
            "but required files are missing:\n"
            + "\n".join(
                f"  - {filename}"
                for filename in missing_files
            )
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(
        "Dataset directory:\n"
        f"  {dataset_dir}"
    )

    print("\nFiles:")

    for filename in REQUIRED_FILES:

        path = dataset_dir / filename

        size_mb = (
            path.stat().st_size
            / 1024**2
        )

        print(
            f"  {filename:<40}"
            f"{size_mb:>8.2f} MB"
        )

    print("=" * 80)

    return dataset_dir


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    download_musique()