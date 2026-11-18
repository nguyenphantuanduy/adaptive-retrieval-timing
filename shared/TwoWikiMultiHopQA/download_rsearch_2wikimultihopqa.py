
#!/usr/bin/env python3
"""
Download the R-Search 2WikiMultiHopQA dataset from Hugging Face.

Repository:
    qingfei1/R-Search_datasets

Remote directory:
    data/corpus/2wikimultihopqa/

Destination:
    <directory-containing-this-script>/data/corpus/2wikimultihopqa/

Downloads:
    - train.json
    - dev.json
    - test.json
    - id_aliases.json

Behavior:
    - Clears the destination corpus directory on every run.
    - Downloads using huggingface_hub.snapshot_download().
    - Uses Hugging Face parallel download.
    - Downloads only the requested corpus directory.
    - Preserves the repository's relative directory structure.

Requires:
    pip install -U huggingface_hub
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download

except ImportError:
    print(
        "Missing dependency: huggingface_hub",
        file=sys.stderr,
    )
    print(
        f"Install it with: {sys.executable} -m pip install -U huggingface_hub",
        file=sys.stderr,
    )
    raise SystemExit(1)


# ============================================================
# CONFIG
# ============================================================

REPO_ID = "qingfei1/R-Search_datasets"

REPO_SUBDIR = "data/corpus/2wikimultihopqa"

MAX_WORKERS = 8

REQUIRED_FILES = (
    "train.json",
    "dev.json",
    "test.json",
)

OPTIONAL_FILES = (
    "id_aliases.json",
)


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    # Final corpus directory.
    corpus_dir = (
        script_dir
        / "data"
        / "corpus"
        / "2wikimultihopqa"
    )

    # Temporary download directory.
    staging_dir = (
        script_dir
        / "_hf_2wikimultihopqa_download"
    )

    print("=" * 80)
    print("DOWNLOADING R-SEARCH 2WIKIMULTIHOPQA")
    print("=" * 80)

    print(f"Script path  : {script_path}")
    print(f"Repository   : {REPO_ID}")
    print(f"Remote folder: {REPO_SUBDIR}")
    print(f"Corpus folder: {corpus_dir}")
    print(f"Max workers  : {MAX_WORKERS}")

    # --------------------------------------------------------
    # 1. Clear previous data.
    # --------------------------------------------------------

    print("\n[1/3] Clearing previous data...")

    try:

        if corpus_dir.exists():
            shutil.rmtree(corpus_dir)
            print(f"Removed: {corpus_dir}")

        if staging_dir.exists():
            shutil.rmtree(staging_dir)
            print(f"Removed: {staging_dir}")

    except OSError as exc:

        print(
            f"Failed to clear previous data: {exc}",
            file=sys.stderr,
        )
        return 1

    # --------------------------------------------------------
    # 2. Download from Hugging Face.
    # --------------------------------------------------------

    print("\n[2/3] Downloading from Hugging Face...")

    try:

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",

            allow_patterns=[
                f"{REPO_SUBDIR}/*",
            ],

            local_dir=str(staging_dir),

            max_workers=MAX_WORKERS,
        )

    except Exception as exc:

        print(
            f"\nDownload failed: {exc}",
            file=sys.stderr,
        )

        print(
            f"Partial download: {staging_dir}",
            file=sys.stderr,
        )

        print(
            "Run the script again to retry.",
            file=sys.stderr,
        )

        return 1

    # --------------------------------------------------------
    # 3. Validate and move downloaded data.
    # --------------------------------------------------------

    print("\n[3/3] Preparing final corpus directory...")

    downloaded_corpus = (
        staging_dir / REPO_SUBDIR
    )

    if not downloaded_corpus.is_dir():

        print(
            f"Expected directory not found: {downloaded_corpus}",
            file=sys.stderr,
        )
        return 1

    # Validate required files.
    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not (downloaded_corpus / filename).is_file()
    ]

    if missing_files:

        print(
            f"Missing required files: {missing_files}",
            file=sys.stderr,
        )
        return 1

    # Check optional files.
    available_optional = [
        filename
        for filename in OPTIONAL_FILES
        if (downloaded_corpus / filename).is_file()
    ]

    try:

        corpus_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.move(
            str(downloaded_corpus),
            str(corpus_dir),
        )

        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

    except OSError as exc:

        print(
            f"Failed to prepare final corpus: {exc}",
            file=sys.stderr,
        )
        return 1

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(f"Corpus directory: {corpus_dir}")

    print("\nDownloaded files:")

    total_bytes = 0

    for filename in REQUIRED_FILES + OPTIONAL_FILES:

        path = corpus_dir / filename

        if path.is_file():

            size = path.stat().st_size
            total_bytes += size

            print(
                f"    {filename:<25} "
                f"{size / (1024 ** 2):.2f} MB"
            )

    print(f"\nTotal size: {total_bytes / (1024 ** 3):.2f} GiB")

    print("\nYou can now build the retrieval corpus with:")

    print(
        "python shared/2WikiMultiHopQA/"
        "build_rsearch_2wikimultihopqa_corups.py"
    )

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
