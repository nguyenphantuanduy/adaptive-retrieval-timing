#!/usr/bin/env python3
"""
Download the R-Search HotpotQA corpus from Hugging Face.

Destination:
    <directory-containing-this-script>/data/corpus/hotpotqa/

Behavior:
    - Clears the destination corpus directory on every run.
    - Downloads the corpus using huggingface_hub.snapshot_download().
    - Uses Hugging Face's parallel download mechanism.
    - Downloads only data/corpus/hotpotqa/* from the dataset repository.
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
    print("Missing dependency: huggingface_hub", file=sys.stderr)
    print(
        f"Install it with: {sys.executable} -m pip install -U huggingface_hub",
        file=sys.stderr,
    )
    raise SystemExit(1)


# ============================================================
# CONFIG
# ============================================================

REPO_ID = "qingfei1/R-Search_datasets"
REPO_SUBDIR = "data/corpus/hotpotqa"

# Number of concurrent downloads.
MAX_WORKERS = 8


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    # Final corpus directory expected by the corpus builder.
    corpus_dir = script_dir / "data" / "corpus" / "hotpotqa"

    # Download into a temporary staging directory beside the script.
    staging_dir = script_dir / "_hf_hotpotqa_download"

    print("=" * 80)
    print("DOWNLOADING R-SEARCH HOTPOTQA CORPUS")
    print("=" * 80)

    print(f"Script path  : {script_path}")
    print(f"Repository   : {REPO_ID}")
    print(f"Remote folder: {REPO_SUBDIR}")
    print(f"Corpus folder: {corpus_dir}")
    print(f"Max workers  : {MAX_WORKERS}")

    # --------------------------------------------------------
    # 1. Clear the previous corpus and staging directory.
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
        print(f"Failed to clear previous data: {exc}", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # 2. Download the requested repository subdirectory.
    # --------------------------------------------------------

    print("\n[2/3] Downloading from Hugging Face...")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=f"{REPO_SUBDIR}/*",
            local_dir=str(staging_dir),
            max_workers=MAX_WORKERS,
        )

    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            f"Partial download, if any, is located at: {staging_dir}",
            file=sys.stderr,
        )
        print(
            "Run the script again to clear the partial download and retry.",
            file=sys.stderr,
        )
        return 1

    # --------------------------------------------------------
    # 3. Move the downloaded corpus to its final destination.
    # --------------------------------------------------------

    print("\n[3/3] Preparing final corpus directory...")

    downloaded_corpus = staging_dir / REPO_SUBDIR

    if not downloaded_corpus.is_dir():
        print(
            f"Expected corpus directory was not found: {downloaded_corpus}",
            file=sys.stderr,
        )
        return 1

    # Validate that at least one source shard was downloaded.
    shard_files = sorted(downloaded_corpus.glob("*/wiki_*.bz2"))

    if not shard_files:
        print(
            "No files matching */wiki_*.bz2 were downloaded.",
            file=sys.stderr,
        )
        return 1

    try:
        corpus_dir.parent.mkdir(parents=True, exist_ok=True)

        # Move the downloaded directory into the expected location.
        shutil.move(str(downloaded_corpus), str(corpus_dir))

        # Remove the now-unneeded staging directory.
        shutil.rmtree(staging_dir, ignore_errors=True)

    except OSError as exc:
        print(f"Failed to prepare final corpus: {exc}", file=sys.stderr)
        return 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    final_shards = sorted(corpus_dir.glob("*/wiki_*.bz2"))

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(f"Corpus directory: {corpus_dir}")
    print(f"Shard files      : {len(final_shards)}")

    total_bytes = sum(path.stat().st_size for path in final_shards)

    print(f"Total size       : {total_bytes:,} bytes")
    print(f"Total size       : {total_bytes / (1024 ** 3):.2f} GiB")

    print("\nYou can now build the corpus with:")
    print(
        f'python build_hotpotqa_corpus.py --corpus-dir "{corpus_dir}"'
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())