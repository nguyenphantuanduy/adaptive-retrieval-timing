#!/usr/bin/env python3
"""
Ensure the R-Search HotpotQA corpus exists in a data folder beside this script.

Destination:
    <directory-containing-this-script>/data/corpus/hotpotqa/

This intentionally does NOT search for a Git repository root, because the
script may live in a nested folder or one of several project branches.
Requires: pip install requests
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

try:
    import requests
except ImportError:
    print("Missing dependency: requests")
    print(f"Install it with: {sys.executable} -m pip install requests")
    raise SystemExit(1)


REPO_ID = "qingfei1/R-Search_datasets"
REPO_SUBDIR = "data/corpus/hotpotqa"
HF_API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/{REPO_SUBDIR}"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"


def list_remote_files() -> list[str]:
    """Return paths relative to REPO_SUBDIR from the Hugging Face API."""
    files: list[str] = []
    url = HF_API_URL
    params = {"recursive": "true", "expand": "false", "limit": 1000}

    while url:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        items = response.json()

        for item in items:
            if item.get("type") != "file":
                continue
            full_path = item.get("path", "")
            prefix = REPO_SUBDIR.rstrip("/") + "/"
            if full_path.startswith(prefix):
                files.append(full_path[len(prefix):])

        url = response.links.get("next", {}).get("url")
        params = None

    return sorted(set(files))


def download_file(relative_path: str, destination: Path) -> None:
    """Stream a file directly to its destination, using a temporary .part file."""
    remote_path = f"{REPO_SUBDIR}/{relative_path}"
    url = f"{HF_RESOLVE_BASE}/{quote(remote_path, safe='/')}?download=true"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")

    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with temporary_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {relative_path}: {downloaded * 100 / total:6.2f}% "
                        f"({downloaded:,}/{total:,} bytes)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  {relative_path}: {downloaded:,} bytes",
                        end="",
                        flush=True,
                    )

    print()
    if not temporary_path.exists() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is empty: {relative_path}")

    temporary_path.replace(destination)


def main() -> int:
    script_path = Path(__file__).resolve()
    # Save data/ beside this script, e.g.:
    # HotpotQA_Distractor/download_rsearch_hotpotqa_corpus.py
    # -> HotpotQA_Distractor/data/corpus/hotpotqa/
    parent_dir = script_path.parent
    data_dir = parent_dir / "data"
    corpus_dir = data_dir / "corpus" / "hotpotqa"

    print(f"Script path  : {script_path}")
    print(f"Script folder: {script_path.parent}")
    print(f"Parent folder: {parent_dir}")
    print(f"Data folder  : {data_dir}")
    print(f"Corpus folder: {corpus_dir}")

    try:
        remote_files = list_remote_files()
    except requests.RequestException as exc:
        print(f"Could not list files on Hugging Face: {exc}", file=sys.stderr)
        return 1

    if not remote_files:
        print(
            "No files found in the specified Hugging Face folder. "
            "Check repository access or the folder path.",
            file=sys.stderr,
        )
        return 1

    corpus_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for relative_path in remote_files:
        local_path = corpus_dir / relative_path
        if not local_path.is_file() or local_path.stat().st_size == 0:
            pending.append((relative_path, local_path))

    if not pending:
        print(
            f"Corpus is already present: all {len(remote_files)} remote files "
            "exist and are non-empty. Nothing to download."
        )
        return 0

    print(f"Need to download {len(pending)} of {len(remote_files)} files.")
    for relative_path, local_path in pending:
        print(f"Downloading {relative_path} ...")
        try:
            download_file(relative_path, local_path)
        except (requests.RequestException, OSError, RuntimeError) as exc:
            print(f"\nDownload failed for {relative_path}: {exc}", file=sys.stderr)
            print("Run the script again to retry; completed files are retained.")
            return 1

    print(f"\nDone. Corpus saved to: {corpus_dir}")
    print(f"Files checked: {len(remote_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
