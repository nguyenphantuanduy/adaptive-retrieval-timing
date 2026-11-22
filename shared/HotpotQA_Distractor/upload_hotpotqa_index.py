
import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = (
    "adaptive-retrieval-timing/HotpotQA-ART-E5-Retrieval-Index"
)

FILES_TO_UPLOAD = [
    "e5_Flat.index",
    "hotpotqa_corpus.jsonl",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./HotpotQA-ART-E5-Retrieval-Index",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {input_dir}"
        )

    for filename in FILES_TO_UPLOAD:
        path = input_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        print(
            f"{filename}: "
            f"{path.stat().st_size / 1024**3:.2f} GiB"
        )

    api = HfApi()

    # Create the repo only if it does not already exist.
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    for filename in FILES_TO_UPLOAD:
        path = input_dir / filename

        print(f"\nUploading {filename}...")

        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=filename,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload {filename}",
        )

        print(f"Uploaded {filename}")

    print("\nUpload completed.")
    print(
        f"https://huggingface.co/datasets/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
