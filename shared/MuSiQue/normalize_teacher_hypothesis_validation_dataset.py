"""Remove verbatim question prefixes from teacher-generated validation targets.

The source and destination are the same Hugging Face dataset repository. Each
config is downloaded in full, normalized locally, verified, then uploaded back
to the same config. Run from the repository root:

    .\\.venv\\Scripts\\python.exe \
        shared\\MuSiQue\\data\\musique\\normalize_teacher_hypothesis_validation_dataset.py

Authenticate with Hugging Face first (for example, ``huggingface-cli login``).
Use ``--dry-run`` to inspect counts without changing the remote dataset.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from datasets import Dataset, DatasetDict, load_dataset


DATASET_REPO = "adaptive-retrieval-timing/teacher-generated-hypothesis-validation"
CONFIGS = ("2-hop", "3-hop", "4-hop")


def has_verbatim_question_prefix(sample: Mapping[str, object]) -> bool:
    """Return whether output starts with the exact question after leading space."""
    question = sample.get("question")
    teacher_output = sample.get("teacher_generation_output")

    return (
        isinstance(question, str)
        and bool(question)
        and isinstance(teacher_output, str)
        and teacher_output.lstrip().startswith(question)
    )


def remove_verbatim_question_prefix(
    sample: Mapping[str, object],
) -> dict[str, object]:
    """Remove only an exact question prefix; do not alter any other output."""
    teacher_output = sample.get("teacher_generation_output")

    if not has_verbatim_question_prefix(sample):
        return {"teacher_generation_output": teacher_output}

    question = sample["question"]
    assert isinstance(question, str)
    assert isinstance(teacher_output, str)

    # Drop leading output whitespace, then the question itself, and finally the
    # separator whitespace between the repeated question and the actual answer.
    normalized_output = teacher_output.lstrip()[len(question) :].lstrip()

    return {"teacher_generation_output": normalized_output}


def count_question_prefixes(dataset: Dataset) -> int:
    return sum(has_verbatim_question_prefix(sample) for sample in dataset)


def normalize_config(config_name: str, dry_run: bool) -> tuple[int, int]:
    """Normalize and optionally publish every split belonging to one config."""
    print("\n" + "=" * 80)
    print(f"LOADING CONFIG: {config_name}")
    print("=" * 80)

    source = load_dataset(DATASET_REPO, name=config_name)
    normalized = DatasetDict()
    before_count = 0
    after_count = 0

    for split_name, split_dataset in source.items():
        count_before = count_question_prefixes(split_dataset)
        cleaned_split = split_dataset.map(remove_verbatim_question_prefix)
        count_after = count_question_prefixes(cleaned_split)

        if count_after:
            raise RuntimeError(
                f"{config_name}/{split_name}: {count_after} repeated-question "
                "prefixes remain after normalization."
            )

        normalized[split_name] = cleaned_split
        before_count += count_before
        after_count += count_after

        print(
            f"{split_name}: removed {count_before:,} repeated-question prefixes "
            f"from {len(split_dataset):,} samples"
        )

    if dry_run:
        print(f"DRY RUN: {config_name} was not uploaded.")
    else:
        print(f"Uploading normalized {config_name} back to {DATASET_REPO}...")
        normalized.push_to_hub(
            DATASET_REPO,
            config_name=config_name,
            commit_message=(
                f"Normalize repeated question prefixes in {config_name} "
                "teacher outputs"
            ),
        )
        print(f"Uploaded {config_name}.")

    return before_count, after_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalize and verify locally without uploading to Hugging Face.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("NORMALIZE TEACHER HYPOTHESIS-VALIDATION DATASET")
    print("=" * 80)
    print(f"Dataset: {DATASET_REPO}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPLOAD'}")

    total_before = 0
    total_after = 0

    for config_name in CONFIGS:
        before_count, after_count = normalize_config(config_name, args.dry_run)
        total_before += before_count
        total_after += after_count

    print("\n" + "=" * 80)
    print("NORMALIZATION COMPLETE")
    print("=" * 80)
    print(f"Repeated prefixes before normalization: {total_before:,}")
    print(f"Repeated prefixes after normalization:  {total_after:,}")


if __name__ == "__main__":
    main()
