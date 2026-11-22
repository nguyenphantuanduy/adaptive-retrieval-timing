"""Count samples whose teacher output begins by repeating the question verbatim."""

from __future__ import annotations

from datasets import load_dataset


DATASET_NAME = "adaptive-retrieval-timing/teacher-generated-hypothesis-validation"
CONFIGS = ("2-hop", "3-hop", "4-hop")
SPLIT = "validation"


def output_starts_with_question(sample: dict) -> bool:
    """Check literal prefix equality, allowing only leading output whitespace."""
    question = sample.get("question")
    teacher_output = sample.get("teacher_generation_output")

    if not isinstance(question, str) or not question:
        return False

    if not isinstance(teacher_output, str):
        return False

    return teacher_output.lstrip().startswith(question)


def main() -> None:
    print("=" * 80)
    print("TEACHER OUTPUTS THAT REPEAT THE QUESTION AT THE START")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Split:   {SPLIT}")

    total_samples = 0
    total_matches = 0

    for config_name in CONFIGS:
        dataset = load_dataset(
            DATASET_NAME,
            name=config_name,
            split=SPLIT,
        )

        matches = [
            (index, sample)
            for index, sample in enumerate(dataset)
            if output_starts_with_question(sample)
        ]

        total_samples += len(dataset)
        total_matches += len(matches)

        print("\n" + "-" * 80)
        print(f"{config_name}: {len(matches):,}/{len(dataset):,} matching samples")

        for index, sample in matches:
            sample_id = sample.get("id", f"<index={index}>")
            question = sample["question"]
            print(f"  [{index}] id={sample_id}")
            print(f"      question={question!r}")

    rate = 100 * total_matches / total_samples if total_samples else 0.0
    print("\n" + "=" * 80)
    print(f"TOTAL: {total_matches:,}/{total_samples:,} samples ({rate:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
