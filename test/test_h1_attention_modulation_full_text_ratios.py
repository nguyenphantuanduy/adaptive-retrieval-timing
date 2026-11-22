"""Print H1 full_text inputs for every valid document ratio of one 2-hop sample.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe -m test.test_h1_attention_modulation_full_text_ratios

Optionally select another validation row:

    .\\.venv\\Scripts\\python.exe -m test.test_h1_attention_modulation_full_text_ratios --index 5
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.build_lm_loss_samples import (
    create_lm_loss_samples,
)
from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.model_input import (
    create_model_input_for_lm_loss,
)


DATASET_REPO = "adaptive-retrieval-timing/teacher-generated-hypothesis-validation"
CONFIG_NAME = "2-hop"
SPLIT = "validation"
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the 2-hop validation sample to inspect.",
    )
    args = parser.parse_args()

    print("Loading dataset and tokenizer...")
    dataset = load_dataset(
        DATASET_REPO,
        name=CONFIG_NAME,
        split=SPLIT,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)

    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"--index must be in [0, {len(dataset) - 1}], got {args.index}."
        )

    sample = dataset[args.index]
    lm_samples = create_lm_loss_samples(sample)

    print("=" * 100)
    print("H1 ATTENTION MODULATION: FULL TEXT BY DOCUMENT RATIO")
    print("=" * 100)
    print(f"Dataset: {DATASET_REPO} ({CONFIG_NAME}/{SPLIT})")
    print(f"Sample index: {args.index}")
    print(f"Sample id: {sample.get('id', '<missing>')}")
    print(f"Question: {sample['question']}")
    print(f"Valid ratios: {list(lm_samples)}")

    for ratio, lm_sample in lm_samples.items():
        model_input = create_model_input_for_lm_loss(lm_sample, tokenizer)
        num_positive, num_negative = ratio

        print("\n" + "=" * 100)
        print(
            f"RATIO {num_positive}:{num_negative} "
            f"({num_positive} supporting, {num_negative} non-supporting documents)"
        )
        print(f"Input shape: {tuple(model_input['input_ids'].shape)}")
        print("-" * 100)
        print(model_input["full_text"])
        print("=" * 100)


if __name__ == "__main__":
    main()
