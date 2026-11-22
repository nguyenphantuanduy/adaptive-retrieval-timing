"""Score baseline completions after removing Qwen text-control tokens."""
import argparse
import json
import re
from pathlib import Path

from REVIVE_training.evaluate_art_pure import get_output_file, load_results
from REVIVE_training.utils import acc_score, compute_exact, f1_scorer

# Text/chat control tokens from Qwen2.5-3B-Instruct's tokenizer_config.json.
TEXT_CONTROL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
TEXT_CONTROL_PATTERN = re.compile("|".join(map(re.escape, TEXT_CONTROL_TOKENS)))


def clean_prediction(prediction):
    return TEXT_CONTROL_PATTERN.sub("", prediction)


def score_baseline(results):
    predictions = [clean_prediction(row["pure_art_answer"]) for row in results]
    answers = [row["golden_answers"] for row in results]
    return {
        "num_samples": len(results),
        "metrics": {
            "Acc": acc_score(predictions, answers),
            "F1": f1_scorer(predictions, answers),
            "EM": compute_exact(predictions, answers),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filename", type=Path, required=True, help="Baseline JSONL to score.")
    args = parser.parse_args()

    try:
        results = load_results(args.filename)
    except FileNotFoundError:
        parser.error(f"baseline result not found: {args.filename}")
    if not results:
        parser.error("input contains no baseline predictions")
    metrics = score_baseline(results)
    output = get_output_file(args.filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        parser.error(f"metrics output already exists: {output}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
