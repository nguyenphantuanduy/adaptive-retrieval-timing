"""Generate the question-only Qwen baseline."""
from REVIVE_training.baseline_generation import run_baseline


if __name__ == "__main__":
    run_baseline("no_rag.jsonl")
