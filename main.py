import json
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("teacher_hypothesis_demonstrations.jsonl")


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE.resolve()}")
        return

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Error] Invalid JSON at line {idx}: {e}")
                continue

            print("=" * 100)
            print(f"Sample: {idx}")
            print(f"Hop: {sample.get('hop', 'N/A')}")
            print(f"Question: {sample.get('question', 'N/A')}")
            print("-" * 100)
            print("Teacher Generation Output:")
            print(sample.get("teacher_generation_output", "N/A"))
            print("=" * 100)
            print()


if __name__ == "__main__":
    main()