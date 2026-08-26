from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
SPLIT = "validation"


# ============================================================
# TEST — LOAD HOTPOTQA DISTRACTOR
# ============================================================

def test_load_hotpotqa():

    print("=" * 80)
    print("TEST — LOAD HOTPOTQA DISTRACTOR")
    print("=" * 80)

    print(f"Dataset : {DATASET_NAME}")
    print(f"Config  : {DATASET_CONFIG}")
    print(f"Split   : {SPLIT}")
    print("Mode    : streaming")
    print()

    # --------------------------------------------------------
    # Load dataset in streaming mode
    # --------------------------------------------------------

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=SPLIT,
        streaming=True,
    )

    print("✓ Dataset loaded")
    print()

    # --------------------------------------------------------
    # Get first sample
    # --------------------------------------------------------

    sample = next(iter(dataset))

    # --------------------------------------------------------
    # Print raw sample
    # --------------------------------------------------------

    print("=" * 80)
    print("FIRST VALIDATION SAMPLE")
    print("=" * 80)

    print(sample)

    print()
    print("=" * 80)
    print("SAMPLE KEYS")
    print("=" * 80)

    for key, value in sample.items():
        print(
            f"{key}: "
            f"type={type(value).__name__}"
        )

    print()
    print("=" * 80)
    print("BASIC INFORMATION")
    print("=" * 80)

    print(f"ID     : {sample['id']}")
    print(f"Type   : {sample['type']}")
    print(f"Level  : {sample['level']}")
    print(f"Question: {sample['question']}")
    print(f"Answer : {sample['answer']}")

    print()
    print("Supporting facts:")
    print(sample["supporting_facts"])

    print()
    print("Context:")
    print(sample["context"])

    print()
    print("✓ HotpotQA validation sample loaded successfully")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_load_hotpotqa()