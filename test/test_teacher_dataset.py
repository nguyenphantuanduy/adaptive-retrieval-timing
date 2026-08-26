from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = (
    "adaptive-retrieval-timing/"
    "teacher-generated-hypothesis-validation"
)


# ============================================================
# LOAD + TEST
# ============================================================

def main():

    print("=" * 80)
    print("TEST HUGGING FACE DATASET")
    print("=" * 80)

    for hop in (2, 3, 4):

        config_name = f"{hop}-hop"

        print()
        print("=" * 80)
        print(f"LOADING {config_name}")
        print("=" * 80)

        dataset = load_dataset(
            DATASET_NAME,
            config_name,
        )

        validation = dataset["validation"]

        print()
        print(f"Config       : {config_name}")
        print(f"Split        : validation")
        print(f"Num samples  : {len(validation):,}")

        print()
        print("-" * 80)
        print("SAMPLE 0")
        print("-" * 80)

        sample = validation[0]

        for key, value in sample.items():

            print()
            print(f"[{key}]")

            if isinstance(value, str) and len(value) > 2000:

                print(
                    value[:2000]
                    + "\n... [truncated]"
                )

            else:

                print(value)

    print()
    print("=" * 80)
    print("✓ ALL HOP DATASETS LOADED SUCCESSFULLY")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()