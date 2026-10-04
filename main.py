
from shared.MuSiQue.rsearch_musique_test_dataset import load_rsearch_musique


def main():

    # ========================================================
    # LOAD R-SEARCH MuSiQue TEST SET
    # ========================================================

    dataset = load_rsearch_musique()

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    print("\n" + "=" * 80)
    print("DATASET TEST")
    print("=" * 80)

    print(
        f"Number of samples: {len(dataset)}"
    )

    # ========================================================
    # SHOW FIRST SAMPLE
    # ========================================================

    print("\n" + "=" * 80)
    print("FIRST SAMPLE")
    print("=" * 80)

    sample = dataset[0]

    print(
        f"ID:\n"
        f"  {sample.get('id')}"
    )

    print(
        f"\nQuestion:\n"
        f"  {sample.get('question')}"
    )

    print(
        f"\nGolden answers:\n"
        f"  {sample.get('golden_answers')}"
    )

    # ========================================================
    # SHOW ALL FIELDS
    # ========================================================

    print("\n" + "=" * 80)
    print("AVAILABLE FIELDS")
    print("=" * 80)

    print(
        list(sample.keys())
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    assert "question" in sample
    assert "golden_answers" in sample

    assert isinstance(
        sample["question"],
        str,
    )

    assert isinstance(
        sample["golden_answers"],
        list,
    )

    print("\n" + "=" * 80)
    print("TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()