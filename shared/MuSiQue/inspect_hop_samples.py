"""
Pretty-print one MuSiQue-Ans sample for each hop count: 2-hop, 3-hop, and 4-hop.
Select samples using zero-based indices within each hop-count group.
Print only gold supporting documents while preserving their original indices.

Run from the repository root:
    python -m shared.MuSiQue.inspect_hop_samples
"""

from shared.MuSiQue.load_dataset import load_musique_ans


# ============================================================
# Configuration
# ============================================================

# Zero-based index within samples of each hop count.
# 0 = first sample, 1 = second sample, and so on.
hop_2_idx = 1
hop_3_idx = 2
hop_4_idx = 0

DEFAULT_SPLIT = "dev"
PARAGRAPH_WIDTH = 100

HOP_SAMPLE_INDICES = {
    2: hop_2_idx,
    3: hop_3_idx,
    4: hop_4_idx,
}


# ============================================================
# Text formatting
# ============================================================

def wrap_text(text, width=PARAGRAPH_WIDTH, indent="    "):
    """Wrap text while keeping a readable indentation."""
    import textwrap

    return textwrap.fill(
        str(text),
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


# ============================================================
# Sample selection
# ============================================================

def get_hop_count(sample):
    """Determine hop count from question decomposition."""
    decomposition = sample.get("question_decomposition")

    if not isinstance(decomposition, list):
        return None

    return len(decomposition)


def find_sample_by_hop(samples, hop_count, sample_idx):
    """
    Return the sample at the specified zero-based index among
    samples with the requested hop count.
    """
    if sample_idx < 0:
        raise ValueError(
            f"Sample index must be >= 0, got {sample_idx} "
            f"for {hop_count}-hop samples."
        )

    matching_samples = [
        sample
        for sample in samples
        if get_hop_count(sample) == hop_count
    ]

    if sample_idx >= len(matching_samples):
        print(
            f"\n[NOT FOUND] {hop_count}-hop sample index {sample_idx} "
            f"is out of range. "
            f"Found {len(matching_samples)} samples with "
            f"{hop_count} hops in the selected split."
        )
        return None

    return matching_samples[sample_idx]


# ============================================================
# Sample printing
# ============================================================

def print_sample(sample, hop_count, split, sample_idx):
    line = "=" * 100
    thin_line = "-" * 100

    print("\n" + line)
    print(
        f"  MuSiQue-Ans | {hop_count}-HOP SAMPLE "
        f"| SAMPLE INDEX: {sample_idx} | SPLIT: {split.upper()}"
    )
    print(line)

    print(f"ID         : {sample.get('id', 'N/A')}")
    print(f"Answer     : {sample.get('answer', 'N/A')}")
    print(f"Answerable : {sample.get('answerable', 'N/A')}")

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------
    print("\nQUESTION")
    print(thin_line)

    print(
        wrap_text(
            sample.get("question", "N/A"),
            indent="  ",
        )
    )

    # --------------------------------------------------------
    # Question decomposition
    # Keep all steps to support manual demonstration design.
    # --------------------------------------------------------
    decomposition = sample.get("question_decomposition") or []

    print("\nQUESTION DECOMPOSITION")
    print(thin_line)

    if not decomposition:
        print("  No question decomposition available.")
    else:
        for index, step in enumerate(decomposition, start=1):
            print(f"  Step {index}: {step.get('question', 'N/A')}")
            print(
                f"    Answer                   : "
                f"{step.get('answer', 'N/A')}"
            )
            print(
                f"    Supporting paragraph index: "
                f"{step.get('paragraph_support_idx', 'N/A')}"
            )

    # --------------------------------------------------------
    # Golden documents only
    # Preserve both original list position and paragraph idx.
    # --------------------------------------------------------
    paragraphs = sample.get("paragraphs") or []

    supporting_paragraphs = [
        (original_position, paragraph)
        for original_position, paragraph in enumerate(
            paragraphs,
            start=1,
        )
        if paragraph.get("is_supporting", False)
    ]

    print(
        f"\nGOLDEN DOCUMENTS "
        f"({len(supporting_paragraphs)} supporting paragraphs)"
    )
    print(thin_line)

    if not supporting_paragraphs:
        print("  No supporting paragraphs found.")
    else:
        for original_position, paragraph in supporting_paragraphs:
            title = paragraph.get("title", "Untitled")
            paragraph_idx = paragraph.get("idx", "N/A")
            text = paragraph.get("paragraph_text", "")

            print(
                f"\n  Original position: {original_position}"
                f" | idx={paragraph_idx}"
                f" | {title}"
            )

            print(
                wrap_text(
                    text,
                    indent="      ",
                )
            )

    print("\n" + line)


# ============================================================
# Main
# ============================================================

def main():
    dataset = load_musique_ans()
    split = DEFAULT_SPLIT
    samples = dataset.get(split, [])

    if not samples:
        raise RuntimeError(
            f"The '{split}' split is empty or was not returned "
            "by load_musique_ans()."
        )

    print("\nSelecting samples by hop count and sample index...")

    found = 0

    for hop_count, sample_idx in HOP_SAMPLE_INDICES.items():
        sample = find_sample_by_hop(
            samples=samples,
            hop_count=hop_count,
            sample_idx=sample_idx,
        )

        if sample is None:
            continue

        print_sample(
            sample=sample,
            hop_count=hop_count,
            split=split,
            sample_idx=sample_idx,
        )

        found += 1

    if found == 0:
        raise RuntimeError(
            f"No requested samples were found in the '{split}' split. "
            "Check the sample indices or change DEFAULT_SPLIT."
        )


if __name__ == "__main__":
    main()