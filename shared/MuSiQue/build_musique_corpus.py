# ============================================================
# MuSiQue Corpus Builder
#
# Build a retrieval corpus from MuSiQue v1.0.
#
# Each MuSiQue paragraph becomes ONE retrieval document.
#
# R-Search document format:
#
# "<title>"
# <paragraph_text>
#
# E5 prefix ("passage:") is NOT stored here.
# It should be added during embedding:
#
#     passage: "<title>"
#     <paragraph_text>
#
# The dataset is downloaded / located through dataset.py.
# ============================================================

from pathlib import Path
import json
from itertools import chain

from dataset import download_musique


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FILENAME = "musique_corpus.jsonl"


REQUIRED_FILES = [
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_test.jsonl",
    "musique_full_v1.0_train.jsonl",
    "musique_full_v1.0_dev.jsonl",
    "musique_full_v1.0_test.jsonl",
]


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path: Path):
    """
    Load a JSONL file.

    Each line corresponds to one MuSiQue question.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return [
            json.loads(line.strip())
            for line in f
            if line.strip()
        ]


# ============================================================
# LOAD ALL MUSIQUE DATA
# ============================================================

def load_all_musique(dataset_dir: Path):
    """
    Load train/dev/test from both:

        - musique_ans_v1.0_*.jsonl
        - musique_full_v1.0_*.jsonl

    This follows the corpus construction logic used by R-Search.
    """

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train = (
        load_jsonl(
            dataset_dir
            / "musique_ans_v1.0_train.jsonl"
        )
        +
        load_jsonl(
            dataset_dir
            / "musique_full_v1.0_train.jsonl"
        )
    )

    # --------------------------------------------------------
    # Dev
    # --------------------------------------------------------

    dev = (
        load_jsonl(
            dataset_dir
            / "musique_ans_v1.0_dev.jsonl"
        )
        +
        load_jsonl(
            dataset_dir
            / "musique_full_v1.0_dev.jsonl"
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test = (
        load_jsonl(
            dataset_dir
            / "musique_ans_v1.0_test.jsonl"
        )
        +
        load_jsonl(
            dataset_dir
            / "musique_full_v1.0_test.jsonl"
        )
    )

    return train, dev, test


# ============================================================
# BUILD R-SEARCH DOCUMENT
# ============================================================

def format_rsearch_document(
    title: str,
    content: str,
):
    """
    Format one paragraph exactly according to the
    document representation used by R-Search.

    Format:

        "<title>"
        <paragraph_text>

    Important:
        The E5 prefix "passage:" is NOT included here.

    It is added later during E5 encoding.
    """

    return (
        f'"{title}"\n'
        f"{content}"
    )


# ============================================================
# BUILD CORPUS
# ============================================================

def build_musique_corpus(
    dataset_dir: Path,
):
    """
    Build the MuSiQue retrieval corpus.

    Each paragraph is treated as one independent
    retrieval document.

    R-Search document format:

        "<title>"
        <paragraph_text>

    Duplicate paragraphs are removed using:

        title + " " + paragraph_text
    """

    train, dev, test = load_all_musique(
        dataset_dir
    )

    all_samples = chain(
        train,
        dev,
        test,
    )

    corpus = []

    seen = set()

    document_id = 0

    # --------------------------------------------------------
    # Iterate over every MuSiQue sample
    # --------------------------------------------------------

    for sample in all_samples:

        # ----------------------------------------------------
        # Every paragraph becomes one retrieval document
        # ----------------------------------------------------

        for paragraph in sample["paragraphs"]:

            title = paragraph["title"]
            content = paragraph["paragraph_text"]

            # ------------------------------------------------
            # Deduplication key
            #
            # Same logic as R-Search:
            #
            # title + paragraph_text
            # ------------------------------------------------

            dedup_key = (
                title
                + " "
                + content
            )

            if dedup_key in seen:
                continue

            seen.add(dedup_key)

            # ------------------------------------------------
            # R-Search document representation
            #
            # "<title>"
            # <paragraph_text>
            # ------------------------------------------------

            document = format_rsearch_document(
                title=title,
                content=content,
            )

            corpus.append(
                {
                    "id": document_id,
                    "title": title,
                    "content": content,
                    "document": document,
                }
            )

            document_id += 1

    return corpus


# ============================================================
# SAVE CORPUS
# ============================================================

def save_corpus(
    corpus,
    output_path: Path,
):
    """
    Save corpus as JSONL.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for document in corpus:

            f.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BUILDING MuSiQue RETRIEVAL CORPUS")
    print("=" * 80)

    # --------------------------------------------------------
    # Download / locate MuSiQue using existing utility
    # --------------------------------------------------------

    dataset_dir = download_musique()

    print("\nDataset directory:")
    print(f"  {dataset_dir}")

    # --------------------------------------------------------
    # Verify required files
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("VERIFYING DATASET FILES")
    print("=" * 80)

    missing_files = []

    for filename in REQUIRED_FILES:

        path = dataset_dir / filename

        if path.exists():

            print(f"[OK] {filename}")

        else:

            print(f"[MISSING] {filename}")
            missing_files.append(filename)

    if missing_files:

        raise FileNotFoundError(
            "\nMissing MuSiQue files:\n"
            + "\n".join(
                f"  - {filename}"
                for filename in missing_files
            )
        )

    # --------------------------------------------------------
    # Build corpus
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("BUILDING CORPUS")
    print("=" * 80)

    corpus = build_musique_corpus(
        dataset_dir
    )

    print(
        f"\nTotal unique passages: "
        f"{len(corpus):,}"
    )

    # --------------------------------------------------------
    # Save next to dataset
    # --------------------------------------------------------

    output_path = (
        dataset_dir
        / OUTPUT_FILENAME
    )

    save_corpus(
        corpus,
        output_path,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CORPUS COMPLETE")
    print("=" * 80)

    print(
        f"Output:\n"
        f"  {output_path}"
    )

    print(
        f"\nNumber of passages:\n"
        f"  {len(corpus):,}"
    )

    print(
        "\nDocument format:\n"
        '  "<title>"'
        "\n"
        "  <paragraph_text>"
    )

    print(
        "\nE5 encoding format:\n"
        '  passage: "<title>"'
        "\n"
        "  <paragraph_text>"
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXAMPLE PASSAGES")
    print("=" * 80)

    for document in corpus[:3]:

        print(
            f"\nID: {document['id']}\n"
            f"{document['document']}"
        )

    print("\n" + "=" * 80)


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    main()