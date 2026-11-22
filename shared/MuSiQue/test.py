# ============================================================
# Test MuSiQue E5 Passage Token Length
#
# Check whether any R-Search formatted passage exceeds
# the E5 max sequence length of 512 tokens.
#
# Actual E5 input:
#
#     passage: "<title>"
#     <paragraph_text>
#
# Tokenizer:
#     intfloat/e5-base-v2
# ============================================================

from pathlib import Path
import json

from transformers import AutoTokenizer
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "intfloat/e5-base-v2"

CORPUS_PATH = Path(
    "data/musique/musique_corpus.jsonl"
)

MAX_LENGTH = 512


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print("=" * 80)
    print("LOADING E5 TOKENIZER")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    print("Tokenizer loaded.")

    return tokenizer


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus(path: Path):

    print("\n" + "=" * 80)
    print("LOADING CORPUS")
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found:\n{path.resolve()}"
        )

    corpus = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():
                corpus.append(
                    json.loads(line)
                )

    print(
        f"\nCorpus size: "
        f"{len(corpus):,} passages"
    )

    return corpus


# ============================================================
# CHECK TOKEN LENGTH
# ============================================================

def check_token_lengths(
    corpus,
    tokenizer,
):

    print("\n" + "=" * 80)
    print("CHECKING TOKEN LENGTHS")
    print("=" * 80)

    lengths = []

    over_limit = []

    max_length = 0
    max_document = None

    # --------------------------------------------------------
    # Iterate through corpus
    # --------------------------------------------------------

    for document in tqdm(
        corpus,
        desc="Tokenizing passages",
    ):

        title = document["title"]
        content = document["content"]

        # ----------------------------------------------------
        # EXACT R-SEARCH E5 INPUT
        # ----------------------------------------------------
        #
        # R-Search:
        #
        #     contents = '"' + title + '"\n' + text
        #
        # Then:
        #
        #     passage: {contents}
        #
        # ----------------------------------------------------

        e5_input = (
            f'passage: "{title}"\n'
            f"{content}"
        )

        # ----------------------------------------------------
        # Tokenize WITHOUT truncation
        #
        # Important:
        # We want the REAL length, not the truncated length.
        # ----------------------------------------------------

        token_ids = tokenizer(
            e5_input,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        token_count = len(token_ids)

        lengths.append(token_count)

        # ----------------------------------------------------
        # Track maximum
        # ----------------------------------------------------

        if token_count > max_length:

            max_length = token_count
            max_document = {
                "id": document["id"],
                "title": title,
                "content": content,
                "token_count": token_count,
            }

        # ----------------------------------------------------
        # Check > 512
        # ----------------------------------------------------

        if token_count > MAX_LENGTH:

            over_limit.append(
                {
                    "id": document["id"],
                    "title": title,
                    "token_count": token_count,
                    "over_by": token_count - MAX_LENGTH,
                }
            )

    return lengths, over_limit, max_document


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    corpus,
    lengths,
    over_limit,
    max_document,
):

    total = len(lengths)

    if total == 0:
        print("\nCorpus is empty.")
        return

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    avg_length = sum(lengths) / total

    sorted_lengths = sorted(lengths)

    median = sorted_lengths[
        total // 2
    ]

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

    over_count = len(over_limit)

    over_percentage = (
        over_count / total * 100
    )

    within_count = total - over_count

    within_percentage = (
        within_count / total * 100
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TOKEN LENGTH SUMMARY")
    print("=" * 80)

    print(
        f"\nTotal passages:"
        f" {total:,}"
    )

    print(
        f"Maximum token length:"
        f" {max(lengths):,}"
    )

    print(
        f"Minimum token length:"
        f" {min(lengths):,}"
    )

    print(
        f"Average token length:"
        f" {avg_length:.2f}"
    )

    print(
        f"Median token length:"
        f" {median:,}"
    )

    print(
        f"\nMaximum allowed:"
        f" {MAX_LENGTH}"
    )

    print(
        f"\nWithin 512 tokens:"
        f" {within_count:,}"
        f" ({within_percentage:.2f}%)"
    )

    print(
        f"Over 512 tokens:"
        f" {over_count:,}"
        f" ({over_percentage:.2f}%)"
    )

    # --------------------------------------------------------
    # Longest passage
    # --------------------------------------------------------

    if max_document is not None:

        print(
            "\n" + "=" * 80
        )
        print("LONGEST PASSAGE")
        print(
            "=" * 80
        )

        print(
            f"\nID:"
            f" {max_document['id']}"
        )

        print(
            f"Token count:"
            f" {max_document['token_count']}"
        )

        print(
            f"Title:"
            f" {max_document['title']}"
        )

        print(
            "\nContent preview:"
        )

        print(
            max_document["content"][:500]
        )

    # --------------------------------------------------------
    # Over-limit examples
    # --------------------------------------------------------

    if over_limit:

        print(
            "\n" + "=" * 80
        )
        print(
            "OVER-LIMIT PASSAGES"
        )
        print(
            "=" * 80
        )

        # Show first 20
        for item in over_limit[:20]:

            print(
                f"\nID: {item['id']}"
            )

            print(
                f"Tokens: "
                f"{item['token_count']}"
            )

            print(
                f"Over by: "
                f"{item['over_by']}"
            )

            print(
                f"Title: "
                f"{item['title']}"
            )

        if len(over_limit) > 20:

            print(
                f"\n... and "
                f"{len(over_limit) - 20:,}"
                f" more."
            )

    else:

        print(
            "\n" + "=" * 80
        )

        print(
            "RESULT: ALL PASSAGES <= 512 TOKENS"
        )

        print(
            "=" * 80
        )


# ============================================================
# MAIN
# ============================================================

def main():

    tokenizer = load_tokenizer()

    corpus = load_corpus(
        CORPUS_PATH
    )

    lengths, over_limit, max_document = (
        check_token_lengths(
            corpus,
            tokenizer,
        )
    )

    print_summary(
        corpus,
        lengths,
        over_limit,
        max_document,
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    main()