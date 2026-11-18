
#!/usr/bin/env python3
"""
============================================================
2WikiMultiHopQA Corpus Builder

Build a retrieval corpus from the downloaded R-Search
2WikiMultiHopQA dataset.

Input:
    data/corpus/2wikimultihopqa/train.json
    data/corpus/2wikimultihopqa/dev.json
    data/corpus/2wikimultihopqa/test.json

Output:
    data/corpus/2wikimultihopqa/2wikimultihopqa_corpus.jsonl

Each output document contains:
    - id
    - title
    - content
    - document

R-Search document format:
    "<title>"
    <content>

The E5 prefix ("passage:") is NOT stored here.
It should be added during embedding.

Corpus construction follows R-Search's
load_corpus() implementation.
============================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ============================================================
# CONFIG
# ============================================================

INPUT_FILES = (
    "train.json",
    "dev.json",
    "test.json",
)

OUTPUT_FILENAME = "2wikimultihopqa_corpus.jsonl"


# ============================================================
# FORMAT DOCUMENT
# ============================================================

def format_rsearch_document(
    title: str,
    content: str,
) -> str:
    """
    Format a document using R-Search's representation:

        "<title>"
        <content>

    Do not add the E5 prefix here.
    """

    return f'"{title}"\n{content}'


# ============================================================
# PROCESS CONTEXT
# ============================================================

def process_context(
    item: dict[str, Any],
) -> list[tuple[str, str]]:
    """
    Follow R-Search's load_corpus() logic:

        for title, sentences in item["context"]:
            para = " ".join(sentences)
            data[para] = title

    Return (content, title) pairs.
    """

    documents = []

    for context in item["context"]:
        if not isinstance(context, (list, tuple)) or len(context) != 2:
            raise ValueError(
                f"Invalid context entry: {context!r}"
            )

        title, sentences = context

        if not isinstance(title, str):
            raise TypeError(
                f"Invalid title type: {type(title)}"
            )

        if not isinstance(sentences, list):
            raise TypeError(
                f"Invalid sentences type for {title}: "
                f"{type(sentences)}"
            )

        # Match R-Search exactly:
        # join sentences using a single space.
        content = " ".join(sentences)

        documents.append((content, title))

    return documents


# ============================================================
# BUILD CORPUS
# ============================================================

def build_2wikimultihopqa_corpus(
    corpus_dir: Path,
    output_path: Path,
) -> int:
    """
    Reproduce R-Search's corpus construction:

        1. Load train, dev and test.
        2. Iterate over item["context"].
        3. Join sentences using " ".
        4. Deduplicate using paragraph content as key.
        5. Preserve the associated title.
        6. Assign sequential IDs.

    Stream output to JSONL to reduce RAM usage.

    Note:
        The deduplication dictionary is retained in memory
        to reproduce R-Search's original behavior.
    """

    corpus = {}

    total_contexts = 0

    for split in INPUT_FILES:
        input_path = corpus_dir / split

        if not input_path.is_file():
            raise FileNotFoundError(
                f"Required input file not found: {input_path}"
            )

        print(f"Loading {split}...")

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as source:
            data = json.load(source)

        print(f"    Examples: {len(data):,}")

        for item_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Invalid item in {split}, "
                    f"index {item_index}"
                )

            documents = process_context(item)

            for content, title in documents:
                # Match R-Search exactly:
                # dictionary key = paragraph content.
                corpus[content] = title

                total_contexts += 1

        print(
            f"    Unique documents so far: {len(corpus):,}"
        )

        del data

    if not corpus:
        raise RuntimeError(
            "No documents were found in the input files."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for document_id, (content, title) in enumerate(
                corpus.items()
            ):
                document = format_rsearch_document(
                    title=title,
                    content=content,
                )

                record = {
                    "id": document_id,
                    "title": title,
                    "content": content,
                    "document": document,
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    ) + "\n"
                )

        # Replace final output only after successful writing.
        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"\nTotal contexts: {total_contexts:,}")
    print(f"Unique documents: {len(corpus):,}")
    print(
        f"Removed duplicates: "
        f"{total_contexts - len(corpus):,}"
    )

    return len(corpus)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Build the R-Search-format "
            "2WikiMultiHopQA corpus."
        )
    )

    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=(
            script_dir
            / "data"
            / "corpus"
            / "2wikimultihopqa"
        ),
        help=(
            "Directory containing train.json, "
            "dev.json and test.json."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to "
            "2wikimultihopqa_corpus.jsonl "
            "inside corpus-dir."
        ),
    )

    args = parser.parse_args()

    corpus_dir = args.corpus_dir.expanduser().resolve()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else corpus_dir / OUTPUT_FILENAME
    )

    print("=" * 80)
    print("BUILDING 2WIKIMULTIHOPQA RETRIEVAL CORPUS")
    print("=" * 80)

    print(f"Corpus directory: {corpus_dir}")
    print(f"Output file     : {output_path}")

    if not corpus_dir.is_dir():
        print(
            f"Corpus directory does not exist: {corpus_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        total_documents = build_2wikimultihopqa_corpus(
            corpus_dir=corpus_dir,
            output_path=output_path,
        )

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"Failed to build corpus: {exc}",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 80)
    print("CORPUS COMPLETE")
    print("=" * 80)

    print(f"Output file      : {output_path}")
    print(f"Total documents  : {total_documents:,}")

    print("\nDocument format:")
    print('    "<title>"')
    print("    <content>")

    print("\nOutput fields:")
    print("    id")
    print("    title")
    print("    content")
    print("    document")

    # Display one example without loading the entire output.
    print("\nExample document:")

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as output_file:
        first_line = output_file.readline()

    if first_line:
        example = json.loads(first_line)

        print(f"\nID: {example['id']}")
        print(f"Title: {example['title']}")
        print(f"Content: {example['content'][:300]}")
        print(f"Document:\n{example['document'][:400]}")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
