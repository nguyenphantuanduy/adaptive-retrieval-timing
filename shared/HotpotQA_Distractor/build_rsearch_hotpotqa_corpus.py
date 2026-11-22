#!/usr/bin/env python3
"""
============================================================
HotpotQA Corpus Builder

Build a retrieval corpus from the downloaded R-Search
HotpotQA corpus shards.

Input:
    data/corpus/hotpotqa/*/wiki_*.bz2

Output:
    data/corpus/hotpotqa/hotpotqa_corpus.jsonl

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

The source reading and text processing follow R-Search.
============================================================
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
from pathlib import Path
from typing import Any


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FILENAME = "hotpotqa_corpus.jsonl"


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
# PROCESS SOURCE RECORD
# ============================================================

def process_record(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Follow R-Search's process_line() logic.

    Original R-Search:
        {
            "id": data["id"],
            "title": data["title"],
            "text": "".join(data["text"]),
        }

    Extended output:
        {
            "id": ...,
            "title": ...,
            "content": ...,
            "document": ...
        }
    """

    title = data["title"]

    text = data["text"]

    # Match R-Search: concatenate text list without a separator.
    if isinstance(text, list):
        content = "".join(text)
    else:
        content = text

    document = format_rsearch_document(
        title=title,
        content=content,
    )

    return {
        "id": data["id"],
        "title": title,
        "content": content,
        "document": document,
    }


# ============================================================
# BUILD CORPUS
# ============================================================

def build_hotpotqa_corpus(
    corpus_dir: Path,
    output_path: Path,
) -> int:
    """
    Process all files matching R-Search's pattern:

        <corpus_dir>/*/wiki_*.bz2

    Stream records directly to JSONL to reduce RAM usage.

    No deduplication or ID reassignment is performed.
    """

    files = sorted(
        corpus_dir.glob("*/wiki_*.bz2")
    )

    if not files:
        raise FileNotFoundError(
            f"No files matching '*/wiki_*.bz2' in {corpus_dir}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        output_path.name + ".part"
    )

    total_records = 0

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for file_index, bz2_path in enumerate(
                files,
                start=1,
            ):
                file_records = 0

                print(
                    f"[{file_index}/{len(files)}] "
                    f"Processing {bz2_path.name}"
                )

                with bz2.open(
                    bz2_path,
                    "rt",
                    encoding="utf-8",
                ) as source:

                    for line_number, line in enumerate(
                        source,
                        start=1,
                    ):
                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)

                            document = process_record(data)

                        except (
                            json.JSONDecodeError,
                            KeyError,
                            TypeError,
                        ) as exc:
                            raise ValueError(
                                f"Invalid record in {bz2_path}, "
                                f"line {line_number}: {exc}"
                            ) from exc

                        output_file.write(
                            json.dumps(
                                document,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        file_records += 1
                        total_records += 1

                print(
                    f"    Records: {file_records:,}"
                )

        if total_records == 0:
            temporary_path.unlink(
                missing_ok=True
            )
            raise RuntimeError(
                "No records were found in the source files."
            )

        # Replace the final output only after successful processing.
        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise

    return total_records


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Build the R-Search-format HotpotQA corpus."
    )

    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=(
            script_dir
            / "data"
            / "corpus"
            / "hotpotqa"
        ),
        help="Directory containing the downloaded HotpotQA shards.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to hotpotqa_corpus.jsonl inside corpus-dir.",
    )

    args = parser.parse_args()

    corpus_dir = args.corpus_dir.expanduser().resolve()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else corpus_dir / OUTPUT_FILENAME
    )

    print("=" * 80)
    print("BUILDING HOTPOTQA RETRIEVAL CORPUS")
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
        total_records = build_hotpotqa_corpus(
            corpus_dir=corpus_dir,
            output_path=output_path,
        )

    except (
        OSError,
        ValueError,
        RuntimeError,
        EOFError,
    ) as exc:
        print(
            f"Failed to build corpus: {exc}",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 80)
    print("CORPUS COMPLETE")
    print("=" * 80)

    print(f"Output file     : {output_path}")
    print(f"Total documents: {total_records:,}")

    print("\nDocument format:")
    print('  "<title>"')
    print("  <content>")

    print("\nOutput fields:")
    print("  id")
    print("  title")
    print("  content")
    print("  document")

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