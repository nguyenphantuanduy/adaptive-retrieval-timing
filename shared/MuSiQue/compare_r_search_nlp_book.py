# ============================================================
# MuSiQue Dataset Exact Comparison
#
# Compare:
#
#   data/musique/
#       vs
#   data/musique_rsearch/
#
# Comparison:
#   1. Exact file size in bytes
#   2. Exact line count
#   3. SHA-256
#
# IMPORTANT:
#   SHA-256 equality means the two files are byte-for-byte
#   identical with overwhelming cryptographic confidence.
#
# This script DOES NOT:
#   - download anything
#   - modify anything
#   - use Hugging Face cache
#   - use temporary files
# ============================================================

from pathlib import Path
import hashlib
import csv
import sys


# ============================================================
# CONFIG
# ============================================================

REQUIRED_FILES = [
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_dev.jsonl",
    "musique_ans_v1.0_test.jsonl",
    "musique_full_v1.0_train.jsonl",
    "musique_full_v1.0_dev.jsonl",
    "musique_full_v1.0_test.jsonl",
]


# ============================================================
# PATH
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

LOCAL_DATASET_DIR = (
    THIS_DIR
    / "data"
    / "musique"
)

RSEARCH_DATASET_DIR = (
    THIS_DIR
    / "data"
    / "musique_rsearch"
)

REPORT_PATH = (
    THIS_DIR
    / "data"
    / "musique_comparison.csv"
)


# ============================================================
# FORMAT SIZE
# ============================================================

def format_size(size_bytes):
    """
    Convert bytes to human-readable format.
    """

    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"

    if size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.2f} MB"

    return f"{size_bytes / 1024**3:.2f} GB"


# ============================================================
# SHA-256
# ============================================================

def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    """
    Calculate SHA-256 hash of a file.

    The file is processed in chunks, so even very large
    JSONL files do not need to be loaded into RAM.
    """

    sha256 = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# ============================================================
# EXACT BYTE COUNT
# ============================================================

def get_file_size(path):
    """
    Return exact file size in bytes.
    """

    return path.stat().st_size


# ============================================================
# EXACT LINE COUNT
# ============================================================

def count_lines(path):
    """
    Count exact number of newline-separated records.

    Files are opened in binary mode so Python does not perform
    newline translation.
    """

    count = 0

    with open(
        path,
        "rb",
    ) as file:

        for _ in file:
            count += 1

    return count


# ============================================================
# COMPARE ONE FILE
# ============================================================

def compare_file(filename):

    local_path = (
        LOCAL_DATASET_DIR
        / filename
    )

    rsearch_path = (
        RSEARCH_DATASET_DIR
        / filename
    )

    # --------------------------------------------------------
    # Initial result
    # --------------------------------------------------------

    result = {
        "filename": filename,

        "local_exists": local_path.is_file(),
        "rsearch_exists": rsearch_path.is_file(),

        "local_size_bytes": "",
        "rsearch_size_bytes": "",

        "local_lines": "",
        "rsearch_lines": "",

        "local_sha256": "",
        "rsearch_sha256": "",

        "size_identical": False,
        "lines_identical": False,
        "sha256_identical": False,
        "byte_for_byte_identical": False,
    }

    # --------------------------------------------------------
    # Missing file
    # --------------------------------------------------------

    if not local_path.is_file():
        return result

    if not rsearch_path.is_file():
        return result

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    local_size = get_file_size(
        local_path
    )

    rsearch_size = get_file_size(
        rsearch_path
    )

    result["local_size_bytes"] = (
        local_size
    )

    result["rsearch_size_bytes"] = (
        rsearch_size
    )

    result["size_identical"] = (
        local_size == rsearch_size
    )

    # --------------------------------------------------------
    # Line count
    # --------------------------------------------------------

    print(
        "  Counting lines..."
    )

    local_lines = count_lines(
        local_path
    )

    rsearch_lines = count_lines(
        rsearch_path
    )

    result["local_lines"] = (
        local_lines
    )

    result["rsearch_lines"] = (
        rsearch_lines
    )

    result["lines_identical"] = (
        local_lines == rsearch_lines
    )

    # --------------------------------------------------------
    # SHA-256
    # --------------------------------------------------------

    print(
        "  Computing local SHA-256..."
    )

    local_sha256 = sha256_file(
        local_path
    )

    print(
        "  Computing R-Search SHA-256..."
    )

    rsearch_sha256 = sha256_file(
        rsearch_path
    )

    result["local_sha256"] = (
        local_sha256
    )

    result["rsearch_sha256"] = (
        rsearch_sha256
    )

    result["sha256_identical"] = (
        local_sha256 == rsearch_sha256
    )

    # --------------------------------------------------------
    # Final exact comparison
    # --------------------------------------------------------
    #
    # SHA-256 is the decisive comparison.
    #
    # If SHA-256 is identical, the files are considered
    # byte-for-byte identical.
    #
    # Size and line count are additionally reported as
    # sanity checks.
    # --------------------------------------------------------

    result["byte_for_byte_identical"] = (
        result["sha256_identical"]
    )

    return result


# ============================================================
# CHECK DATASET STRUCTURE
# ============================================================

def check_dataset_structure(
    dataset_dir,
    dataset_name,
):
    """
    Verify that all expected files exist.
    """

    print()
    print("=" * 80)
    print(
        f"CHECKING {dataset_name}"
    )
    print("=" * 80)

    print(
        f"\nDirectory:\n"
        f"  {dataset_dir}"
    )

    missing = []

    for filename in REQUIRED_FILES:

        path = (
            dataset_dir
            / filename
        )

        if path.is_file():

            size = (
                path.stat().st_size
            )

            print(
                f"  OK       "
                f"{filename:<40}"
                f"{format_size(size):>12}"
            )

        else:

            print(
                f"  MISSING  "
                f"{filename}"
            )

            missing.append(
                filename
            )

    return missing


# ============================================================
# COMPARE ALL FILES
# ============================================================

def compare_all_files():

    print()
    print("=" * 80)
    print("EXACT MuSiQue DATASET COMPARISON")
    print("=" * 80)

    print(
        "\nDataset A:"
    )

    print(
        f"  {LOCAL_DATASET_DIR}"
    )

    print(
        "\nDataset B:"
    )

    print(
        f"  {RSEARCH_DATASET_DIR}"
    )

    results = []

    for index, filename in enumerate(
        REQUIRED_FILES,
        start=1,
    ):

        print()
        print("=" * 80)
        print(
            f"[{index}/{len(REQUIRED_FILES)}] "
            f"{filename}"
        )
        print("=" * 80)

        result = compare_file(
            filename
        )

        results.append(
            result
        )

        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        if not result["local_exists"]:

            print(
                "\n  Dataset A: MISSING"
            )

            continue

        if not result["rsearch_exists"]:

            print(
                "\n  Dataset B: MISSING"
            )

            continue

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        print(
            "\n  FILE SIZE"
        )

        print(
            f"    Dataset A: "
            f"{result['local_size_bytes']:,} bytes "
            f"({format_size(result['local_size_bytes'])})"
        )

        print(
            f"    Dataset B: "
            f"{result['rsearch_size_bytes']:,} bytes "
            f"({format_size(result['rsearch_size_bytes'])})"
        )

        print(
            f"    Match:     "
            f"{result['size_identical']}"
        )

        # ----------------------------------------------------
        # Lines
        # ----------------------------------------------------

        print(
            "\n  LINE COUNT"
        )

        print(
            f"    Dataset A: "
            f"{result['local_lines']:,}"
        )

        print(
            f"    Dataset B: "
            f"{result['rsearch_lines']:,}"
        )

        print(
            f"    Match:     "
            f"{result['lines_identical']}"
        )

        # ----------------------------------------------------
        # SHA-256
        # ----------------------------------------------------

        print(
            "\n  SHA-256"
        )

        print(
            f"    Dataset A:"
        )

        print(
            f"      {result['local_sha256']}"
        )

        print(
            f"    Dataset B:"
        )

        print(
            f"      {result['rsearch_sha256']}"
        )

        print(
            f"\n    SHA-256 match: "
            f"{result['sha256_identical']}"
        )

        # ----------------------------------------------------
        # Final
        # ----------------------------------------------------

        if result["byte_for_byte_identical"]:

            print(
                "\n  >>> BYTE-FOR-BYTE IDENTICAL"
            )

        else:

            print(
                "\n  >>> DIFFERENT"
            )


    return results


# ============================================================
# SAVE CSV REPORT
# ============================================================

def save_report(results):

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "filename",

        "local_exists",
        "rsearch_exists",

        "local_size_bytes",
        "rsearch_size_bytes",

        "local_lines",
        "rsearch_lines",

        "local_sha256",
        "rsearch_sha256",

        "size_identical",
        "lines_identical",
        "sha256_identical",
        "byte_for_byte_identical",
    ]

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print()
    print("=" * 80)
    print("CSV REPORT")
    print("=" * 80)

    print(
        f"\nSaved to:\n"
        f"  {REPORT_PATH}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(results):

    total = len(results)

    identical = sum(
        result["byte_for_byte_identical"]
        for result in results
    )

    different = sum(
        (
            result["local_exists"]
            and result["rsearch_exists"]
            and not result["byte_for_byte_identical"]
        )
        for result in results
    )

    missing = sum(
        not (
            result["local_exists"]
            and result["rsearch_exists"]
        )
        for result in results
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"\n  Total files:             {total}"
    )

    print(
        f"  Byte-for-byte identical: {identical}"
    )

    print(
        f"  Different:               {different}"
    )

    print(
        f"  Missing:                 {missing}"
    )

    # --------------------------------------------------------
    # All identical
    # --------------------------------------------------------

    if identical == total:

        print()
        print(
            "  =================================================="
        )

        print(
            "  ALL SIX FILES ARE BYTE-FOR-BYTE IDENTICAL."
        )

        print(
            "  =================================================="
        )

        print()
        print(
            "  The MuSiQue files in:"
        )

        print(
            f"    {LOCAL_DATASET_DIR}"
        )

        print(
            "  are exactly identical to the files in:"
        )

        print(
            f"    {RSEARCH_DATASET_DIR}"
        )

        print()
        print(
            "  SHA-256 confirms the files contain exactly"
        )

        print(
            "  the same bytes."
        )

    # --------------------------------------------------------
    # Some different
    # --------------------------------------------------------

    elif different > 0:

        print()
        print(
            "  =================================================="
        )

        print(
            "  AT LEAST ONE FILE IS DIFFERENT."
        )

        print(
            "  =================================================="
        )

        print()
        print(
            "  Check the SHA-256 values above and the CSV report:"
        )

        print(
            f"    {REPORT_PATH}"
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    else:

        print()
        print(
            "  Comparison incomplete because some files are missing."
        )

    print()
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MuSiQue DATASET EXACT COMPARISON")
    print("=" * 80)

    print(
        "\nProject:"
    )

    print(
        f"  {THIS_DIR}"
    )

    # --------------------------------------------------------
    # Check Dataset A
    # --------------------------------------------------------

    missing_local = check_dataset_structure(
        LOCAL_DATASET_DIR,
        "DATASET A — NLP BOOK",
    )

    # --------------------------------------------------------
    # Check Dataset B
    # --------------------------------------------------------

    missing_rsearch = check_dataset_structure(
        RSEARCH_DATASET_DIR,
        "DATASET B — R-SEARCH",
    )

    # --------------------------------------------------------
    # Stop if incomplete
    # --------------------------------------------------------

    if missing_local or missing_rsearch:

        print()
        print("=" * 80)
        print("ERROR — DATASET INCOMPLETE")
        print("=" * 80)

        if missing_local:

            print(
                "\nMissing from Dataset A:"
            )

            for filename in missing_local:

                print(
                    f"  - {filename}"
                )

        if missing_rsearch:

            print(
                "\nMissing from Dataset B:"
            )

            for filename in missing_rsearch:

                print(
                    f"  - {filename}"
                )

        sys.exit(1)

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    results = compare_all_files()

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    main()