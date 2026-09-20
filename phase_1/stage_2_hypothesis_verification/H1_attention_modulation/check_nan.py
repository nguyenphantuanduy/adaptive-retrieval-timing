from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("lm_loss_results")

FILES = [
    "lm_loss_2_hop.csv",
    "lm_loss_3_hop.csv",
    "lm_loss_4_hop.csv",
]


def check_file(filepath):
    print("=" * 100)
    print(f"FILE: {filepath}")
    print("=" * 100)

    # Keep empty strings as empty strings.
    # This is important because "" means "not applicable",
    # while "NaN" means the evaluation explicitly failed.
    df = pd.read_csv(
        filepath,
        keep_default_na=False,
    )

    print(f"Total samples: {len(df):,}")
    print(f"Total columns: {len(df.columns):,}")

    # --------------------------------------------------------
    # Only check actual NaN values
    # --------------------------------------------------------

    # Since keep_default_na=False, empty CSV fields remain "".
    # Therefore isna() now detects actual NaN values only.
    nan_mask = df.isna().any(axis=1)

    nan_samples = df.loc[nan_mask]

    num_nan = len(nan_samples)
    num_valid = len(df) - num_nan

    print()
    print(
        f"Samples with actual NaN    : {num_nan:,} "
        f"({num_nan / len(df) * 100:.2f}%)"
    )

    print(
        f"Samples without actual NaN : {num_valid:,} "
        f"({num_valid / len(df) * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Detailed information
    # --------------------------------------------------------

    if num_nan > 0:

        print()
        print("Samples containing actual NaN:")
        print("-" * 100)

        for _, row in nan_samples.iterrows():

            nan_columns = [
                column
                for column in df.columns
                if pd.isna(row[column])
            ]

            print(
                f"sample_id={row['sample_id']}, "
                f"sample_index={row['sample_index']}"
            )

            print(
                f"  NaN columns ({len(nan_columns)}):"
            )

            for column in nan_columns:
                print(f"    - {column}")

    else:

        print()
        print("No actual NaN values found.")

    # --------------------------------------------------------
    # Count NaN by column
    # --------------------------------------------------------

    nan_by_column = df.isna().sum()

    nan_by_column = (
        nan_by_column[
            nan_by_column > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if len(nan_by_column) > 0:

        print()
        print("Actual NaN count by column:")
        print("-" * 100)

        for column, count in nan_by_column.items():

            print(
                f"{column}: {count:,} "
                f"({count / len(df) * 100:.2f}%)"
            )

    return {
        "file": filepath.name,
        "total_samples": len(df),
        "samples_with_nan": num_nan,
        "samples_without_nan": num_valid,
        "nan_percentage": (
            num_nan / len(df) * 100
        ),
    }


def main():

    print()
    print("=" * 100)
    print("ART LM-LOSS RESULTS - ACTUAL NaN CHECK")
    print("=" * 100)

    summaries = []

    for filename in FILES:

        filepath = (
            RESULTS_DIR / filename
        )

        if not filepath.exists():

            print()
            print(
                f"WARNING: File not found: "
                f"{filepath}"
            )

            continue

        summary = check_file(
            filepath
        )

        summaries.append(
            summary
        )

    # ========================================================
    # Overall summary
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    if not summaries:

        print("No files were found.")
        return

    summary_df = pd.DataFrame(
        summaries
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    total_samples = (
        summary_df[
            "total_samples"
        ].sum()
    )

    total_nan_samples = (
        summary_df[
            "samples_with_nan"
        ].sum()
    )

    print()
    print("-" * 100)

    print(
        f"Total samples across files : "
        f"{total_samples:,}"
    )

    print(
        f"Samples containing NaN      : "
        f"{total_nan_samples:,}"
    )

    print(
        f"Samples without NaN         : "
        f"{total_samples - total_nan_samples:,}"
    )


if __name__ == "__main__":
    main()