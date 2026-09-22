from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(".")

INPUT_FILE = DATA_DIR / "gradient_2_hop.csv"

OUTPUT_DIR = DATA_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET GRADIENT COLUMNS
# ============================================================

def get_gradient_columns(df):
    """
    Expected gradient column format:

        3:0/0.0000
        3:0/0.0312
        3:0/0.0625
        ...
        2:1/0.0000
        ...

    Returns:

        {
            "3:0": {
                0.0000: "3:0/0.0000",
                0.0312: "3:0/0.0312",
                ...
            },
            "2:1": {
                ...
            }
        }
    """

    result = {}

    for column in df.columns:

        # Metadata columns do not contain "/"
        if "/" not in column:
            continue

        ratio, score = column.split("/")

        score = float(score)

        result.setdefault(
            ratio,
            {}
        )[score] = column

    return result


# ============================================================
# COMPUTE MEAN GRADIENT CURVES
# ============================================================

def compute_curves(df):
    """
    Compute mean gradient for every:

        retrieval ratio × decision score

    Blank / invalid values are ignored.

    The gradient represents:

        ∂L / ∂d
    """

    columns = get_gradient_columns(df)

    curves = {}

    for ratio, score_columns in columns.items():

        scores = sorted(
            score_columns.keys()
        )

        means = []
        counts = []

        for score in scores:

            column = score_columns[score]

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            if len(values) == 0:

                mean_gradient = np.nan

            else:

                mean_gradient = values.mean()

            means.append(
                mean_gradient
            )

            counts.append(
                len(values)
            )

        curves[ratio] = {
            "scores": scores,
            "means": means,
            "counts": counts,
        }

    return curves


# ============================================================
# PLOT GRADIENT CURVES
# ============================================================

def plot_curves(
    curves,
    title,
    output_path,
):
    """
    Plot mean ∂L/∂d vs decision score.

    One line per retrieval ratio.

    The horizontal y=0 line is used as a reference
    for gradient direction.
    """

    plt.figure(
        figsize=(10, 6)
    )

    for ratio in sorted(
        curves.keys()
    ):

        scores = np.array(
            curves[ratio]["scores"],
            dtype=float,
        )

        means = np.array(
            curves[ratio]["means"],
            dtype=float,
        )

        # Keep only valid points
        valid = ~np.isnan(means)

        scores = scores[valid]
        means = means[valid]

        # Skip ratio if no valid data
        if len(scores) == 0:
            continue

        plt.plot(
            scores,
            means,
            marker="o",
            markersize=3,
            linewidth=2,
            label=ratio,
        )

    # --------------------------------------------------------
    # Zero-gradient reference
    # --------------------------------------------------------

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xlabel(
        "Decision score",
        fontsize=12,
    )

    plt.ylabel(
        r"Mean gradient $\partial L / \partial d$",
        fontsize=12,
    )

    plt.title(
        f"{title}: Mean Gradient vs Decision Score",
        fontsize=14,
    )

    # Decision score ∈ [0, 1]
    plt.xlim(
        0,
        1,
    )

    plt.xticks(
        np.linspace(
            0,
            1,
            11,
        )
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend(
        title="Retrieval ratio"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PRINT INFO
# ============================================================

def print_info(
    df,
    curves,
):
    """
    Print basic statistics.
    """

    print()
    print("=" * 60)
    print("2-hop")
    print("=" * 60)

    print(
        f"Samples: {len(df):,}"
    )

    print(
        f"Ratios: {sorted(curves.keys())}"
    )

    for ratio in sorted(
        curves.keys()
    ):

        counts = curves[ratio]["counts"]

        valid_counts = [
            count
            for count in counts
            if count > 0
        ]

        if valid_counts:

            print(
                f"  {ratio}: "
                f"{min(valid_counts):,} - "
                f"{max(valid_counts):,} samples"
            )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_summary_csv(
    curves,
    output_path,
):
    """
    Save gradient statistics into one
    long-format CSV.

    Columns:

        retrieval_ratio
        decision_score
        mean_gradient
        n_samples
    """

    rows = []

    for ratio in sorted(
        curves.keys()
    ):

        scores = curves[ratio]["scores"]
        means = curves[ratio]["means"]
        counts = curves[ratio]["counts"]

        for (
            score,
            mean_gradient,
            count,
        ) in zip(
            scores,
            means,
            counts,
        ):

            rows.append({
                "retrieval_ratio": ratio,
                "decision_score": score,
                "mean_gradient": mean_gradient,
                "n_samples": count,
            })

    summary = pd.DataFrame(
        rows
    )

    # Consistent ratio ordering
    ratio_order = [
        "3:0",
        "2:1",
        "1:2",
        "0:3",
    ]

    summary["retrieval_ratio"] = pd.Categorical(
        summary["retrieval_ratio"],
        categories=ratio_order,
        ordered=True,
    )

    summary = summary.sort_values(
        [
            "retrieval_ratio",
            "decision_score",
        ]
    )

    summary["retrieval_ratio"] = (
        summary["retrieval_ratio"].astype(str)
    )

    summary.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )

    print(
        f"Saved summary CSV: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("H2 Gradient Plotting")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{INPUT_FILE.resolve()}"
        )

    print(
        f"\nLoading: {INPUT_FILE.resolve()}"
    )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        keep_default_na=False,
    )

    print(
        f"Samples: {len(df):,}"
    )

    # --------------------------------------------------------
    # Compute curves
    # --------------------------------------------------------

    curves = compute_curves(
        df
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print_info(
        df,
        curves,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_path = (
        OUTPUT_DIR
        / "gradient_vs_decision_score_2hop.png"
    )

    plot_curves(
        curves,
        "2-hop",
        plot_path,
    )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "gradient_summary_2hop.csv"
    )

    save_summary_csv(
        curves,
        summary_path,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print("\nGenerated files:")

    print(
        plot_path.resolve()
    )

    print(
        summary_path.resolve()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()