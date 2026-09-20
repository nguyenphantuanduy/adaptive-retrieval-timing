from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("lm_loss_results")

FILES = {
    "2-hop": DATA_DIR / "lm_loss_2_hop.csv",
    "3-hop": DATA_DIR / "lm_loss_3_hop.csv",
    "4-hop": DATA_DIR / "lm_loss_4_hop.csv",
}

OUTPUT_DIR = DATA_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET LOSS COLUMNS
# ============================================================

def get_loss_columns(df):
    """
    Expected LM-loss column format:

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

        result.setdefault(ratio, {})[score] = column

    return result


# ============================================================
# COMPUTE MEAN CURVES
# ============================================================

def compute_curves(df):
    """
    Compute mean LM loss for every:

        retrieval ratio × decision score

    Blank/non-applicable cells are ignored.
    """

    columns = get_loss_columns(df)

    curves = {}

    for ratio, score_columns in columns.items():

        scores = sorted(score_columns.keys())

        means = []
        counts = []

        for score in scores:

            column = score_columns[score]

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                mean_loss = np.nan
            else:
                mean_loss = values.mean()

            means.append(mean_loss)
            counts.append(len(values))

        curves[ratio] = {
            "scores": scores,
            "means": means,
            "counts": counts,
        }

    return curves


# ============================================================
# PLOT CURVES
# ============================================================

def plot_curves(curves, title, output_path):
    """
    Plot mean LM loss vs decision score.

    One line per retrieval ratio.
    """

    plt.figure(figsize=(10, 6))

    for ratio in sorted(curves.keys()):

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

    plt.xlabel(
        "Decision score",
        fontsize=12,
    )

    plt.ylabel(
        "Mean LM loss",
        fontsize=12,
    )

    plt.title(
        f"{title}: Mean LM Loss vs Decision Score",
        fontsize=14,
    )

    # Decision score ∈ [0, 1]
    plt.xlim(0, 1)

    plt.xticks(
        np.linspace(0, 1, 11)
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

    print(f"Saved: {output_path}")


# ============================================================
# COMPUTE TOTAL
# ============================================================

def compute_total_curves(dataframes):
    """
    Pool ALL valid sample-level losses across:

        2-hop
        3-hop
        4-hop

    for every:

        retrieval ratio × decision score

    IMPORTANT:

        This computes:

            mean(all applicable samples)

        NOT:

            mean(
                mean_2hop,
                mean_3hop,
                mean_4hop
            )

    Therefore, each sample contributes equally.
    """

    combined = {}

    for df in dataframes.values():

        columns = get_loss_columns(df)

        for ratio, score_columns in columns.items():

            combined.setdefault(
                ratio,
                {}
            )

            for score, column in score_columns.items():

                values = pd.to_numeric(
                    df[column],
                    errors="coerce",
                ).dropna()

                combined[ratio].setdefault(
                    score,
                    []
                )

                combined[ratio][score].extend(
                    values.tolist()
                )

    # --------------------------------------------------------
    # Compute pooled means
    # --------------------------------------------------------

    curves = {}

    for ratio, score_values in combined.items():

        scores = sorted(
            score_values.keys()
        )

        means = []
        counts = []

        for score in scores:

            values = score_values[score]

            if len(values) == 0:
                mean_loss = np.nan
            else:
                mean_loss = np.mean(values)

            means.append(mean_loss)
            counts.append(len(values))

        curves[ratio] = {
            "scores": scores,
            "means": means,
            "counts": counts,
        }

    return curves


# ============================================================
# PRINT INFO
# ============================================================

def print_info(hop_name, df, curves):
    """
    Print basic statistics for each hop.
    """

    print()
    print("=" * 60)
    print(hop_name)
    print("=" * 60)

    print(
        f"Samples: {len(df):,}"
    )

    print(
        f"Ratios: {sorted(curves.keys())}"
    )

    for ratio in sorted(curves.keys()):

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

def save_summary_csv(all_curves, output_path):
    """
    Save all statistics into one long-format CSV.

    Columns:

        hop
        retrieval_ratio
        decision_score
        mean_lm_loss
        n_samples
    """

    rows = []

    for hop_name, curves in all_curves.items():

        for ratio in sorted(curves.keys()):

            scores = curves[ratio]["scores"]
            means = curves[ratio]["means"]
            counts = curves[ratio]["counts"]

            for score, mean_loss, count in zip(
                scores,
                means,
                counts,
            ):

                rows.append({
                    "hop": hop_name,
                    "retrieval_ratio": ratio,
                    "decision_score": score,
                    "mean_lm_loss": mean_loss,
                    "n_samples": count,
                })

    summary = pd.DataFrame(rows)

    # Consistent ordering
    hop_order = [
        "2-hop",
        "3-hop",
        "4-hop",
        "Total",
    ]

    ratio_order = [
        "3:0",
        "2:1",
        "1:2",
        "0:3",
    ]

    summary["hop"] = pd.Categorical(
        summary["hop"],
        categories=hop_order,
        ordered=True,
    )

    summary["retrieval_ratio"] = pd.Categorical(
        summary["retrieval_ratio"],
        categories=ratio_order,
        ordered=True,
    )

    summary = summary.sort_values(
        [
            "hop",
            "retrieval_ratio",
            "decision_score",
        ]
    )

    # Convert categorical columns back to strings
    # before saving.
    summary["hop"] = summary["hop"].astype(str)
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
    print("H1 LM-Loss Plotting")
    print("=" * 60)

    # --------------------------------------------------------
    # Load CSVs
    # --------------------------------------------------------

    dataframes = {}

    for hop_name, path in FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"CSV not found:\n{path}"
            )

        print(
            f"\nLoading {hop_name}: {path}"
        )

        df = pd.read_csv(
            path,
            keep_default_na=False,
        )

        dataframes[hop_name] = df

        print(
            f"  Samples: {len(df):,}"
        )

    # --------------------------------------------------------
    # Store all curves for summary CSV
    # --------------------------------------------------------

    all_curves = {}

    # --------------------------------------------------------
    # Individual hop plots
    # --------------------------------------------------------

    for hop_name, df in dataframes.items():

        curves = compute_curves(df)

        all_curves[hop_name] = curves

        print_info(
            hop_name,
            df,
            curves,
        )

        filename = (
            "lm_loss_vs_decision_score_"
            f"{hop_name.replace('-', '')}.png"
        )

        output_path = (
            OUTPUT_DIR / filename
        )

        plot_curves(
            curves,
            hop_name,
            output_path,
        )

    # --------------------------------------------------------
    # Compute TOTAL
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Computing TOTAL")
    print("=" * 60)

    total_curves = compute_total_curves(
        dataframes
    )

    all_curves["Total"] = total_curves

    for ratio in sorted(total_curves.keys()):

        counts = total_curves[ratio]["counts"]

        print(
            f"  {ratio}: "
            f"{min(counts):,} - "
            f"{max(counts):,} samples"
        )

    # --------------------------------------------------------
    # Total plot
    # --------------------------------------------------------

    total_path = (
        OUTPUT_DIR
        / "lm_loss_vs_decision_score_total.png"
    )

    plot_curves(
        total_curves,
        "Total",
        total_path,
    )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "lm_loss_summary.csv"
    )

    save_summary_csv(
        all_curves,
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
        OUTPUT_DIR
        / "lm_loss_vs_decision_score_2hop.png"
    )

    print(
        OUTPUT_DIR
        / "lm_loss_vs_decision_score_3hop.png"
    )

    print(
        OUTPUT_DIR
        / "lm_loss_vs_decision_score_4hop.png"
    )

    print(
        OUTPUT_DIR
        / "lm_loss_vs_decision_score_total.png"
    )

    print(
        OUTPUT_DIR
        / "lm_loss_summary.csv"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()