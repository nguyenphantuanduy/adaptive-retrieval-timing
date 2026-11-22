"""Generate report assets from saved experiment artifacts. No model execution."""
from contextlib import chdir
import csv
import json
import re
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1]
OUT = REPORT / "graphics"
PROBES = ROOT / "phase_1/stage_2_hypothesis_verification"
NAMES = {"musique": "MuSiQue", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2WikiMultiHopQA"}


def table(filename, caption, label, columns, header, rows):
    lines = [r"\begin{table}[htbp]\centering\small", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
             r"\toprule", header + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def probes():
    h1 = PROBES / "H1_attention_modulation/lm_loss_results/plots/lm_loss_summary.csv"
    with h1.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
    for ratio in ("0:3", "1:2", "2:1", "3:0"):
        series = sorted((float(x["decision_score"]), float(x["mean_lm_loss"]))
                        for x in rows if x["hop"] == "Total" and x["retrieval_ratio"] == ratio)
        assert len(series) == 33
        ax.plot(*zip(*series), ".-", label=ratio)
    ax.set(xlabel="Hệ số trộn ép d", ylabel="Mất mát LM trung bình", xlim=(0, 1))
    ax.legend(title="Hỗ trợ:gây nhiễu")
    ax.grid(alpha=.2)
    fig.savefig(OUT / "h1.pdf")
    plt.close(fig)

    path = PROBES / "H2_gradient_direction/analyze_threshold_and_cost_simplified.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    assert "tau = 0.4297" in "".join(cells[8]["source"])
    namespace = {}
    with chdir(path.parent), patch.object(plt, "show"):
        for index in (2, 4, 8, 10):
            exec("".join(cells[index]["source"]), namespace)
        for index, name in [(11, "h2-gradient.pdf"), (13, "h2-cost.pdf")]:
            source = "".join(cells[index]["source"])
            # The notebook stores a literal tab before "au" in its legend.
            exec(source.replace("$\tau=", r"$\tau="), namespace)
            if index == 13:
                # Separate adjacent annotation boxes while keeping their data anchors.
                annotations = sorted(plt.gca().texts, key=lambda label: label.xy[0])
                for left, right in zip(annotations, annotations[1:]):
                    if right.xy[0] - left.xy[0] < 0.007:
                        right.set_position((0, 43))
            plt.gcf().savefig(OUT / name)
            plt.close()


def stats(metrics):
    return tuple(f"{metrics[key]:.2f}" for key in ("Acc", "F1", "EM"))


def benchmarks():
    archive = json.loads((ROOT / "REVIVE_training/metrics/retrieval_cost/retrieval_cost_summary.json").read_text())
    rows = [x for x in archive["configurations"] if x["threshold"] is not None]
    with (ROOT / "REVIVE_training/metrics/retrieval_cost/retrieval_cost_per_sample.csv").open(newline="", encoding="utf-8") as handle:
        sample_rows = list(csv.DictReader(handle))
    for row in rows:
        counts = [int(x["retrieval_count"]) for x in sample_rows if x["source_filename"] == row["source_filename"]]
        verdict = json.loads((ROOT / row["metric_filename"]).read_text())
        assert row["metrics"] == verdict["metrics"]
        assert len(counts) == row["sample_count"] == 500
        assert sum(counts) == row["total_queries"]
        assert sum(counts) / len(counts) == row["average_queries"]
        assert sum(x > 0 for x in counts) * 100 / len(counts) == row["retrieval_rate_percent"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.25), layout="constrained")
    local, sweep = [], []
    ids = {"musique": ("M05", "M06", "M12"), "hotpotqa": ("M03", "M04", "M08"),
           "2wikimultihopqa": ("M01", "M02", "M07")}
    for ax, (key, title) in zip(axes, NAMES.items()):
        group = sorted((r for r in rows if r["dataset"] == key), key=lambda r: r["threshold"])
        folder = ROOT / "REVIVE_training/metrics/baselines" / key
        for i, (method, basename) in enumerate([("No-RAG", "no_rag"),
                                                ("Vanilla RAG", "vanilla_rag_top5")]):
            path, = folder.glob(basename + "*metrics.json")
            receipt = json.loads(path.read_text())
            local.append([title if i == 0 else "", method, *stats(receipt["metrics"]), ids[key][i]])
        local.append(["", "ART", *stats(group[0]["metrics"]), ids[key][2]])
        ax.margins(x=.24, y=.22)
        for i, row in enumerate(group):
            tau = row["threshold"]
            sweep.append([title if i == 0 else "", f"{tau:.4f}" if i == 0 else f"{tau:.2f}",
                          *stats(row["metrics"]), f"{row['average_queries']:.3f}",
                          f"{row['retrieval_rate_percent']:.1f}", row["total_queries"]])
            x, y = row["average_queries"], row["metrics"]["F1"]
            ax.scatter(x, y, color="#206897", zorder=2)
            if row["pareto_efficient_within_threshold_sweep"]:
                ax.scatter(x, y, facecolors="none", edgecolors="#206897", s=110, zorder=2)
            ax.annotate(f"{tau:g}", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Truy vấn/mẫu")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("F1 (%)")
    fig.savefig(OUT / "tradeoff.pdf")
    plt.close(fig)
    table("local-table.tex", "Đối chứng địa phương, 500 mẫu mỗi bộ. ART dùng ngưỡng 0.4297. Chỉ số theo phần trăm.", "tab:local",
          "llrrrl", "Bộ dữ liệu & Hệ thống & Acc & F1 & EM & ID", local)
    table("sweep-table.tex", r"Quét ngưỡng ART. TB là truy vấn/mẫu, Có (\%) là tỷ lệ mẫu có truy vấn và Tổng là số truy vấn ghi nhận.", "tab:sweep",
          "lrrrrrrr", r"Bộ dữ liệu & $\tau$ & Acc & F1 & EM & TB & Có (\%) & Tổng", sweep)


if __name__ == "__main__":
    probes()
    benchmarks()
