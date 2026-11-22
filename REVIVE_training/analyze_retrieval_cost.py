"""Analyze ART query logs and join them to archived benchmark summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE = Path("art-backup/demo-3")
OUTPUT = Path("REVIVE_training/metrics/retrieval_cost")


def threshold_for(filename):
    explicit = re.search(r"(?:^|_)threshold(50|60|70)(?:_|\.|$)", filename)
    if explicit:
        return int(explicit[1]) / 100
    if "_art_top5" in filename and "threshold" not in filename:
        return 0.4297
    return None


def read_queries(path):
    samples = []
    identity = hashlib.sha256()
    repeat_rows = 0
    score_conflicts = 0
    passage_ratios = Counter()
    zero_with_passages = 0

    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: malformed JSON ({error.msg})") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            queries = row.get("pure_art_queries")
            if not isinstance(queries, list) or any(
                not isinstance(query, str) or not query.strip() for query in queries
            ):
                raise ValueError(f"line {line_number}: pure_art_queries must be a list of nonblank strings")

            count = len(queries)
            samples.append((row.get("id"), count))
            identity.update(
                (json.dumps([row.get(k) for k in ("id", "question", "golden_answers")], sort_keys=True) + "\n")
                .encode("utf-8")
            )
            repeat_rows += len(set(queries)) < count
            scores = row.get("pure_art_retrieval_scores")
            if isinstance(scores, list):
                score_conflicts += count > len(scores)
            passages = row.get("pure_art_passages")
            if isinstance(passages, list):
                if count:
                    passage_ratios[len(passages) / count] += 1
                else:
                    zero_with_passages += bool(passages)

    if not samples:
        raise ValueError("empty input: no prediction rows")
    return samples, {
        "ordered_identity_sha256": identity.hexdigest(),
        "samples_with_repeated_queries": repeat_rows,
        "samples_with_more_queries_than_scores": score_conflicts,
        "passages_per_query_ratio_distribution": dict(sorted(passage_ratios.items())),
        "zero_query_samples_with_passages": zero_with_passages,
    }


def summarize(counts):
    ordered = sorted(counts)
    sample_count = len(ordered)
    if not sample_count:
        raise ValueError("cannot summarize zero samples")

    def percentile(fraction):
        position = (sample_count - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, sample_count - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

    distribution = Counter(ordered)
    zero = distribution[0]
    return {
        "sample_count": sample_count,
        "total_queries": sum(ordered),
        "average_queries": statistics.mean(ordered),
        "median_queries": statistics.median(ordered),
        "population_stddev": statistics.pstdev(ordered),
        "min_queries": ordered[0],
        "max_queries": ordered[-1],
        "p90_queries": percentile(0.9),
        "p95_queries": percentile(0.95),
        "zero_retrieval_samples": zero,
        "zero_retrieval_percent": 100 * zero / sample_count,
        "retrieval_samples": sample_count - zero,
        "retrieval_rate_percent": 100 * (sample_count - zero) / sample_count,
        "distribution": {count: distribution[count] for count in range(ordered[-1] + 1)},
    }


def benchmark_scores(path, sample_count):
    metrics = json.loads(path.read_text(encoding="utf-8-sig"))
    if metrics["num_samples"] != sample_count:
        raise ValueError(f"sample-count mismatch: predictions={sample_count}, metrics={metrics['num_samples']}")
    scores = {name: metrics["metrics"][name] for name in ("Acc", "F1", "EM")}
    if any(type(value) not in (int, float) or not 0 <= value <= 100 for value in scores.values()):
        raise ValueError("benchmark scores must be numeric percentages")
    return scores


def pareto_efficient(configurations):
    """Return configurations undominated on higher F1 and lower query cost."""
    return [
        run for run in configurations
        if not any(
            other["metrics"]["F1"] >= run["metrics"]["F1"]
            and other["average_queries"] <= run["average_queries"]
            and (
                other["metrics"]["F1"] > run["metrics"]["F1"]
                or other["average_queries"] < run["average_queries"]
            )
            for other in configurations
        )
    ]


def analyze_archive(archive):
    results_root = archive / "result"
    metrics_root = archive / "metrics"
    by_metric = defaultdict(list)
    for prediction in sorted(results_root.rglob("*.jsonl")):
        relative = prediction.relative_to(results_root)
        name = prediction.stem.removesuffix("_results") + "_metrics.json"
        by_metric[metrics_root / relative.with_name(name)].append(prediction)

    configurations = []
    per_sample = []
    issues = []
    metric_files = set(metrics_root.rglob("*_metrics.json"))
    for missing in sorted(by_metric.keys() - metric_files):
        for prediction in by_metric[missing]:
            issues.append({"source_filename": prediction.as_posix(), "reason": f"no matching metric: {missing.as_posix()}"})

    for metric in sorted(metric_files):
        candidates = by_metric.get(metric, [])
        if len(candidates) != 1 or len(metric.relative_to(metrics_root).parts) != 2:
            issues.append({
                "metric_filename": metric.as_posix(),
                "reason": "expected one dataset directory and one corresponding prediction file",
                "candidates": [path.as_posix() for path in candidates],
            })
            continue

        prediction = candidates[0]
        dataset = metric.parent.name
        threshold = threshold_for(prediction.name)
        try:
            samples, diagnostics = read_queries(prediction)
            scores = benchmark_scores(metric, len(samples))
        except (ValueError, OSError, KeyError, TypeError) as error:
            issues.append({
                "source_filename": prediction.as_posix(),
                "metric_filename": metric.as_posix(),
                "reason": str(error),
            })
            continue

        if threshold is None:
            issues.append({"source_filename": prediction.as_posix(), "reason": "threshold unverified; excluded from threshold sweep"})
        configurations.append({
            "dataset": dataset,
            "threshold": threshold,
            "source_filename": prediction.as_posix(),
            "metric_filename": metric.as_posix(),
            "metrics": scores,
            **summarize(count for _, count in samples),
            "diagnostics": diagnostics,
        })
        per_sample.extend({
            "dataset": dataset,
            "threshold": threshold,
            "sample_id": sample_id,
            "retrieval_count": count,
            "source_filename": prediction.as_posix(),
        } for sample_id, count in samples)

    configurations.sort(key=lambda run: (run["dataset"], run["threshold"] is None, run["threshold"] or 0, run["source_filename"]))
    grouped = defaultdict(list)
    for run in configurations:
        grouped[run["dataset"]].append(run)
    for runs in grouped.values():
        sweep = [run for run in runs if run["threshold"] is not None]
        efficient_all = pareto_efficient(runs)
        efficient_sweep = pareto_efficient(sweep)
        for run in runs:
            run["pareto_efficient_within_dataset"] = run in efficient_all
            run["pareto_efficient_within_threshold_sweep"] = run in efficient_sweep if run in sweep else None

    return {
        "count_method": "Length of pure_art_queries in completed prediction rows, including repeats.",
        "count_provenance": "Logged search-call proxy. Local generation appends immediately before search; no per-call success or failure status, and historical remote code revisions are unpinned.",
        "percentile_method": "Linear interpolation at (n - 1) * percentile_fraction.",
        "configurations": configurations,
        "issues": issues,
    }, per_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE, help="Archive directory containing result and metrics.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not all((args.archive / name).is_dir() for name in ("result", "metrics")):
        parser.error("--archive must contain result and metrics directories")

    summary, per_sample = analyze_archive(args.archive)
    if not summary["configurations"]:
        parser.error(f"no analyzable configuration: {summary['issues']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "retrieval_cost_summary.json"
    csv_path = args.output_dir / "retrieval_cost_per_sample.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "threshold", "sample_id", "retrieval_count", "source_filename"])
        writer.writeheader()
        writer.writerows(per_sample)
    print(f"Wrote {len(summary['configurations'])} configurations and {len(per_sample)} samples.")
    for issue in summary["issues"]:
        print(f"NOTE: {issue}")


if __name__ == "__main__":
    main()
