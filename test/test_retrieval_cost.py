import json
import unittest
from pathlib import Path
from uuid import uuid4

from REVIVE_training.analyze_retrieval_cost import (
    analyze_archive,
    benchmark_scores,
    pareto_efficient,
    read_queries,
    summarize,
    threshold_for,
)


class RetrievalCostTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".retrieval-cost-test-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.root.rmdir)

    def write(self, filename, content):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.addCleanup(path.unlink)
        return path

    def write_predictions(self, rows):
        return self.write("predictions.jsonl", "".join(json.dumps(row) + "\n" for row in rows))

    def test_counts_queries_in_input_order_and_keeps_duplicate_events(self):
        path = self.write_predictions([
            {"pure_art_queries": [], "pure_art_answer": "<search>marker</search>"},
            {"id": 9, "pure_art_queries": ["query"]},
            {"id": 3, "pure_art_queries": ["same", "same", "other"]},
        ])
        samples, diagnostics = read_queries(path)
        self.assertEqual(samples, [(None, 0), (9, 1), (3, 3)])
        self.assertEqual(diagnostics["samples_with_repeated_queries"], 1)
        stats = summarize(count for _, count in samples)
        self.assertEqual(stats["total_queries"], 4)
        self.assertEqual(stats["average_queries"], 4 / 3)
        self.assertEqual(stats["median_queries"], 1)
        self.assertAlmostEqual(stats["population_stddev"], (14 / 9) ** 0.5)
        self.assertEqual(stats["distribution"], {0: 1, 1: 1, 2: 0, 3: 1})
        self.assertAlmostEqual(stats["retrieval_rate_percent"], 200 / 3)
        self.assertAlmostEqual(stats["p90_queries"], 2.6)
        self.assertAlmostEqual(stats["p95_queries"], 2.8)

    def test_missing_and_malformed_fields_are_not_counted_as_zero(self):
        for row in ({}, {"pure_art_queries": "query"}, {"pure_art_queries": [1]}, {"pure_art_queries": [" "]}):
            with self.subTest(row=row):
                path = self.write(f"invalid-{uuid4().hex}.jsonl", json.dumps(row) + "\n")
                with self.assertRaisesRegex(ValueError, "line 1: pure_art_queries"):
                    read_queries(path)

    def test_empty_and_malformed_jsonl(self):
        with self.assertRaisesRegex(ValueError, "empty input"):
            read_queries(self.write("empty.jsonl", "\n"))
        with self.assertRaisesRegex(ValueError, "line 1: malformed JSON"):
            read_queries(self.write("malformed.jsonl", "{\n"))

    def test_sample_count_mismatch(self):
        path = self.write("metrics.json", json.dumps({"num_samples": 2, "metrics": {"Acc": 1, "F1": 2, "EM": 3}}))
        self.assertEqual(benchmark_scores(path, 2)["F1"], 2)
        with self.assertRaisesRegex(ValueError, "sample-count mismatch"):
            benchmark_scores(path, 1)

    def test_pairing_and_reporting_bad_runs(self):
        for name in ("result", "metrics"):
            folder = self.root / name / "musique"
            folder.mkdir(parents=True)
            self.addCleanup(folder.parent.rmdir)
            self.addCleanup(folder.rmdir)
        self.write("result/musique/threshold50.jsonl", '{"pure_art_queries":[]}\n')
        self.write("result/musique/threshold60.jsonl", '{"pure_art_queries":["q"]}\n')
        for threshold, count in ((50, 1), (60, 2)):
            self.write(f"metrics/musique/threshold{threshold}_metrics.json", json.dumps({
                "num_samples": count, "metrics": {"Acc": 1, "F1": 2, "EM": 3}
            }))
        summary, per_sample = analyze_archive(self.root)
        self.assertEqual([row["threshold"] for row in summary["configurations"]], [0.50])
        self.assertEqual([row["retrieval_count"] for row in per_sample], [0])
        self.assertIn("sample-count mismatch", summary["issues"][0]["reason"])

    def test_threshold_and_pareto_conventions(self):
        self.assertEqual(threshold_for("rsearch_musique_art_top5_results.jsonl"), 0.4297)
        self.assertEqual(threshold_for("rsearch_musique_art_top5_threshold70_results.jsonl"), 0.70)
        self.assertIsNone(threshold_for("rsearch_musique_art_results.jsonl"))
        fast, accurate, dominated = [
            {"average_queries": cost, "metrics": {"F1": f1}}
            for cost, f1 in ((1, 10), (2, 11), (3, 10))
        ]
        self.assertEqual(pareto_efficient([fast, accurate, dominated]), [fast, accurate])


if __name__ == "__main__":
    unittest.main()
