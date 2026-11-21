import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from REVIVE_training.baseline_generation import (
    fresh_output,
    preprocess_question,
    render_prompt,
    validate_output,
)
from REVIVE_training.evaluate_baseline import score_baseline


class BaselineProtocolTests(unittest.TestCase):
    def test_question_preprocessing(self):
        self.assertEqual(preprocess_question(" \tWho wrote it? \n"), "Who wrote it?")
        self.assertEqual(preprocess_question(" \tWho wrote it \n"), "Who wrote it?")
        self.assertEqual(preprocess_question("Who wrote it."), "Who wrote it.?")

    def test_no_rag_uses_exact_single_user_prompt(self):
        tokenizer = Mock(chat_template="enabled")
        expected = (
            "Answer this question :\\nWho wrote it?\\n"
            "Give me only the answer without including any other words.\\n\\nAnswer:"
        )
        rendered = render_prompt(tokenizer, preprocess_question(" Who wrote it "))
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": expected}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertIs(rendered, tokenizer.apply_chat_template.return_value)

    def test_vanilla_rag_preserves_reference_rank_and_newlines(self):
        tokenizer = Mock(chat_template="enabled")
        passages = [
            {"title": "Second alphabetically", "document": "First line\nNext line"},
            {"title": "Alpha", "document": "Other text"},
        ]
        expected = (
            'Answer the question based on the given document. Only give me the answer and do not output any other words."'
            "\\nThe following are given documents.\\n\\n"
            "(Title: Second alphabetically) First line\nNext line\n"
            "(Title: Alpha) Other text\n"
            "\\n\\nQuestion: Who wrote it?\\nAnswer:"
        )
        render_prompt(tokenizer, "Who wrote it?", passages)
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": expected}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenizer.chat_template = None
        tokenizer.apply_chat_template.reset_mock()
        self.assertEqual(render_prompt(tokenizer, "Who wrote it?", passages), expected)
        tokenizer.apply_chat_template.assert_not_called()

    def test_all_metrics_receive_the_unmodified_completion(self):
        response = "  <answer>Paris</answer>\n"
        results = [{"pure_art_answer": response, "golden_answers": ["Paris"]}]
        with (
            patch("REVIVE_training.evaluate_baseline.acc_score") as acc,
            patch("REVIVE_training.evaluate_baseline.f1_scorer") as f1,
            patch("REVIVE_training.evaluate_baseline.compute_exact") as em,
        ):
            score_baseline(results)
            for metric in (acc, f1, em):
                metric.assert_called_once_with([response], [["Paris"]])
        self.assertEqual(score_baseline(results)["metrics"]["EM"], 0.0)


class BaselineOutputTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".baseline-output-test-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.cleanup_output)

    def cleanup_output(self):
        for path in self.root.iterdir():
            path.unlink()
        self.root.rmdir()

    def test_success_replaces_only_the_selected_file(self):
        for filename in ("no_rag.jsonl", "vanilla_rag_top5.jsonl"):
            with self.subTest(filename=filename):
                output = self.root / filename
                output.write_text('{"old":1}\n', encoding="utf-8")
                sibling = self.root / "art_results.jsonl"
                sibling.write_text('{"art":1}\n', encoding="utf-8")
                before = set(self.root.iterdir())
                with fresh_output(output) as handle:
                    handle.write('{"fresh":1}\n')
                    self.assertEqual(output.read_text(encoding="utf-8"), '{"old":1}\n')
                self.assertEqual(output.read_text(encoding="utf-8"), '{"fresh":1}\n')
                self.assertEqual(sibling.read_text(encoding="utf-8"), '{"art":1}\n')
                self.assertEqual(set(self.root.iterdir()), before)

    def test_failed_run_preserves_previous_output_and_cleans_temporary_file(self):
        output = self.root / "no_rag.jsonl"
        output.write_text('{"old":1}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with fresh_output(output) as handle:
                handle.write('{"partial":1}\n')
                raise RuntimeError("generation failed")
        self.assertEqual(output.read_text(encoding="utf-8"), '{"old":1}\n')
        self.assertEqual(list(self.root.iterdir()), [output])

    def test_rejects_other_managed_outputs(self):
        paths = (
            "REVIVE_training/result/baselines/hotpotqa/no_rag.jsonl",
            "REVIVE_training/result/baselines/musique/vanilla_rag_top5.jsonl",
            "REVIVE_training/result/rsearch_musique_art_results.jsonl",
            "art-backup/demo-3/result/musique/threshold60.jsonl",
        )
        for path in paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_output(Path(path), "musique", "no_rag.jsonl")
        validate_output(
            Path("REVIVE_training/result/baselines/musique/no_rag.jsonl"),
            "musique",
            "no_rag.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
