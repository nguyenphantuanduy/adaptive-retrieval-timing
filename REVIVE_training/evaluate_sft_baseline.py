"""Score standard Qwen generation on the exact archived DACN ART samples."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from REVIVE_training.evaluate_art_pure import acc_score, compute_exact, extract_answer, f1_scorer
from REVIVE_training.training_sft_baseline import BACKBONE, render_prompt

DATASETS = ("musique", "hotpotqa", "2wikimultihopqa")
ARCHIVE = Path("art-backup/demo-3/result")


def read_samples(dataset):
    source = ARCHIVE / dataset / f"rsearch_{dataset}_art_top5_results.jsonl"
    with source.open(encoding="utf-8") as stream:
        rows = [{key: row[key] for key in ("id", "question", "golden_answers")}
                for line in stream if line.strip() for row in [json.loads(line)]]
    if len(rows) != 500 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 500 distinct archived samples in {source}")
    return source, rows


def score_rows(rows):
    predictions = [row["extracted_answer"] for row in rows]
    gold = [row["golden_answers"] for row in rows]
    tags = sum(row["has_answer_tag"] for row in rows)
    return {"num_samples": len(rows), "answer_extraction": {"has_answer_tag": tags,
            "no_answer_tag": len(rows) - tags, "has_answer_tag_percent": round(100 * tags / len(rows), 2),
            "no_answer_tag_percent": round(100 * (len(rows) - tags) / len(rows), 2)},
            "metrics": {"Acc": acc_score(predictions, gold), "F1": f1_scorer(predictions, gold),
                        "EM": compute_exact(predictions, gold)}}


def write_summary(output_dir):
    summary = {}
    for dataset in DATASETS:
        records = {
            "original-norag": output_dir / "original-norag" / dataset / "metrics.json",
            "sft-norag": output_dir / "sft-norag" / dataset / "metrics.json",
            "art": Path("art-backup/demo-3/metrics") / dataset / f"rsearch_{dataset}_art_top5_metrics.json",
        }
        summary[dataset] = {}
        for arm, path in records.items():
            if not path.exists():
                summary[dataset][arm] = {"status": "missing", "source": str(path)}
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if metrics["num_samples"] != 500 or metrics.get("smoke_test"):
                raise ValueError(f"Unexpected sample count or smoke metrics in {path}")
            summary[dataset][arm] = {"source": str(path), **metrics["metrics"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def load_checkpoint(source, revision, dtype, local_files_only=False):
    model = AutoModelForCausalLM.from_pretrained(
        source, revision=revision, torch_dtype=dtype, local_files_only=local_files_only
    )
    resolved_revision = model.config._commit_hash or revision
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=resolved_revision, local_files_only=local_files_only
    )
    return model, tokenizer, resolved_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--model", default="sft-norag-training-output/model",
                        help="Hugging Face model repository ID or local checkpoint directory")
    parser.add_argument("--revision", help="Hugging Face branch, tag or commit SHA")
    parser.add_argument("--local-files-only", action="store_true", help="Load from disk or the local Hub cache only")
    parser.add_argument("--output-dir", type=Path, default=Path("sft-norag-evaluation"))
    parser.add_argument("--limit", type=int, choices=range(1, 500), metavar="1..499")
    parser.add_argument("--cpu-smoke", action="store_true", help="Test generation and scoring with a small local CPU model")
    args = parser.parse_args()
    if args.summary:
        write_summary(args.output_dir)
        return
    if not args.dataset:
        parser.error("--dataset is required for generation")
    source, samples = read_samples(args.dataset)
    if args.cpu_smoke and not args.limit:
        parser.error("--cpu-smoke requires --limit")
    if not args.cpu_smoke and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error("Use CUDA with BF16 for the archived ART generation settings")
    device = "cpu" if args.cpu_smoke else "cuda"
    arm = "original-norag" if args.model == BACKBONE else "sft-norag"
    output = args.output_dir / arm / args.dataset
    if args.limit:
        output = output / f"smoke-{args.limit}"
        samples = samples[:args.limit]
    if (output / "results.jsonl").exists() or (output / "metrics.json").exists():
        parser.error("Evaluation outputs already exist. Choose another --output-dir.")
    model, tokenizer, resolved_revision = load_checkpoint(
        args.model, args.revision, torch.float32 if args.cpu_smoke else torch.bfloat16, args.local_files_only
    )
    model = model.to(device).eval()
    generation = GenerationConfig(do_sample=True, temperature=0.1, top_p=0.9, top_k=0,
                                  repetition_penalty=1.0, max_new_tokens=2 if args.cpu_smoke else 1024, num_beams=1,
                                  num_return_sequences=1, eos_token_id=model.generation_config.eos_token_id,
                                  pad_token_id=tokenizer.pad_token_id)
    set_seed(44)
    rows = []
    with torch.inference_mode():
        for sample in samples:
            inputs = tokenizer(render_prompt(tokenizer, sample["question"]),
                               add_special_tokens=False, return_tensors="pt").to(device)
            response = model.generate(**inputs, generation_config=generation)
            raw = tokenizer.decode(response[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
            answer, has_tag = extract_answer(raw)
            rows.append({**sample, "pure_art_answer": raw, "extracted_answer": answer,
                         "has_answer_tag": has_tag, "pure_art_queries": [], "pure_art_passages": []})
    metrics = score_rows(rows)
    metrics.update(source=str(source), model=args.model, revision=resolved_revision,
                   seed=44, generation=generation.to_dict(), smoke_test=args.limit is not None)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.jsonl").open("x", encoding="utf-8") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with (output / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics["metrics"], indent=2))


if __name__ == "__main__":
    main()
