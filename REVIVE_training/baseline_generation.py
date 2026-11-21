"""Shared generation pipeline for the two Qwen evaluation baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from REVIVE_training.utils import seed_everything

MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct"
SEED = 44
DATASETS = ("musique", "hotpotqa", "2wikimultihopqa")
OUTPUT_ROOT = Path("REVIVE_training/result/baselines")

# FlashRAG English quick-start demo, including its literal whitespace.
ASSISTANT_PROMPT = (
    "You are a friendly AI Assistant."
    "Respond to the input as a friendly AI assistant, "
    "generating human-like text, and follow the "
    "instructions in the input if applicable."
)
RAG_REFERENCES = (
    "\nThe following are provided references. "
    "You can use them for answering question.\n\n"
)


def load_dataset(dataset):
    if dataset == "musique":
        from shared.MuSiQue.rsearch_musique_test_dataset import load_rsearch_musique

        return load_rsearch_musique()
    if dataset == "hotpotqa":
        from shared.HotpotQA_Distractor.rsearch_hotpotqa_test_dataset import load_rsearch_hotpotqa

        return load_rsearch_hotpotqa()
    if dataset == "2wikimultihopqa":
        from shared.TwoWikiMultiHopQA.rsearch_2wikimultihopqa_test_dataset import load_rsearch_2wikimultihopqa

        return load_rsearch_2wikimultihopqa()
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_messages(question, passages=None):
    if passages is None:
        system_prompt = ASSISTANT_PROMPT + "\n"
    else:
        reference = "".join(
            f"Doc {rank}(Title: {hit['title']}) {hit['document']}\n"
            for rank, hit in enumerate(passages, start=1)
        )
        system_prompt = ASSISTANT_PROMPT + RAG_REFERENCES + reference
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def run_baseline(filename, retrieve_for_dataset=None):
    parser = argparse.ArgumentParser(description=f"Generate {Path(filename).stem} baseline results.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, help="Use the first N samples in ART dataset order.")
    parser.add_argument("--output", type=Path, help="Path to a new JSONL file.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    output = args.output or OUTPUT_ROOT / args.dataset / filename
    if output.suffix.lower() != ".jsonl":
        parser.error("--output must end in .jsonl")
    if output.exists() or output.is_symlink():
        parser.error(f"output already exists: {output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    samples = load_dataset(args.dataset)[:args.limit]
    if not samples:
        parser.error("the selected dataset contains no samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
    ).to(device)
    model.eval()
    retrieve = retrieve_for_dataset(args.dataset) if retrieve_for_dataset is not None else None
    config = GenerationConfig(
        do_sample=True, temperature=0.1, top_p=0.9, top_k=0,
        repetition_penalty=1.0, max_new_tokens=1024, num_beams=1, num_return_sequences=1,
    )
    seed_everything(SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for index, sample in enumerate(samples):
            question = sample["question"]
            sample["golden_answers"]  # Require the existing scorer's gold field.
            passages = retrieve(question) if retrieve is not None else None
            prompt = tokenizer.apply_chat_template(
                build_messages(question, passages), tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(**inputs, generation_config=config)
            answer = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False
            )

            result = dict(sample)
            result.setdefault("id", index)
            result["pure_art_answer"] = answer
            result["pure_art_queries"] = [question] if passages is not None else []
            result["pure_art_passages"] = [
                f"Title: {hit['title']}\n{hit['document']}" for hit in passages or []
            ]
            result["pure_art_retrieval_scores"] = []
            if passages is not None:
                result["retrieved_passages"] = passages
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{index + 1}/{len(samples)}] ID={result['id']}", flush=True)
    print(f"Saved {len(samples)} samples to {output}")
