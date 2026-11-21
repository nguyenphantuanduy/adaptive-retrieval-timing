"""Shared generation pipeline for the two Qwen evaluation baselines."""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from REVIVE_training.utils import seed_everything

MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct"
SEED = 44
DATASETS = ("musique", "hotpotqa", "2wikimultihopqa")
OUTPUT_ROOT = Path("REVIVE_training/result/baselines")
REPO_ROOT = Path(__file__).resolve().parent.parent

# R-Search src/eval/config.yaml uses single-quoted YAML scalars, so \n
# remains literal in the templates. The extra double quote is intentional.
NO_RAG_TEMPLATE = (
    r"Answer this question :\n{question}\nGive me only the answer without including any other words.\n\nAnswer:"
)
VANILLA_RAG_TEMPLATE = (
    r'Answer the question based on the given document. Only give me the answer and do not output any other words."'
    r'\nThe following are given documents.\n\n{refs}\n\nQuestion: {question}\nAnswer:'
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


def preprocess_question(question):
    question = question.strip()
    return question if question.endswith("?") else question + "?"


def build_messages(question, passages=None):
    if passages is None:
        prompt = NO_RAG_TEMPLATE.format(question=question)
    else:
        reference = "".join(
            f"(Title: {hit['title']}) {hit['document']}\n"
            for hit in passages
        )
        prompt = VANILLA_RAG_TEMPLATE.format(refs=reference, question=question)
    return [{"role": "user", "content": prompt}]


def render_prompt(tokenizer, question, passages=None):
    messages = build_messages(question, passages)
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"]


def validate_output(output, dataset, filename):
    if output.is_symlink():
        raise ValueError(f"--output cannot be a symbolic link: {output}")

    target = output.resolve()
    for root in (REPO_ROOT, Path.cwd()):
        baselines = (root / OUTPUT_ROOT).resolve()
        if target.is_relative_to(baselines):
            relative = target.relative_to(baselines)
            wrong_dataset = bool(relative.parts) and relative.parts[0] in DATASETS and relative.parts[0] != dataset
            other_managed_output = (
                target.name in {"no_rag.jsonl", "vanilla_rag_top5.jsonl"}
                and target != baselines / dataset / filename
            )
            if wrong_dataset or other_managed_output:
                raise ValueError(f"--output belongs to a different baseline or dataset: {output}")
        elif any(
            target.is_relative_to((root / directory).resolve())
            for directory in (OUTPUT_ROOT.parent, Path("art-backup"), Path("temp_model"))
        ):
            raise ValueError(f"--output points into an ART result directory: {output}")


@contextmanager
def fresh_output(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            yield handle
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_baseline(filename, retrieve_for_dataset=None):
    parser = argparse.ArgumentParser(description=f"Generate {Path(filename).stem} baseline results.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, help="Use the first N samples in ART dataset order.")
    parser.add_argument("--output", type=Path, help="Baseline JSONL path to replace on success.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    output = args.output or OUTPUT_ROOT / args.dataset / filename
    if output.suffix.lower() != ".jsonl":
        parser.error("--output must end in .jsonl")
    try:
        validate_output(output, args.dataset, filename)
    except ValueError as error:
        parser.error(str(error))

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
    with fresh_output(output) as handle, torch.inference_mode():
        for index, sample in enumerate(samples):
            question = preprocess_question(sample["question"])
            sample["golden_answers"]  # Require the existing scorer's gold field.
            passages = retrieve(question) if retrieve is not None else None
            prompt = render_prompt(tokenizer, question, passages)
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
