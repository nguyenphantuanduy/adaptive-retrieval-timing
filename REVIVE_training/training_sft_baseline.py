"""Teacher-forced Qwen SFT ablation with no retrieval components."""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

BACKBONE = "Qwen/Qwen2.5-3B-Instruct"
DATASET = "adaptive-retrieval-timing/MuSiQue-ART-Training-v2"
DATASET_REVISION = "aaf4b709de390e665047492016b9e8401b2def2d"
HUB_REPO = "adaptive-retrieval-timing/Qwen2.5-3B-Instruct-SFT-NoRAG"
SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""
MARKERS = re.compile(r"<\s*/?\s*(?:observation|tool_call|tool_response|search|retrieve|retrieval)\b|[\"']tool_calls[\"']\s*:", re.I)
EVIDENCE = re.compile(r"provided (?:documents|passages|evidence)|retrieved (?:documents|passages)|according to (?:the |document )?(?:passage|document)", re.I)


def render_prompt(tokenizer, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_sample(sample, tokenizer):
    target = sample["teacher_generation_output"]
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Invalid or empty teacher_generation_output")
    prompt = tokenizer(render_prompt(tokenizer, sample["question"]), add_special_tokens=False)["input_ids"]
    answer = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def collate_samples(samples, pad_id):
    length = max(len(row["input_ids"]) for row in samples)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in samples:
        size = len(row["input_ids"])
        padding = length - size
        batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
        batch["labels"].append(row["labels"] + [-100] * padding)
        batch["attention_mask"].append([1] * size + [0] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def freeze_backbone(model):
    model.requires_grad_(False)
    layers = model.model.layers
    trainable = math.ceil(0.40 * len(layers))
    for layer in layers[-trainable:]:
        layer.requires_grad_(True)
    return list(range(len(layers) - trainable, len(layers)))


def audit_targets(dataset):
    report = {"samples": len(dataset), "marker_samples": 0, "evidence_phrase_samples": 0, "marker_examples": []}
    for index, row in enumerate(dataset):
        target = row["teacher_generation_output"]
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Invalid teacher target at shuffled position {index}")
        match = MARKERS.search(target)
        if match:
            report["marker_samples"] += 1
            if len(report["marker_examples"]) < 10:
                report["marker_examples"].append({"position": index, "marker": match.group()})
        report["evidence_phrase_samples"] += bool(EVIDENCE.search(target))
    return report


def describe_lengths(lengths):
    ordered = sorted(lengths)
    return {"min": ordered[0], "median": (ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2,
            "p95": ordered[math.ceil(len(ordered)*0.95)-1], "max": ordered[-1], "total": sum(ordered)}


class CountingTrainer(Trainer):
    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        self.processed_samples += inputs["labels"].shape[0]
        self.supervised_tokens += int(inputs["labels"][:, 1:].ne(-100).sum().item())
        return loss


def smoke_check(model, tokenizer, row, output, device):
    batch = {k: v.to(device) for k, v in collate_samples([row], tokenizer.pad_token_id).items()}
    model.train()
    loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite smoke-test loss")
    loss.backward()
    if not any(p.grad is not None for p in model.parameters() if p.requires_grad):
        raise RuntimeError("No trainable gradient")
    for name, p in model.named_parameters():
        if p.grad is not None and (not p.requires_grad or not torch.isfinite(p.grad).all()):
            raise RuntimeError(f"Invalid gradient: {name}")
    model.zero_grad(set_to_none=True)
    model.config.use_cache = True
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return float(loss.detach())


def upload_model(model_dir, metrics_dir):
    from huggingface_hub import HfApi
    from REVIVE_training.evaluate_sft_baseline import DATASETS

    if not (model_dir / "training_configuration.json").is_file():
        raise ValueError("Expected the final SFT model and its training manifest")
    manifest = json.loads((model_dir / "training_configuration.json").read_text(encoding="utf-8"))
    if "optimizer_updates_total" not in manifest:
        raise ValueError("Smoke checkpoints cannot be published as the completed baseline")
    required = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"}
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if not weight_map:
            raise ValueError("Empty weight map in the model shard index")
        required.update(weight_map.values())
    else:
        required.add("model.safetensors")
    missing = sorted(name for name in required if not (model_dir / name).is_file())
    if missing:
        raise ValueError(f"Incomplete inference checkpoint: missing {', '.join(missing)}")
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    if not tokenizer_config.get("chat_template") and not (model_dir / "chat_template.jinja").is_file():
        raise ValueError("Incomplete inference checkpoint: missing the tokenizer chat template")
    results = []
    for dataset in DATASETS:
        metrics_path = metrics_dir / "sft-norag" / dataset / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics["num_samples"] != 500 or metrics["smoke_test"] or Path(metrics["model"]).resolve() != model_dir.resolve():
                continue
            scores = [f"{metrics['metrics'][key]:.2f}" for key in ("Acc", "F1", "EM")]
        except (OSError, ValueError, TypeError, KeyError):
            print(f"Skipping unreadable or invalid optional metrics: {metrics_path}")
            continue
        results.append(f"| {dataset} | {' | '.join(scores)} |")
    api = HfApi()
    orgs = api.whoami().get("orgs", [])
    if not any(org["name"] == HUB_REPO.split("/")[0] for org in orgs):
        raise PermissionError("Hugging Face account is not a member of the target organization")
    # The authenticated create call is the write permission check and refuses existing repositories.
    api.create_repo(HUB_REPO, repo_type="model", private=True, exist_ok=False)
    card = (
        f"---\nbase_model: {BACKBONE}\ndatasets:\n- {DATASET}\n---\n"
        "# Qwen2.5-3B-Instruct SFT No-RAG\n\n"
        "Pure teacher-forced SFT baseline for the ART ablation. Only the final 40% "
        "of decoder blocks (rounded up) are updated. The loss is causal LM cross-entropy "
        "on the original teacher target and EOS.\n\n"
    )
    if results:
        card += "| Dataset | Acc (%) | F1 (%) | EM (%) |\n| --- | ---: | ---: | ---: |\n"
        card += "\n".join(results) + "\n"
    else:
        card += "Benchmark results will be added after evaluation.\n"
    (model_dir / "README.md").write_text(card, encoding="utf-8")
    print(api.upload_folder(repo_id=HUB_REPO, folder_path=model_dir, repo_type="model"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("sft-norag-training-output"))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--backbone-revision")
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--accept-target-markers", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Publish the completed baseline to its own private repo")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload after successful training and final save")
    parser.add_argument("--metrics-dir", type=Path, default=Path("sft-norag-evaluation"))
    args = parser.parse_args()
    if args.push_to_hub and (args.upload or args.audit_only or args.smoke_test):
        parser.error("--push-to-hub requires full training")
    if args.upload:
        upload_model(args.output_dir / "model", args.metrics_dir)
        return
    if args.resume and (args.audit_only or args.smoke_test):
        parser.error("--resume applies only to full training")
    set_seed(42)
    splits = [load_dataset(DATASET, f"{hop}-hop", split="train", revision=args.dataset_revision)
              for hop in (2, 3, 4)]
    raw = concatenate_datasets(splits).shuffle(seed=42)
    audit = audit_targets(raw)
    audit["hop_counts"] = dict(zip(("2-hop", "3-hop", "4-hop"), map(len, splits)))
    audit["fingerprint"] = raw._fingerprint
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        return
    if audit["marker_samples"] and not args.accept_target_markers:
        parser.error("Teacher markers found. Review the audit before explicitly accepting unchanged targets.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() > 1:
        parser.error("Use one process to match the ART microbatch and workload accounting")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("A CUDA GPU with BF16 support is required")
    output = args.output_dir
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output directory already contains files. Use --resume or choose a new directory.")
    resume = get_last_checkpoint(str(output)) if args.resume == "latest" else args.resume
    if args.resume and not resume:
        parser.error("No checkpoint found for --resume")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, revision=args.backbone_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = raw.map(lambda row: tokenize_sample(row, tokenizer), remove_columns=raw.column_names)
    lengths = [len(x) for x in data["input_ids"]]
    target_lengths = [sum(token != -100 for token in x) for x in data["labels"]]
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE, revision=args.backbone_revision, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    if max(lengths) > model.config.max_position_embeddings:
        raise ValueError("A training sequence exceeds the model context; truncation would change the experiment.")
    selected = freeze_backbone(model)
    model.enable_input_require_grads()
    model.config.use_cache = False
    manifest = {"dataset": DATASET, "dataset_revision": args.dataset_revision, "backbone": BACKBONE,
                "backbone_revision": model.config._commit_hash or args.backbone_revision,
                "audit": audit, "trainable_layer_indices": selected,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
                "sequence_lengths": describe_lengths(lengths), "target_lengths": describe_lengths(target_lengths)}
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        row = data[0]
        cutoff = row["labels"].index(next(x for x in row["labels"] if x != -100))
        assert row["labels"][:cutoff] == [-100] * cutoff
        print("Prompt:", tokenizer.decode(row["input_ids"][:cutoff]))
        print("Supervised target:", tokenizer.decode(row["labels"][cutoff:]))
        model.gradient_checkpointing_enable()
        manifest["smoke_loss"] = smoke_check(model.cuda(), tokenizer, row, output / "model", "cuda")
        del model
        torch.cuda.empty_cache()
        restored = AutoModelForCausalLM.from_pretrained(output / "model", torch_dtype=torch.bfloat16).cuda().eval()
        inputs = tokenizer(render_prompt(tokenizer, raw[0]["question"]), return_tensors="pt").to("cuda")
        print(tokenizer.decode(restored.generate(**inputs, max_new_tokens=8)[0, inputs.input_ids.shape[1]:]))
    else:
        training = TrainingArguments(
            output_dir=str(output), num_train_epochs=0.75, per_device_train_batch_size=1,
            gradient_accumulation_steps=32, learning_rate=1e-5, weight_decay=0.0,
            max_grad_norm=1.0, bf16=True, fp16=False, gradient_checkpointing=True,
            lr_scheduler_type="linear", logging_steps=1, logging_first_step=True,
            save_steps=100, save_strategy="steps", eval_strategy="no", report_to="tensorboard",
            run_name="ART-SFT-NoRAG", dataloader_num_workers=0, dataloader_pin_memory=False,
            seed=42, data_seed=42, remove_unused_columns=False,
        )
        trainer = CountingTrainer(model=model, args=training, train_dataset=data,
                                  data_collator=lambda rows: collate_samples(rows, tokenizer.pad_token_id))
        trainer.processed_samples = trainer.supervised_tokens = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = trainer.train(resume_from_checkpoint=resume)
        manifest.update(training_arguments=training.to_dict(), optimizer_updates_total=result.global_step,
                        processed_samples_this_run=trainer.processed_samples,
                        supervised_tokens_this_run=trainer.supervised_tokens,
                        training_seconds_this_run=time.perf_counter()-start,
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        model.config.use_cache = True
        trainer.save_model(str(output / "model"))
        tokenizer.save_pretrained(output / "model")
        trainer.save_state()
    (output / "model" / "training_configuration.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.push_to_hub:
        upload_model(output / "model", args.metrics_dir)


if __name__ == "__main__":
    main()
