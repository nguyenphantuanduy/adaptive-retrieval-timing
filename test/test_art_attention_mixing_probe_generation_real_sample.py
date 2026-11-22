"""Generate at every retrieval level using a real MuSiQue validation sample.

Run from the repository root:

    .\\.venv\\Scripts\\python.exe -m test.test_art_attention_mixing_probe_generation_real_sample

Select another source sample with ``--index``. This is generation-only: the
dataset answer and teacher_generation_output are never read or passed to the
model.
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.model_input import (
    Document,
    NUM_DECISION_LEVELS,
    SampleObject,
    create_model_input,
)


MODEL_REPO = "adaptive-retrieval-timing/ART-Attention-Mixing-Probe"
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"
DATASET_REPO = "adaptive-retrieval-timing/teacher-generated-hypothesis-validation"
CONFIG_NAME = "2-hop"
SPLIT = "validation"
MAX_DOCUMENTS = 3
MAX_NEW_TOKENS = 512


def get_input_device(model) -> torch.device:
    """Return the device holding input embeddings, including device_map='auto'."""
    return model.get_input_embeddings().weight.device


def select_documents(sample: dict) -> list[Document]:
    """Use up to three real passages, prioritizing supporting evidence."""
    paragraphs = sample["paragraphs"]
    supporting = [paragraph for paragraph in paragraphs if paragraph["is_supporting"]]
    non_supporting = [
        paragraph for paragraph in paragraphs if not paragraph["is_supporting"]
    ]
    selected = (supporting + non_supporting)[:MAX_DOCUMENTS]

    if not selected:
        raise ValueError("The selected dataset sample has no paragraphs.")

    return [
        Document(
            title=paragraph["title"],
            content=paragraph["paragraph_text"],
        )
        for paragraph in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the 2-hop validation sample to generate from.",
    )
    args = parser.parse_args()

    print("Loading real MuSiQue sample...")
    dataset = load_dataset(DATASET_REPO, name=CONFIG_NAME, split=SPLIT)
    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"--index must be in [0, {len(dataset) - 1}], got {args.index}."
        )

    source_sample = dataset[args.index]
    sample = SampleObject(
        question=source_sample["question"],
        documents=select_documents(source_sample),
    )

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else "cpu"

    print("Loading tokenizer and published ART probe...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    model_input = create_model_input(sample, tokenizer)
    input_device = get_input_device(model)
    prompt_length = model_input["input_ids"].shape[1]

    print("=" * 100)
    print("ART ATTENTION-MIXING PROBE: REAL MUSIQUE GENERATION SAMPLE")
    print("=" * 100)
    print(f"Dataset: {DATASET_REPO} ({CONFIG_NAME}/{SPLIT})")
    print(f"Sample index: {args.index}")
    print(f"Sample id: {source_sample.get('id', '<missing>')}")
    print(f"Question: {sample.question}")
    print(f"Documents used: {len(sample.documents)}")
    for index, document in enumerate(sample.documents, start=1):
        print(f"  {index}. {document.title}")
    print(f"Prompt tokens: {prompt_length}")
    print("\nFull input text (shared by every retrieval-decision level):")
    print("-" * 100)
    print(model_input["full_text"])
    print("-" * 100)

    # Generate sequentially: all inputs are identical except retrieval_decision.
    with torch.inference_mode():
        for decision_level in range(NUM_DECISION_LEVELS + 1):
            row = {
                name: value[decision_level : decision_level + 1].to(input_device)
                for name, value in model_input.items()
                if torch.is_tensor(value)
            }

            generated_ids = model.generate(
                input_ids=row["input_ids"],
                attention_mask=row["attention_mask"],
                document_attention_mask=row["document_attention_mask"],
                retrieval_decision=row["retrieval_decision"],
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            completion = tokenizer.decode(
                generated_ids[0, prompt_length:].cpu(),
                skip_special_tokens=True,
            ).strip()

            print("\n" + "=" * 100)
            print(
                f"retrieval_decision = {decision_level}/{NUM_DECISION_LEVELS} "
                f"({decision_level / NUM_DECISION_LEVELS:.5f})"
            )
            print("-" * 100)
            print(completion or "[empty completion]")

            del generated_ids

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
