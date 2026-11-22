"""Generate once at every ART retrieval-decision level.

Run from the repository root after publishing the custom Qwen2.5 probe:

    .\\.venv\\Scripts\\python.exe test\\test_art_attention_mixing_probe_generation.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.model_input import (
    Document,
    NUM_DECISION_LEVELS,
    SampleObject,
    create_model_input,
)


MODEL_REPO = "adaptive-retrieval-timing/ART-Attention-Mixing-Probe"
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 256


def get_input_device(model) -> torch.device:
    """Return the device that receives input embeddings, including device_map='auto'."""
    return model.get_input_embeddings().weight.device


def main() -> None:
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else "cpu"

    print("=" * 80)
    print("ART ATTENTION-MIXING PROBE: GENERATION ACROSS RETRIEVAL LEVELS")
    print("=" * 80)

    print("\nLoading tokenizer...")
    # The published probe contains custom model code/weights. Its tokenizer is
    # the unchanged tokenizer of the original Qwen2.5-3B-Instruct checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading published ART probe...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    sample = SampleObject(
        question=(
            "According to the diary entry, where did Elena hide the brass key?"
        ),
        documents=[
            Document(
                title="Elena's diary — 14 October",
                content=(
                    "The rain started just after supper, so I cancelled my walk to "
                    "the station. Before bed, I wrapped the brass key in a blue "
                    "handkerchief and tucked it inside the third cookbook on the "
                    "kitchen shelf. I must remember this when Uncle Marin arrives "
                    "on Friday."
                ),
            ),
        ],
    )

    model_input = create_model_input(sample, tokenizer)
    input_device = get_input_device(model)
    prompt_length = model_input["input_ids"].shape[1]

    print(f"\nModel class: {type(model).__name__}")
    print(f"Input device: {input_device}")
    print(f"Prompt tokens: {prompt_length}")
    print("\nFull input text (shared by every retrieval-decision level):")
    print("-" * 80)
    print(model_input["full_text"])
    print("-" * 80)

    # Generate sequentially to keep memory bounded. Every row comes from the
    # same input sequence and differs only in retrieval_decision.
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

            completion_ids = generated_ids[0, prompt_length:]
            completion = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            ).strip()

            print()
            print("=" * 80)
            print(
                f"retrieval_decision = {decision_level}/{NUM_DECISION_LEVELS} "
                f"({decision_level / NUM_DECISION_LEVELS:.5f})"
            )
            print("-" * 80)
            print(completion or "[empty completion]")

            del generated_ids

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
