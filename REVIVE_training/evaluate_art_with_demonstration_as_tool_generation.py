from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# Configuration
# ============================================================

MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct"

DTYPE = torch.bfloat16

MAX_NEW_TOKENS = 512

INPUT_FILE = Path(
    "REVIVE_training/result/"
    "rsearch_musique_art_with_demonstrations_results.jsonl"
)

OUTPUT_FILE = Path(
    "REVIVE_training/result/"
    "rsearch_musique_art_with_demonstrations_as_tool_results.jsonl"
)

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
Documents retrieved by another system are provided within <documents> and </documents> tags. Use the information from these documents when needed to help answer the question.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. Load input JSONL
# ============================================================

dataset = []

with INPUT_FILE.open(
    "r",
    encoding="utf-8",
) as input_fp:

    for line in input_fp:
        line = line.strip()

        if not line:
            continue

        dataset.append(
            json.loads(line)
        )


# ============================================================
# 3. Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
)


# ============================================================
# 4. Load Qwen reader
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    dtype=DTYPE,
    trust_remote_code=True,
    attn_implementation="eager",
)

model = model.to(device)
model.eval()


# ============================================================
# 5. Prepare output directory
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 6. Process dataset
# ============================================================

total_samples = len(dataset)

print(
    f"Starting ART-as-a-tool evaluation: "
    f"{total_samples} samples"
)


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as output_fp:

    for sample_index, sample in enumerate(
        dataset,
        start=1,
    ):

        question = sample["question"]

        passages = sample.get(
            "pure_art_passages",
            [],
        )

        sample_id = sample.get(
            "id",
            sample_index - 1,
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"[{sample_index}/{total_samples}] "
            f"ID={sample_id} "
            f"passages={len(passages)}",
            flush=True,
        )

        # ----------------------------------------------------
        # Build document context
        # ----------------------------------------------------

        context = "\n\n".join(
            passages
        )

        # ----------------------------------------------------
        # Build user prompt
        # ----------------------------------------------------

        user_prompt = f"""Question:
{question}

Given documents:
<documents>
{context}
</documents>"""

        # ----------------------------------------------------
        # Build chat prompt
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = inputs[
            "input_ids"
        ].to(device)

        attention_mask = inputs[
            "attention_mask"
        ].to(device)

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # ----------------------------------------------------
        # Decode generated response
        # ----------------------------------------------------

        prompt_length = input_ids.shape[1]

        generated_ids = outputs[
            0,
            prompt_length:,
        ]

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )

        # ----------------------------------------------------
        # Build output sample
        # ----------------------------------------------------

        result = dict(sample)

        result["art_as_tool_answer"] = (
            generated_text
        )

        # ----------------------------------------------------
        # Write JSONL
        # ----------------------------------------------------

        output_fp.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )

        output_fp.flush()


# ============================================================
# 7. Finished
# ============================================================

print(
    f"Finished. Output: {OUTPUT_FILE}"
)