from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from shared.MuSiQue.rsearch_musique_test_dataset import (
    load_rsearch_musique,
)


# ============================================================
# Configuration
# ============================================================

MODEL_REPO = "adaptive-retrieval-timing/ART-REVIVE"
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"

DTYPE = torch.bfloat16

MAX_NEW_TOKENS = 1024

DEMONSTRATION_FILE = Path(
    "REVIVE_training/result/"
    "teacher_hypothesis_demonstrations.jsonl"
)

OUTPUT_FILE = Path(
    "REVIVE_training/result/"
    "rsearch_musique_art_with_demonstrations_results.jsonl"
)

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
If additional information is needed, the system will perform retrieval and return the top search results within <observation> and </observation>. Continue your reasoning using the retrieved information.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""


# ============================================================
# Load demonstrations
# ============================================================

def load_demonstrations(
    path: Path = DEMONSTRATION_FILE,
) -> str:

    demonstrations = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            demonstrations.append(
                json.loads(line)
            )

    # --------------------------------------------------------
    # Sort by hop
    # --------------------------------------------------------

    demonstrations = sorted(
        demonstrations,
        key=lambda x: x["hop"],
    )

    # --------------------------------------------------------
    # Build demonstration prompt
    # --------------------------------------------------------

    parts = [
        "Here are some demonstration examples:"
    ]

    for i, sample in enumerate(
        demonstrations,
        start=1,
    ):

        parts.append(
            f"<demonstration {i}>\n"
            f"Question: {sample['question']}\n"
            f"Answer: {sample['teacher_generation_output']}\n"
            f"</demonstration {i}>"
        )

    parts.append(
        "Now solve the following question:"
    )

    return "\n\n".join(parts)


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. Load demonstrations
# ============================================================

demonstration_prompt = load_demonstrations()

print("=" * 80)
print("DEMONSTRATIONS")
print("=" * 80)
print(demonstration_prompt)
print()


# ============================================================
# 3. Load dataset
# ============================================================

dataset = load_rsearch_musique()


# ============================================================
# 4. Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_REPO,
    trust_remote_code=True,
)


# ============================================================
# 5. Load ART model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    dtype=DTYPE,
    attn_implementation="eager",
)

model = model.to(device)
model.eval()


# ============================================================
# 6. Prepare output directory
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 7. Process dataset
# ============================================================

total_samples = len(dataset)

print(
    f"Starting ART evaluation with demonstrations: "
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

        sample_id = sample.get(
            "id",
            sample_index - 1,
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"[{sample_index}/{total_samples}] "
            f"ID={sample_id}",
            flush=True,
        )

        # ----------------------------------------------------
        # Build user prompt
        #
        # Demonstrations are visible to the language model.
        # The original question is appended afterward.
        # ----------------------------------------------------

        user_prompt = (
            f"{demonstration_prompt}\n"
            f"{question}"
        )

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
        # Reset ART retrieval state
        # ----------------------------------------------------

        if hasattr(model, "query"):
            model.query = []

        if hasattr(model, "doc"):
            model.doc = []

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                question=question,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
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
        # Retrieve ART queries
        # ----------------------------------------------------

        if hasattr(model, "query") and model.query:

            queries = list(
                model.query
            )

        else:

            queries = []

        # ----------------------------------------------------
        # Retrieve ART documents
        # ----------------------------------------------------

        if hasattr(model, "doc") and model.doc:

            passages = list(
                model.doc
            )

        else:

            passages = []

        # ----------------------------------------------------
        # Build output sample
        # ----------------------------------------------------

        result = dict(sample)

        result["pure_art_answer"] = (
            generated_text
        )

        result["pure_art_passages"] = (
            passages
        )

        result["pure_art_queries"] = (
            queries
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
# 8. Finished
# ============================================================

print(
    f"Finished. Output: {OUTPUT_FILE}"
)