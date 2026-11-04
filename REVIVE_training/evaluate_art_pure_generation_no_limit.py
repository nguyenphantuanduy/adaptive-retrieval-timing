
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
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

SEED = 44
MAX_NEW_TOKENS = 1024

OUTPUT_FILE = Path(
    "REVIVE_training/result/rsearch_musique_art_results.jsonl"
)

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
If additional information is needed, the system will perform retrieval and return the top search results within <observation> and </observation>. Continue your reasoning using the retrieved information.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""


# ============================================================
# 1. Seed
# ============================================================

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Generation Configuration
# R-Search evaluation-aligned settings
# ============================================================

generation_config = GenerationConfig(
    # Sampling
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    top_k=0,

    # Penalties (vLLM defaults)
    repetition_penalty=1.0,

    # Output
    max_new_tokens=MAX_NEW_TOKENS,
    num_beams=1,
    num_return_sequences=1,
)


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. Load dataset
# ============================================================

dataset = load_rsearch_musique()


# ============================================================
# 5. Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_REPO,
    trust_remote_code=True,
)


# ============================================================
# 6. Load ART model
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
# 7. Prepare output directory
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 8. Initialize random seed
# Match R-Search: initialize once before evaluation
# ============================================================

seed_everything(SEED)


# ============================================================
# 9. Process dataset
# ============================================================

total_samples = len(dataset)

print("=" * 80)
print("ART EVALUATION")
print("=" * 80)

print(f"Model          : {MODEL_REPO}")
print(f"Tokenizer      : {TOKENIZER_REPO}")
print(f"Dataset        : MuSiQue R-Search")
print(f"Device         : {device}")
print(f"Seed           : {SEED}")
print(f"Max new tokens : {MAX_NEW_TOKENS}")

print("\nGeneration configuration:")
print(generation_config)

print("=" * 80)

print(
    f"Starting ART evaluation: {total_samples} samples"
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
        # Build chat prompt
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
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

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # ----------------------------------------------------
        # Reset ART retrieval state
        # ----------------------------------------------------

        if hasattr(model, "query"):
            model.query = []

        if hasattr(model, "doc"):
            model.doc = []

        if hasattr(model, "retrieval_score"):
            model.retrieval_score = []

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                question=question,
                generation_config=generation_config,
                max_retrieval_context_tokens=-1,
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
            queries = list(model.query)
        else:
            queries = []

        # ----------------------------------------------------
        # Retrieve ART documents
        # ----------------------------------------------------

        if hasattr(model, "doc") and model.doc:
            passages = list(model.doc)
        else:
            passages = []

        # ----------------------------------------------------
        # Retrieve ART scores
        # ----------------------------------------------------

        if hasattr(model, "retrieval_score") and model.retrieval_score:
            retrieval_scores = list(model.retrieval_score)
        else:
            retrieval_scores = []

        # ----------------------------------------------------
        # Build output sample
        # ----------------------------------------------------

        result = dict(sample)

        result["pure_art_answer"] = generated_text
        result["pure_art_passages"] = passages
        result["pure_art_queries"] = queries
        result["pure_art_retrieval_scores"] = retrieval_scores

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
# 10. Finished
# ============================================================

print("=" * 80)
print("Evaluation finished.")
print(f"Output: {OUTPUT_FILE}")
print("=" * 80)
