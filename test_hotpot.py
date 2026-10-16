
from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# Configuration
# ============================================================

MODEL_REPO = "adaptive-retrieval-timing/ART-REVIVE"
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"

DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512

QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
If additional information is needed, the system will perform retrieval and return the top search results within <observation> and </observation>. Continue your reasoning using the retrieved information.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type == "cuda" and not torch.cuda.is_bf16_supported():
    DTYPE = torch.float16


# ============================================================
# 2. Load tokenizer
# ============================================================

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_REPO,
    trust_remote_code=True,
)


# ============================================================
# 3. Load model
# ============================================================

print("Loading ART model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    dtype=DTYPE,
    attn_implementation="eager",
)

model = model.to(device)
model.eval()


# ============================================================
# 4. Build prompt
# ============================================================

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": QUESTION,
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

print("\n" + "=" * 80)
print("QUESTION")
print("=" * 80)
print(QUESTION)


# ============================================================
# 5. Reset ART retrieval state
# ============================================================

if hasattr(model, "query"):
    model.query = []

if hasattr(model, "doc"):
    model.doc = []


# ============================================================
# 6. Generate
# ============================================================

print("\nGenerating...", flush=True)

with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        question=QUESTION,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )


# ============================================================
# 7. Decode and print result
# ============================================================

prompt_length = input_ids.shape[1]

generated_ids = outputs[0, prompt_length:]

generated_text = tokenizer.decode(
    generated_ids,
    skip_special_tokens=False,
)

print("\n" + "=" * 80)
print("GENERATED RESULT")
print("=" * 80)
print(generated_text)

print("\n" + "=" * 80)
print("ART RETRIEVAL QUERIES")
print("=" * 80)
print(getattr(model, "query", []))

print("\n" + "=" * 80)
print("ART RETRIEVED DOCUMENTS")
print("=" * 80)
print(getattr(model, "doc", []))
