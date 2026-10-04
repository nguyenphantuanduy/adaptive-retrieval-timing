from __future__ import annotations

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
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
If additional information is needed, the system will perform retrieval and return the top search results within <observation> and </observation>. Continue your reasoning using the retrieved information.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 80)
print("ART SINGLE-SAMPLE TEST")
print("=" * 80)
print(f"Device: {device}")


# ============================================================
# 2. Load dataset
# ============================================================

dataset = load_rsearch_musique()

sample = dataset[0]

question = sample["question"]
sample_id = sample.get("id", 0)

print(f"Sample ID: {sample_id}")
print(f"Question: {question}")


# ============================================================
# 3. Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_REPO,
    trust_remote_code=True,
)


# ============================================================
# 4. Load ART model
# ============================================================

print("Loading ART model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    force_download=True,
    revision="main",
    dtype=DTYPE,
    attn_implementation="eager",
)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# 5. Build chat messages
# ============================================================

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


# ============================================================
# 6. Tokenize
#    SAME approach as working Kaggle cell
# ============================================================

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)


# ------------------------------------------------------------
# Handle Tensor / BatchEncoding
# ------------------------------------------------------------

if isinstance(inputs, torch.Tensor):

    input_ids = inputs

    # Same behavior as Kaggle:
    # all current prompt tokens are valid tokens.
    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.long,
    )

else:

    input_ids = inputs["input_ids"]

    attention_mask = inputs.get(
        "attention_mask",
        None,
    )

    # Safety fallback
    if attention_mask is None:
        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
        )


# ------------------------------------------------------------
# Move to device
# ------------------------------------------------------------

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)


# ============================================================
# 7. Debug tokenizer output
# ============================================================

print("\nTokenizer outputs:")

print(
    "  input_ids:",
    input_ids.shape,
)

print(
    "  attention_mask:",
    attention_mask.shape,
)

print(
    "  attention_mask dtype:",
    attention_mask.dtype,
)

print(
    "  attention_mask device:",
    attention_mask.device,
)

print(
    "  attention_mask unique:",
    attention_mask.unique(),
)


# ============================================================
# 8. Reset ART retrieval state
# ============================================================

if hasattr(model, "query"):
    model.query = []

if hasattr(model, "doc"):
    model.doc = []


# ============================================================
# 9. Generation
# ============================================================

print("\nGenerating...")

with torch.inference_mode():

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        question=question,
        max_new_tokens=MAX_NEW_TOKENS,
    )


# ============================================================
# 10. Decode generated response
# ============================================================

prompt_length = input_ids.shape[1]

generated_ids = outputs[
    0,
    prompt_length:,
]

generated_text = tokenizer.decode(
    generated_ids,
    skip_special_tokens=False,
)


# ============================================================
# 11. Retrieve ART queries
# ============================================================

if hasattr(model, "query") and model.query:
    queries = list(model.query)
else:
    queries = []


# ============================================================
# 12. Retrieve ART documents
# ============================================================

if hasattr(model, "doc") and model.doc:
    passages = list(model.doc)
else:
    passages = []


# ============================================================
# 13. Print result
# ============================================================

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print("\nQuestion:")
print(question)

print("\nGenerated answer:")
print(generated_text)

print("\nART queries:")

for i, query in enumerate(
    queries,
    start=1,
):
    print(f"  [{i}] {query}")


print(
    f"\nART passages: {len(passages)}"
)

for i, passage in enumerate(
    passages,
    start=1,
):
    print(f"\n--- Passage {i} ---")
    print(passage)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)