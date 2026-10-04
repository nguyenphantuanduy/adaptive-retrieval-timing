
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

else:

    input_ids = inputs["input_ids"]


# ------------------------------------------------------------
# Move input_ids to device
# ------------------------------------------------------------

input_ids = input_ids.to(device)


# ============================================================
# 7. DEBUG ATTENTION MASK
#
# IMPORTANT:
# This is intentionally NOT the normal all-ones mask.
#
# Pattern:
#
#   1 1 1 1 1  0 0 0 0 ... 0  1 1 1 1 1
#
# This is ONLY for debugging whether the mask survives
# the GenerationMixin -> _prefill -> prepare_inputs_for_generation
# -> forward pipeline.
# ============================================================

attention_mask = torch.zeros_like(
    input_ids,
    dtype=torch.long,
)

# First 5 tokens are valid
attention_mask[:, :5] = 1

# Last 5 tokens are valid
attention_mask[:, -5:] = 1

attention_mask = attention_mask.to(device)


# ============================================================
# 8. Debug tokenizer / input tensors
# ============================================================

print("\n" + "=" * 80)
print("INPUT DEBUG")
print("=" * 80)

print(
    "input_ids:",
    input_ids.shape,
)

print(
    "attention_mask:",
    attention_mask.shape,
)

print(
    "attention_mask dtype:",
    attention_mask.dtype,
)

print(
    "attention_mask device:",
    attention_mask.device,
)

print(
    "attention_mask unique:",
    attention_mask.unique(),
)

print(
    "attention_mask:",
    attention_mask,
)

print(
    "attention_mask sum:",
    attention_mask.sum().item(),
)

print(
    "attention_mask expected sum:",
    min(10, input_ids.shape[1]),
)


# ============================================================
# 9. Verify prepare_inputs_for_generation DIRECTLY
#
# This checks the function BEFORE generate() is involved.
# ============================================================

print("\n" + "=" * 80)
print("DIRECT prepare_inputs_for_generation DEBUG")
print("=" * 80)

try:

    prepared_inputs = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        is_first_iteration=True,
    )

    print(
        "prepared_inputs keys:",
        prepared_inputs.keys(),
    )

    prepared_attention_mask = prepared_inputs.get(
        "attention_mask",
        None,
    )

    print(
        "prepared attention_mask:",
        prepared_attention_mask,
    )

    if prepared_attention_mask is not None:

        print(
            "prepared attention_mask shape:",
            prepared_attention_mask.shape,
        )

        print(
            "prepared attention_mask dtype:",
            prepared_attention_mask.dtype,
        )

        print(
            "prepared attention_mask device:",
            prepared_attention_mask.device,
        )

        print(
            "prepared attention_mask unique:",
            prepared_attention_mask.unique(),
        )

        print(
            "prepared attention_mask sum:",
            prepared_attention_mask.sum().item(),
        )

    else:

        print(
            "WARNING: prepare_inputs_for_generation "
            "returned attention_mask=None"
        )

except Exception as e:

    print(
        "prepare_inputs_for_generation FAILED:"
    )

    print(
        type(e).__name__,
        str(e),
    )


# ============================================================
# 10. Reset ART retrieval state
# ============================================================

if hasattr(model, "query"):
    model.query = []

if hasattr(model, "doc"):
    model.doc = []


# ============================================================
# 11. Generation
# ============================================================

print("\n" + "=" * 80)
print("GENERATING")
print("=" * 80)

with torch.inference_mode():

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        question=question,
        max_new_tokens=MAX_NEW_TOKENS,
    )


# ============================================================
# 12. Decode generated response
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
# 13. Retrieve ART queries
# ============================================================

if hasattr(model, "query") and model.query:

    queries = list(model.query)

else:

    queries = []


# ============================================================
# 14. Retrieve ART documents
# ============================================================

if hasattr(model, "doc") and model.doc:

    passages = list(model.doc)

else:

    passages = []


# ============================================================
# 15. Print result
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

    print(
        f"  [{i}] {query}"
    )


print(
    f"\nART passages: {len(passages)}"
)

for i, passage in enumerate(
    passages,
    start=1,
):

    print(
        f"\n--- Passage {i} ---"
    )

    print(passage)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
