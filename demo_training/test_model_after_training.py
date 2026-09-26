from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Configuration
# =============================================================================

# Model saved after ART training.
MODEL_PATH = "adaptive-retrieval-timing/ART-Demo-Qwen2.5-3B"

# Tokenizer was also saved to the same directory after training.
TOKENIZER_PATH = "./art-training-output"

DTYPE = torch.bfloat16

QUESTION = (
    "How many times did plague occur in the place where the creator of The Worship of Venus died?"
)

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
If additional information is needed, the system will perform retrieval and return the top search results within <observation> and </observation>. Continue your reasoning using the retrieved information.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("ART TRAINED MODEL GENERATION TEST")
    print("=" * 80)

    # =========================================================================
    # 1. Device
    # =========================================================================

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    # =========================================================================
    # 2. Load tokenizer
    # =========================================================================

    print("\n1. Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
    )

    print(f"   Tokenizer: {TOKENIZER_PATH}")

    # =========================================================================
    # 3. Build chat
    # =========================================================================

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

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(inputs, torch.Tensor):

        input_ids = inputs

        attention_mask = torch.ones_like(
            input_ids,
            dtype=torch.long,
        )

    else:

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # =========================================================================
    # 4. Print formatted prompt
    # =========================================================================

    print("\n2. Formatted chat prompt:")
    print("-" * 80)

    formatted_prompt = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=False,
    )

    print(formatted_prompt)

    print("-" * 80)

    print(
        f"Prompt tokens: {input_ids.shape[1]}"
    )

    # =========================================================================
    # 5. Load TRAINED ART model
    # =========================================================================

    print("\n3. Loading TRAINED ART model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        attn_implementation="eager",
    )

    model = model.to(device)
    model.eval()

    print(
        f"   Model: {MODEL_PATH}"
    )

    print(
        f"   Model class: {type(model).__name__}"
    )

    print(
        "   Model dtype: "
        f"{model.model.embed_tokens.weight.dtype}"
    )

    # =========================================================================
    # 6. Verify TRAINED ART decision head
    # =========================================================================
    #
    # IMPORTANT:
    #
    # Do NOT initialize the decision head here.
    #
    # The trained retrieval decision head is already stored inside
    # ./art-training-output by trainer.save_model().
    #
    # =========================================================================

    print("\n4. Verifying TRAINED ART decision head...")

    decision_projection = (
        model.retrieval_decision_head.projection
    )

    decision_weight = (
        decision_projection.weight
        .detach()
        .float()
        .cpu()
    )

    decision_bias = (
        decision_projection.bias
        .detach()
        .float()
        .cpu()
    )

    print(
        f"   Weight mean: "
        f"{decision_weight.mean().item():+.6e}"
    )

    print(
        f"   Weight abs mean: "
        f"{decision_weight.abs().mean().item():.6e}"
    )

    print(
        f"   Weight std: "
        f"{decision_weight.std().item():.6e}"
    )

    print(
        f"   Bias: "
        f"{decision_bias.item():+.6e}"
    )

    # =========================================================================
    # 7. Generation
    # =========================================================================

    print("\n5. Generating...")

    print(
        f"   Question: {QUESTION}"
    )

    print(
        "   Max new tokens: 512"
    )

    print(
        "   Sampling: disabled (greedy decoding)"
    )

    print()

    with torch.inference_mode():

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,

            # -----------------------------------------------------------------
            # ART-specific input
            # -----------------------------------------------------------------

            question=QUESTION,

            # -----------------------------------------------------------------
            # Generation settings
            # -----------------------------------------------------------------

            max_new_tokens=1024,

            do_sample=False,

            use_cache=True,

            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # =========================================================================
    # 8. Decode generated answer only
    # =========================================================================

    prompt_length = input_ids.shape[1]

    generated_ids = outputs[
        0,
        prompt_length:,
    ]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    # =========================================================================
    # 9. Print result
    # =========================================================================

    print("=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)

    print(generated_text)

    print("=" * 80)

    print(
        f"Generated tokens: {generated_ids.shape[0]}"
    )

    print(
        f"Total tokens: {outputs.shape[1]}"
    )

    # =========================================================================
    # 10. ART state
    # =========================================================================

    print("\nART STATE")
    print("-" * 80)

    if hasattr(model, "retrieval_count"):

        print(
            f"Retrieval count: {model.retrieval_count}"
        )

    if hasattr(model, "max_retrievals"):

        print(
            f"Max retrievals: {model.max_retrievals}"
        )

    if hasattr(model, "isTraining"):

        print(
            f"isTraining: {model.isTraining}"
        )

    print("QUERY:")
    for i, q in enumerate(model.query):
        print(f"[{i}] {q}")

    print("\nDOC:")
    for i, d in enumerate(model.doc):
        print(f"[{i}] {d}")

    print("-" * 80)


if __name__ == "__main__":
    main()