from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Configuration
# =============================================================================

MODEL_REPO = "adaptive-retrieval-timing/ART-HOPE"
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"

DTYPE = torch.bfloat16

QUESTION = (
    "Who was the person that started the Bethel branch of the religion "
    "founded by the black community in the former US capital?"
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
    print("ART-HOPE GENERATION TEST")
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
        TOKENIZER_REPO,
        trust_remote_code=True,
    )

    print(f"   Tokenizer: {TOKENIZER_REPO}")

    if tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer does not have an EOS token."
        )

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
    # 5. Load ART-HOPE model
    # =========================================================================

    print("\n3. Loading ART-HOPE model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        dtype=DTYPE,
        attn_implementation="eager",
    )

    model = model.to(device)
    model.eval()

    print(
        f"   Model: {MODEL_REPO}"
    )

    print(
        f"   Model class: {type(model).__name__}"
    )

    print(
        "   Model dtype: "
        f"{model.model.embed_tokens.weight.dtype}"
    )

    # =========================================================================
    # 6. Verify ART decision head
    # =========================================================================

    print("\n4. Verifying ART decision head...")

    if not hasattr(
        model,
        "retrieval_decision_head",
    ):
        raise RuntimeError(
            "Loaded model does not contain "
            "retrieval_decision_head."
        )

    decision_head = model.retrieval_decision_head

    print(
        f"   Decision head: "
        f"{type(decision_head).__name__}"
    )

    linear_layers = [
        module
        for module in decision_head.modules()
        if isinstance(module, torch.nn.Linear)
    ]

    if not linear_layers:
        raise RuntimeError(
            "No Linear layer found inside "
            "retrieval_decision_head."
        )

    for index, layer in enumerate(linear_layers):

        print(
            f"   Linear {index + 1}: "
            f"{layer.in_features} → "
            f"{layer.out_features}"
        )

        print(
            f"      Weight shape: "
            f"{tuple(layer.weight.shape)}"
        )

        if layer.bias is not None:

            print(
                f"      Bias shape: "
                f"{tuple(layer.bias.shape)}"
            )

    # =========================================================================
    # 7. Verify decision branch
    # =========================================================================

    print("\n5. Verifying ART decision branch...")

    if not hasattr(
        model.model,
        "decision_layers",
    ):
        raise RuntimeError(
            "Loaded model does not contain "
            "model.decision_layers."
        )

    if not hasattr(
        model.model,
        "decision_norm",
    ):
        raise RuntimeError(
            "Loaded model does not contain "
            "model.decision_norm."
        )

    print(
        f"   Decision layers: "
        f"{len(model.model.decision_layers)}"
    )

    print(
        "   Decision norm: present"
    )

    print(
        "   Decision head: present"
    )

    # =========================================================================
    # 8. Check loaded decision head output
    # =========================================================================
    #
    # IMPORTANT:
    #
    # The decision head is loaded directly from ART-HOPE.
    #
    # No manual initialization is performed here.
    #
    # =========================================================================

    print("\n6. Checking loaded decision head output...")

    hidden_size = model.config.hidden_size

    dummy_hidden_state = torch.zeros(
        1,
        1,
        hidden_size,
        device=device,
        dtype=DTYPE,
    )

    with torch.inference_mode():

        decision_score = (
            model.retrieval_decision_head(
                dummy_hidden_state
            )
        )

    decision_score = (
        decision_score
        .detach()
        .float()
        .cpu()
    )

    print(
        "   Zero-hidden decision score: "
        f"{decision_score.tolist()}"
    )

    # =========================================================================
    # 9. Generation
    # =========================================================================

    print("\n7. Generating...")

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

            max_new_tokens=512,

            do_sample=False,

            use_cache=True,

            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # =========================================================================
    # 10. Decode generated answer only
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
    # 11. Print result
    # =========================================================================

    print("=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)

    print(generated_text)

    print("=" * 80)

    print(
        f"Generated tokens: "
        f"{generated_ids.shape[0]}"
    )

    print(
        f"Total tokens: "
        f"{outputs.shape[1]}"
    )

    # =========================================================================
    # 12. ART state
    # =========================================================================

    print("\nART STATE")
    print("-" * 80)

    if hasattr(
        model,
        "retrieval_count",
    ):

        print(
            f"Retrieval count: "
            f"{model.retrieval_count}"
        )

    if hasattr(
        model,
        "max_retrievals",
    ):

        print(
            f"Max retrievals: "
            f"{model.max_retrievals}"
        )

    if hasattr(
        model,
        "isTraining",
    ):

        print(
            f"isTraining: "
            f"{model.isTraining}"
        )

    print("-" * 80)


if __name__ == "__main__":
    main()