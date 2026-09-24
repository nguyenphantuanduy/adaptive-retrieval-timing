from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Configuration
# =============================================================================

MODEL_REPO = "adaptive-retrieval-timing/ART-Demo-Qwen2.5-3B"
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
    print("ART DEMO GENERATION TEST")
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

    # Use Qwen's official chat template.
    #
    # The question appears in the user message.
    # The same question is passed separately to ART later as
    # `question=QUESTION` for retrieval.
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
    # 5. Load ART model
    # =========================================================================

    print("\n3. Loading ART demo model...")

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
    # 6. Initialize ART decision head
    # =========================================================================
    #
    # The decision head is newly added to the original Qwen model, so its
    # parameters are not pretrained.
    #
    # For this DEMO ONLY:
    #
    #     weight = 0
    #     bias   = 0
    #
    # Therefore:
    #
    #     logit    = 0
    #     sigmoid  = 0.5
    #
    # With THRESHOLD = 0.5 and condition score >= threshold,
    # retrieval will be triggered.
    #
    # This is NOT the final ART initialization/training strategy.
    # It is only used to verify that the retrieval pipeline works.
    #
    # =========================================================================

    print("\n4. Initializing ART decision head for retrieval test...")

    with torch.no_grad():

        model.retrieval_decision_head.projection.weight.zero_()
        model.retrieval_decision_head.projection.bias.zero_()

    print(
        "   Decision head weight: ZERO"
    )

    print(
        "   Decision head bias:   ZERO"
    )

    # Verify the initial decision score.
    #
    # Use a dummy hidden state with the same hidden dimension as Qwen.
    hidden_size = model.config.hidden_size

    dummy_hidden_state = torch.zeros(
        1,
        1,
        hidden_size,
        device=device,
        dtype=DTYPE,
    )

    with torch.inference_mode():

        initial_decision_score = (
            model.retrieval_decision_head(
                dummy_hidden_state
            )
        )

    print(
        f"   Initial decision score: "
        f"{initial_decision_score.detach().float().cpu().tolist()}"
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
            #
            # The question is already included in the user prompt.
            #
            # This separate argument is NOT added to the prompt.
            # It is passed to ART so the retrieval module knows the
            # original question.
            #
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

    print("-" * 80)


if __name__ == "__main__":
    main()