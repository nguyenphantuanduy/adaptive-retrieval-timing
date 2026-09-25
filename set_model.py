from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM


# =============================================================================
# CONFIG
# =============================================================================

MODEL_REPO = "adaptive-retrieval-timing/ART-Demo-Qwen2.5-3B"

DTYPE = torch.bfloat16


# =============================================================================
# INITIALIZE RETRIEVAL DECISION HEAD
# =============================================================================

def initialize_decision_head(model):
    """
    Randomly initialize the retrieval decision head.

    The retrieval decision head is:

        decision_logit = weight @ hidden_state + bias
        decision_score = sigmoid(decision_logit)

    The projection layer is initialized using PyTorch's default
    nn.Linear initialization:

        weight -> Kaiming uniform
        bias   -> uniform based on fan-in

    Therefore, the initial decision score is not forced to 0.5.
    """

    print()
    print("=" * 80)
    print("INITIALIZING RETRIEVAL DECISION HEAD")
    print("=" * 80)

    weight = (
        model.retrieval_decision_head
        .projection
        .weight
    )

    bias = (
        model.retrieval_decision_head
        .projection
        .bias
    )

    print()
    print("Before initialization:")

    print(
        f"Weight mean     : "
        f"{weight.detach().float().mean().item():+.6e}"
    )

    print(
        f"Weight std      : "
        f"{weight.detach().float().std().item():.6e}"
    )

    print(
        f"Weight abs mean : "
        f"{weight.detach().float().abs().mean().item():.6e}"
    )

    print(
        f"Bias mean       : "
        f"{bias.detach().float().mean().item():+.6e}"
    )

    print(
        f"Bias std        : "
        f"{bias.detach().float().std().item():.6e}"
    )

    # =========================================================================
    # Random initialization
    # =========================================================================

    with torch.no_grad():

        # Use the same initialization as nn.Linear.reset_parameters().
        torch.nn.init.kaiming_uniform_(
            weight,
            a=5 ** 0.5,
        )

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            weight
        )

        bound = 1 / (fan_in ** 0.5)

        torch.nn.init.uniform_(
            bias,
            -bound,
            bound,
        )

    # =========================================================================
    # Statistics after initialization
    # =========================================================================

    print()
    print("After random initialization:")

    print(
        f"Weight mean     : "
        f"{weight.detach().float().mean().item():+.6e}"
    )

    print(
        f"Weight std      : "
        f"{weight.detach().float().std().item():.6e}"
    )

    print(
        f"Weight abs mean : "
        f"{weight.detach().float().abs().mean().item():.6e}"
    )

    print(
        f"Weight min      : "
        f"{weight.detach().float().min().item():+.6e}"
    )

    print(
        f"Weight max      : "
        f"{weight.detach().float().max().item():+.6e}"
    )

    print(
        f"Bias mean       : "
        f"{bias.detach().float().mean().item():+.6e}"
    )

    print(
        f"Bias std        : "
        f"{bias.detach().float().std().item():.6e}"
    )

    print(
        f"Bias min        : "
        f"{bias.detach().float().min().item():+.6e}"
    )

    print(
        f"Bias max        : "
        f"{bias.detach().float().max().item():+.6e}"
    )

    print("=" * 80)


# =============================================================================
# VERIFY DECISION SCORE
# =============================================================================

def verify_decision_score(model):
    """
    Verify the randomly initialized retrieval decision head.

    Instead of expecting exactly 0.5, this function evaluates the
    decision head on several hidden states and reports the resulting
    decision scores.

    Note:
        A zero hidden state only measures sigmoid(bias), so it does not
        fully test whether the randomly initialized weight is active.
    """

    print()
    print("=" * 80)
    print("VERIFYING DECISION SCORE")
    print("=" * 80)

    weight = (
        model.retrieval_decision_head
        .projection
        .weight
    )

    hidden_size = model.config.hidden_size

    # =========================================================================
    # Create test hidden states
    # =========================================================================

    test_hidden = torch.randn(
        8,
        1,
        hidden_size,
        device=weight.device,
        dtype=weight.dtype,
    )

    with torch.no_grad():

        decision_scores = (
            model.retrieval_decision_head(
                test_hidden
            )
        )

    scores = (
        decision_scores
        .detach()
        .float()
        .cpu()
    )

    # =========================================================================
    # Statistics
    # =========================================================================

    print()

    print(
        f"Decision score mean : "
        f"{scores.mean().item():.8f}"
    )

    print(
        f"Decision score std  : "
        f"{scores.std().item():.8f}"
    )

    print(
        f"Decision score min  : "
        f"{scores.min().item():.8f}"
    )

    print(
        f"Decision score max  : "
        f"{scores.max().item():.8f}"
    )

    print()

    print("Individual test scores:")

    for index, score in enumerate(scores.tolist()):

        print(
            f"  [{index}] "
            f"{score:.8f}"
        )

    # =========================================================================
    # Zero hidden state
    # =========================================================================

    zero_hidden = torch.zeros(
        1,
        1,
        hidden_size,
        device=weight.device,
        dtype=weight.dtype,
    )

    with torch.no_grad():

        zero_decision_score = (
            model.retrieval_decision_head(
                zero_hidden
            )
        )

    zero_score = (
        zero_decision_score
        .detach()
        .float()
        .cpu()
        .item()
    )

    print()

    print(
        f"Decision score for zero hidden state : "
        f"{zero_score:.8f}"
    )

    print(
        "Note: this value is sigmoid(bias)."
    )

    # =========================================================================
    # Sanity check
    # =========================================================================

    weight_abs_mean = (
        weight.detach()
        .float()
        .abs()
        .mean()
        .item()
    )

    if weight_abs_mean > 0:

        print()
        print(
            "✓ Decision head was randomly initialized."
        )

    else:

        print()
        print(
            "✗ WARNING: Decision head weight is still zero."
        )

    print("=" * 80)


# =============================================================================
# PUSH MODEL
# =============================================================================

def push_model(model):
    """
    Upload the modified model weights back to the same Hugging Face repository.
    """

    print()
    print("=" * 80)
    print("UPLOADING MODEL TO HUGGING FACE")
    print("=" * 80)

    print()
    print(
        f"Repository : {MODEL_REPO}"
    )

    print()
    print(
        "Pushing modified model..."
    )

    model.push_to_hub(
        MODEL_REPO,
        commit_message=(
            "Randomly initialize retrieval decision head"
        ),
    )

    print()
    print("✓ Model uploaded successfully.")

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("ART DECISION HEAD RANDOM INITIALIZATION")
    print("=" * 80)

    # =========================================================================
    # 1. Device
    # =========================================================================

    print()
    print("1. Checking device...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =========================================================================
    # 2. Load model
    # =========================================================================

    print()
    print("2. Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        dtype=DTYPE,
        attn_implementation="eager",
    )

    print()
    print(
        f"✓ Model loaded: {MODEL_REPO}"
    )

    # =========================================================================
    # 3. Move model to device
    # =========================================================================

    print()
    print("3. Moving model to device...")

    model = model.to(device)

    print(
        f"✓ Model device: {device}"
    )

    # =========================================================================
    # 4. Initialize decision head
    # =========================================================================

    print()
    print("4. Randomly initializing retrieval decision head...")

    initialize_decision_head(model)

    # =========================================================================
    # 5. Verify
    # =========================================================================

    print()
    print("5. Verifying decision score...")

    verify_decision_score(model)

    # =========================================================================
    # 6. Upload
    # =========================================================================

    print()
    print("6. Uploading modified weights...")

    push_model(model)

    # =========================================================================
    # 7. Cleanup
    # =========================================================================

    print()
    print("7. Cleaning up...")

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        "Retrieval decision head:"
    )

    print(
        "  weight = random nn.Linear initialization"
    )

    print(
        "  bias   = random nn.Linear initialization"
    )

    print(
        "  score  = sigmoid(weight @ hidden + bias)"
    )

    print()
    print(
        f"Updated repository: {MODEL_REPO}"
    )

    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()