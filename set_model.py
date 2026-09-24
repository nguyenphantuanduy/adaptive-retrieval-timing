from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM


# =============================================================================
# CONFIG
# =============================================================================

MODEL_REPO = "adaptive-retrieval-timing/ART-Demo-Qwen2.5-3B"

DTYPE = torch.bfloat16

# -------------------------------------------------------------------------
# Retrieval decision initialization
#
# weight = 0
# bias   = 0
#
# Therefore:
#
# decision_logit = 0
# decision_score = sigmoid(0) = 0.5
# -------------------------------------------------------------------------

DECISION_WEIGHT = 0.0
DECISION_BIAS = 0.0


# =============================================================================
# INITIALIZE RETRIEVAL DECISION HEAD
# =============================================================================

def initialize_decision_head(model):
    """
    Set the retrieval decision head to produce an initial score of 0.5.

    Since:

        decision = sigmoid(weight @ hidden_state + bias)

    and we set:

        weight = 0
        bias   = 0

    the initial decision score becomes:

        sigmoid(0) = 0.5
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
        f"Weight abs mean : "
        f"{weight.detach().float().abs().mean().item():.6e}"
    )

    print(
        f"Bias            : "
        f"{bias.detach().float().cpu().numpy()}"
    )

    # -------------------------------------------------------------------------
    # Set weight and bias
    # -------------------------------------------------------------------------

    with torch.no_grad():

        weight.zero_()

        bias.fill_(DECISION_BIAS)

    # -------------------------------------------------------------------------
    # Expected score
    # -------------------------------------------------------------------------

    expected_score = torch.sigmoid(
        torch.tensor(
            DECISION_BIAS,
            dtype=torch.float32,
        )
    ).item()

    print()
    print("After initialization:")

    print(
        f"Weight mean     : "
        f"{weight.detach().float().mean().item():+.6e}"
    )

    print(
        f"Weight abs mean : "
        f"{weight.detach().float().abs().mean().item():.6e}"
    )

    print(
        f"Bias            : "
        f"{bias.detach().float().cpu().numpy()}"
    )

    print()
    print(
        f"Expected decision score : "
        f"{expected_score:.6f}"
    )

    print("=" * 80)


# =============================================================================
# VERIFY DECISION SCORE
# =============================================================================

def verify_decision_score(model):
    """
    Verify the retrieval decision head using a zero hidden state.

    With weight=0 and bias=0:

        sigmoid(0) = 0.5
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

    # -------------------------------------------------------------------------
    # Zero hidden state
    # -------------------------------------------------------------------------

    zero_hidden = torch.zeros(
        1,
        1,
        hidden_size,
        device=weight.device,
        dtype=weight.dtype,
    )

    with torch.no_grad():

        decision_score = (
            model.retrieval_decision_head(
                zero_hidden
            )
        )

    score = (
        decision_score
        .detach()
        .float()
        .cpu()
        .item()
    )

    print()
    print(
        f"Decision score : {score:.8f}"
    )

    print(
        f"Expected       : 0.50000000"
    )

    print(
        f"Difference     : "
        f"{abs(score - 0.5):.8e}"
    )

    # -------------------------------------------------------------------------
    # Sanity check
    # -------------------------------------------------------------------------

    if abs(score - 0.5) < 1e-5:
        print()
        print("✓ Decision head initialized correctly.")
    else:
        print()
        print("✗ WARNING: Decision score is not approximately 0.5.")

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
            "Initialize retrieval decision head "
            "with zero weight and zero bias"
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
    print("ART DECISION HEAD INITIALIZATION")
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
    print("4. Initializing retrieval decision head...")

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
        "  weight = 0"
    )

    print(
        "  bias   = 0"
    )

    print(
        "  score  = sigmoid(0) = 0.5"
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