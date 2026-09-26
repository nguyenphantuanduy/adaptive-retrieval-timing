"""
Transfer the original Qwen2.5-3B-Instruct weights into the ART demo model
and publish the initialized model.

Run this file from the repository root:

    .\\.venv\\Scripts\\python.exe ^
        shared/ART_Models/ART_demo_version/initialize_art_demo_qwen2_5_3b.py

The script requires every original Qwen state-dict key to exist in the
custom ART demo model with exactly the same tensor shape.

The custom ART demo model may contain additional parameters, such as:

    retrieval_decision_head.*

Those additional parameters are intentionally excluded from the compatibility
check and weight transfer from Qwen.

The ART retrieval decision head is randomly initialized using the same
initialization as PyTorch's nn.Linear:

    weight -> Kaiming uniform
    bias   -> uniform based on fan-in

Therefore, the initial retrieval decision score is NOT forced to 0.5.
"""


from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)


# =============================================================================
# Configuration
# =============================================================================

# The directory containing this script, config.json, and all custom modules.
CUSTOM_MODEL_DIR = Path(__file__).resolve().parent

QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct"

TARGET_REPO = "adaptive-retrieval-timing/ART-Demo-Qwen2.5-3B"

DTYPE = torch.bfloat16


# =============================================================================
# Diagnostics
# =============================================================================

def print_names(
    title: str,
    names: list[str],
) -> None:
    """
    Print a concise diagnostic list.
    """

    print(
        f"{title}: {len(names)}"
    )

    for name in names[:30]:

        print(
            f"  - {name}"
        )

    if len(names) > 30:

        print(
            f"  ... and {len(names) - 30} more"
        )


def verify_qwen_compatibility(
    demo_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
) -> None:
    """
    Verify that every Qwen state-dict parameter exists in the ART demo model.

    This is intentionally a one-way compatibility check:

        Qwen -> ART demo

    The ART demo is allowed to contain additional parameters, such as:

        retrieval_decision_head.*

    Those additional parameters are not required to exist in Qwen.
    """

    demo_keys = set(demo_state)

    qwen_keys = set(qwen_state)

    # -------------------------------------------------------------------------
    # Qwen parameters missing from the ART demo.
    #
    # These are not allowed.
    # -------------------------------------------------------------------------

    missing_in_demo = sorted(
        qwen_keys - demo_keys
    )

    # -------------------------------------------------------------------------
    # Shared parameters with different shapes.
    # -------------------------------------------------------------------------

    shape_mismatches = sorted(
        (
            name,
            tuple(qwen_state[name].shape),
            tuple(demo_state[name].shape),
        )
        for name in qwen_keys & demo_keys
        if qwen_state[name].shape != demo_state[name].shape
    )

    # -------------------------------------------------------------------------
    # Parameters that exist only in the ART demo.
    #
    # These are expected.
    # -------------------------------------------------------------------------

    additional_in_demo = sorted(
        demo_keys - qwen_keys
    )

    print()
    print("State-dict compatibility report")
    print("-" * 80)

    print_names(
        "Qwen parameters missing in ART demo",
        missing_in_demo,
    )

    print(
        f"Shape mismatches: "
        f"{len(shape_mismatches)}"
    )

    for (
        name,
        qwen_shape,
        demo_shape,
    ) in shape_mismatches[:30]:

        print(
            f"  - {name}: "
            f"qwen={qwen_shape}, "
            f"demo={demo_shape}"
        )

    if len(shape_mismatches) > 30:

        print(
            f"  ... and "
            f"{len(shape_mismatches) - 30} more"
        )

    print_names(
        "Additional parameters in ART demo",
        additional_in_demo,
    )

    # -------------------------------------------------------------------------
    # Abort if any Qwen parameter cannot be transferred exactly.
    # -------------------------------------------------------------------------

    if missing_in_demo or shape_mismatches:

        raise RuntimeError(
            "Qwen and ART demo are not compatible for weight transfer; "
            "nothing was uploaded."
        )

    print(
        f"\nAll {len(qwen_keys)} Qwen state-dict tensors "
        "have matching names and shapes in the ART demo."
    )

    print(
        f"ART demo contains {len(additional_in_demo)} additional "
        "parameter tensors that will keep their initialized values."
    )


def verify_transferred_values(
    demo_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
) -> None:
    """
    Verify that every Qwen tensor was transferred exactly.

    Additional ART demo parameters are intentionally ignored.
    """

    unequal = [
        name
        for name in qwen_state
        if not torch.equal(
            demo_state[name].cpu(),
            qwen_state[name].cpu(),
        )
    ]

    if unequal:

        print_names(
            "Qwen tensors with unequal values after transfer",
            unequal,
        )

        raise RuntimeError(
            "Weight transfer verification failed; "
            "nothing was uploaded."
        )

    print(
        f"All {len(qwen_state)} Qwen tensors match "
        "the original checkpoint exactly."
    )


# =============================================================================
# ART INITIALIZATION
# =============================================================================

def initialize_retrieval_decision_head(
    demo,
) -> None:
    """
    Randomly initialize the ART retrieval decision head.

    The retrieval decision head computes:

        decision_logit = weight @ hidden_state + bias

        decision_score = sigmoid(decision_logit)

    The projection layer is initialized using the same initialization
    as PyTorch's nn.Linear.reset_parameters():

        weight -> Kaiming uniform
        bias   -> uniform based on fan-in

    Therefore, the initial decision score is not forced to 0.5.
    """

    print()
    print("=" * 80)
    print("INITIALIZING ART RETRIEVAL DECISION HEAD")
    print("=" * 80)

    projection = (
        demo.retrieval_decision_head.projection
    )

    weight = projection.weight

    bias = projection.bias

    # =========================================================================
    # Before initialization
    # =========================================================================

    print()
    print("Before random initialization:")

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
        f"Bias            : "
        f"{bias.detach().float().item():+.6e}"
    )

    # =========================================================================
    # Random initialization
    #
    # Match torch.nn.Linear.reset_parameters().
    # =========================================================================

    with torch.no_grad():

        torch.nn.init.kaiming_uniform_(
            weight,
            a=5 ** 0.5,
        )

        fan_in, _ = (
            torch.nn.init._calculate_fan_in_and_fan_out(
                weight
            )
        )

        bound = 1 / (fan_in ** 0.5)

        torch.nn.init.uniform_(
            bias,
            -bound,
            bound,
        )

    # =========================================================================
    # After initialization
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
        f"Bias            : "
        f"{bias.detach().float().item():+.6e}"
    )

    print(
        f"Initial zero-hidden score: "
        f"{torch.sigmoid(bias.detach().float()).item():.8f}"
    )

    print()
    print(
        "Initialization method:"
    )

    print(
        "  weight = Kaiming uniform"
    )

    print(
        "  bias   = uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))"
    )

    print(
        "  score  = sigmoid(weight @ hidden + bias)"
    )

    print("=" * 80)


# =============================================================================
# VERIFY RANDOM DECISION HEAD
# =============================================================================

def verify_retrieval_decision_head(
    demo,
) -> None:
    """
    Verify that the retrieval decision head is no longer zero initialized.

    The verification uses random hidden states because evaluating the head
    only on a zero hidden state would test the bias and not the weight.
    """

    print()
    print("=" * 80)
    print("VERIFYING RANDOM RETRIEVAL DECISION HEAD")
    print("=" * 80)

    projection = (
        demo.retrieval_decision_head.projection
    )

    weight = projection.weight

    bias = projection.bias

    hidden_size = demo.config.hidden_size

    # =========================================================================
    # Weight verification
    # =========================================================================

    weight_abs_mean = (
        weight.detach()
        .float()
        .abs()
        .mean()
        .item()
    )

    weight_std = (
        weight.detach()
        .float()
        .std()
        .item()
    )

    print()
    print(
        f"Weight abs mean : "
        f"{weight_abs_mean:.6e}"
    )

    print(
        f"Weight std      : "
        f"{weight_std:.6e}"
    )

    if weight_abs_mean == 0.0:

        raise RuntimeError(
            "Retrieval decision head weight is still zero."
        )

    print(
        "✓ Retrieval decision head weight is non-zero."
    )

    # =========================================================================
    # Bias verification
    # =========================================================================

    bias_value = (
        bias.detach()
        .float()
        .item()
    )

    print(
        f"Bias            : "
        f"{bias_value:+.6e}"
    )

    # =========================================================================
    # Random hidden states
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
            demo.retrieval_decision_head(
                test_hidden
            )
        )

    scores = (
        decision_scores
        .detach()
        .float()
        .cpu()
    )

    print()
    print("Decision scores on random hidden states:")

    print(
        f"Mean : "
        f"{scores.mean().item():.8f}"
    )

    print(
        f"Std  : "
        f"{scores.std().item():.8f}"
    )

    print(
        f"Min  : "
        f"{scores.min().item():.8f}"
    )

    print(
        f"Max  : "
        f"{scores.max().item():.8f}"
    )

    print()

    for index, score in enumerate(
        scores.tolist()
    ):

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
            demo.retrieval_decision_head(
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
        f"Zero-hidden decision score: "
        f"{zero_score:.8f}"
    )

    print(
        "This value is sigmoid(bias)."
    )

    print()
    print(
        "✓ Retrieval decision head is randomly initialized."
    )

    print("=" * 80)


# =============================================================================
# Custom source export
# =============================================================================

def copy_custom_code(
    export_dir: Path,
) -> None:
    """
    Copy all local custom-model Python files and the hand-authored config.
    """

    for source_file in CUSTOM_MODEL_DIR.glob("*.py"):

        shutil.copy2(
            source_file,
            export_dir / source_file.name,
        )

    # Preserve the hand-authored config.json / auto_map.
    shutil.copy2(
        CUSTOM_MODEL_DIR / "config.json",
        export_dir / "config.json",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    print("=" * 80)

    print(
        "INITIALIZE ART DEMO: "
        "QWEN2.5-3B-INSTRUCT"
    )

    print("=" * 80)

    print(
        f"\nCustom model directory:\n"
        f"  {CUSTOM_MODEL_DIR}"
    )

    print(
        f"\nTarget Hugging Face repository:\n"
        f"  {TARGET_REPO}"
    )

    # =========================================================================
    # 1. Create empty ART demo model
    # =========================================================================

    print(
        "\n1. Loading custom ART demo config "
        "and creating an empty model..."
    )

    config = AutoConfig.from_pretrained(
        CUSTOM_MODEL_DIR,
        trust_remote_code=True,
    )

    demo = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        attn_implementation="eager",
        dtype=DTYPE,
    )

    print(
        f"   Demo class: "
        f"{type(demo).__name__}"
    )

    print(
        f"   Demo dtype: "
        f"{demo.model.embed_tokens.weight.dtype}"
    )

    # =========================================================================
    # 1.1 Randomly initialize ART retrieval decision head
    # =========================================================================

    initialize_retrieval_decision_head(
        demo=demo,
    )

    # =========================================================================
    # 2. Load original Qwen checkpoint
    # =========================================================================

    print(
        "\n2. Loading original "
        "Qwen2.5-3B-Instruct checkpoint on CPU..."
    )

    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_REPO,
        dtype=DTYPE,
        device_map="cpu",
    )

    print(
        f"   Original class: "
        f"{type(qwen).__name__}"
    )

    print(
        f"   Original dtype: "
        f"{qwen.model.embed_tokens.weight.dtype}"
    )

    # =========================================================================
    # 3. Collect state dictionaries
    # =========================================================================

    print(
        "\n3. Checking state-dict compatibility..."
    )

    demo_state = demo.state_dict()

    qwen_state = qwen.state_dict()

    verify_qwen_compatibility(
        demo_state=demo_state,
        qwen_state=qwen_state,
    )

    # =========================================================================
    # 4. Transfer Qwen weights
    # =========================================================================

    print(
        "\n4. Transferring Qwen weights into ART demo..."
    )

    # -------------------------------------------------------------------------
    # The ART demo contains:
    #
    #     all Qwen parameters
    #
    # plus:
    #
    #     retrieval_decision_head.*
    #
    # Therefore, merge the Qwen state dict into the complete ART state dict.
    #
    # This preserves the randomly initialized ART-specific parameters.
    # -------------------------------------------------------------------------

    merged_state = demo.state_dict()

    for name, tensor in qwen_state.items():

        merged_state[name] = tensor

    demo.load_state_dict(
        merged_state,
        strict=True,
    )

    print(
        f"   Transferred "
        f"{len(qwen_state)} Qwen state-dict tensors."
    )

    print(
        "   ART-specific parameters were preserved "
        "with their random initialized values."
    )

    # =========================================================================
    # 5. Verify Qwen transfer result
    # =========================================================================

    print(
        "\n5. Verifying transferred Qwen weights..."
    )

    verify_transferred_values(
        demo_state=demo.state_dict(),
        qwen_state=qwen_state,
    )

    # =========================================================================
    # 5.1 Verify ART decision head after Qwen transfer
    # =========================================================================

    print(
        "\n5.1 Verifying ART retrieval decision head..."
    )

    verify_retrieval_decision_head(
        demo=demo,
    )

    # =========================================================================
    # 6. Report ART-specific parameters
    # =========================================================================

    additional_keys = sorted(
        set(demo.state_dict())
        - set(qwen_state)
    )

    print(
        "\n6. ART demo-specific parameters:"
    )

    for name in additional_keys:

        print(
            f"  - {name}"
        )

    # =========================================================================
    # 7. Free original Qwen model
    # =========================================================================

    print(
        "\n7. Freeing original Qwen model..."
    )

    del qwen

    del qwen_state

    del demo_state

    del merged_state

    gc.collect()

    # =========================================================================
    # 8. Save model and custom source
    # =========================================================================

    print(
        "\n8. Saving model weights and custom source "
        "into a temporary publish directory..."
    )

    with tempfile.TemporaryDirectory(
        prefix="art-demo-qwen2.5-3b-"
    ) as temp_dir:

        export_dir = Path(temp_dir)

        demo.save_pretrained(
            export_dir,
            safe_serialization=True,
        )

        copy_custom_code(
            export_dir
        )

        saved_files = sorted(
            path.name
            for path in export_dir.iterdir()
        )

        print(
            "   Files to publish:"
        )

        for filename in saved_files:

            print(
                f"   - {filename}"
            )

        # =====================================================================
        # 9. Upload to Hugging Face
        # =====================================================================

        print(
            f"\n9. Uploading and overwriting "
            f"{TARGET_REPO}..."
        )

        api = HfApi()

        api.create_repo(
            TARGET_REPO,
            repo_type="model",
            exist_ok=True,
        )

        commit = api.upload_folder(
            repo_id=TARGET_REPO,
            repo_type="model",
            folder_path=export_dir,
            commit_message=(
                "Initialize ART demo from "
                "Qwen2.5-3B-Instruct with random "
                "retrieval decision head"
            ),
        )

    # =========================================================================
    # 10. Done
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "PUBLISH COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Base model: "
        f"{QWEN_REPO}"
    )

    print(
        f"Published model: "
        f"https://huggingface.co/{TARGET_REPO}"
    )

    print(
        f"Commit: "
        f"{commit}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()