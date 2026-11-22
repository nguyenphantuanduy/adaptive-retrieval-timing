"""
Transfer the original Qwen2.5-3B-Instruct weights into the ART model.

ART architecture:

    Qwen2.5-3B-Instruct
                |
                v
        +----------------+
        | Shared layers  |
        |    0 ... 21    |
        +----------------+
                |
                +-------------------------+
                |                         |
                v                         v
        +----------------+      +----------------------+
        | LM branch      |      | Decision branch      |
        | layers 22...35 |      | decision_layers 0...13|
        +----------------+      +----------------------+
                |                         |
                v                         v
            LM norm                Decision norm
                |                         |
                v                         v
             LM head             RetrievalDecisionHead
                                      2048 -> 1 -> sigmoid

Weight initialization:

    ART model.embed_tokens
        <- Qwen2.5-3B-Instruct model.embed_tokens

    ART model.layers.0...35
        <- Qwen2.5-3B-Instruct model.layers.0...35

    ART model.decision_layers.0...13
        <- Qwen2.5-3B-Instruct model.layers.22...35

    ART model.norm
        <- Qwen2.5-3B-Instruct model.norm

    ART model.decision_norm
        <- Qwen2.5-3B-Instruct model.norm

    ART lm_head
        <- Qwen2.5-3B-Instruct lm_head

    ART retrieval_decision_head
        <- random PyTorch nn.Linear initialization

The decision branch is a physical copy of the last 14 Qwen layers.
It does NOT share parameters with the LM branch.

The decision normalization layer is also a physical copy of the
original Qwen final RMSNorm. It does NOT share its Parameter object
with the LM normalization layer.

Run this file from the repository root:

    .\\.venv\\Scripts\\python.exe ^
        shared/ART_Models/ART_HOPE/initialize_art_hope_qwen2_5_3b.py
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

CUSTOM_MODEL_DIR = Path(__file__).resolve().parent

QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct"

TARGET_REPO = "adaptive-retrieval-timing/ART-HOPE"

DTYPE = torch.bfloat16


# =============================================================================
# ART architecture configuration
# =============================================================================

QWEN_NUM_LAYERS = 36

DECISION_BRANCH_START_LAYER = 22

NUM_DECISION_LAYERS = 14


assert (
    QWEN_NUM_LAYERS
    - DECISION_BRANCH_START_LAYER
    == NUM_DECISION_LAYERS
), (
    "ART decision branch configuration is inconsistent: "
    f"{QWEN_NUM_LAYERS} - "
    f"{DECISION_BRANCH_START_LAYER} != "
    f"{NUM_DECISION_LAYERS}"
)


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


# =============================================================================
# Build expected ART -> Qwen mapping
# =============================================================================

def build_qwen_transfer_mapping(
    demo_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
) -> dict[str, str]:
    """
    Build the explicit Qwen -> ART parameter mapping.

    Mapping:

        model.embed_tokens.*
            <- model.embed_tokens.*

        model.layers.0...35.*
            <- model.layers.0...35.*

        model.decision_layers.0...13.*
            <- model.layers.22...35.*

        model.norm.*
            <- model.norm.*

        model.decision_norm.*
            <- model.norm.*

        lm_head.*
            <- lm_head.*

    ART-specific retrieval decision head parameters are intentionally
    excluded from this mapping.
    """

    mapping: dict[str, str] = {}

    # =========================================================================
    # 1. Embedding
    # =========================================================================

    qwen_embed_prefix = "model.embed_tokens."
    demo_embed_prefix = "model.embed_tokens."

    for demo_key in demo_state:

        if demo_key.startswith(demo_embed_prefix):

            suffix = demo_key[len(demo_embed_prefix):]

            qwen_key = (
                qwen_embed_prefix
                + suffix
            )

            mapping[demo_key] = qwen_key

    # =========================================================================
    # 2. Original Qwen transformer layers
    #
    # ART model.layers.0...35 remain the original Qwen backbone.
    # =========================================================================

    for layer_idx in range(QWEN_NUM_LAYERS):

        demo_prefix = (
            f"model.layers.{layer_idx}."
        )

        qwen_prefix = (
            f"model.layers.{layer_idx}."
        )

        for demo_key in demo_state:

            if demo_key.startswith(demo_prefix):

                suffix = demo_key[len(demo_prefix):]

                qwen_key = (
                    qwen_prefix
                    + suffix
                )

                mapping[demo_key] = qwen_key

    # =========================================================================
    # 3. Decision branch
    #
    # ART decision_layers.0
    #       <- Qwen model.layers.22
    #
    # ART decision_layers.1
    #       <- Qwen model.layers.23
    #
    # ...
    #
    # ART decision_layers.13
    #       <- Qwen model.layers.35
    # =========================================================================

    for decision_idx in range(NUM_DECISION_LAYERS):

        qwen_layer_idx = (
            DECISION_BRANCH_START_LAYER
            + decision_idx
        )

        demo_prefix = (
            f"model.decision_layers.{decision_idx}."
        )

        qwen_prefix = (
            f"model.layers.{qwen_layer_idx}."
        )

        for demo_key in demo_state:

            if demo_key.startswith(demo_prefix):

                suffix = demo_key[len(demo_prefix):]

                qwen_key = (
                    qwen_prefix
                    + suffix
                )

                mapping[demo_key] = qwen_key

    # =========================================================================
    # 4. LM normalization
    #
    # ART model.norm
    #       <- Qwen model.norm
    # =========================================================================

    qwen_norm_prefix = "model.norm."
    demo_norm_prefix = "model.norm."

    for demo_key in demo_state:

        if demo_key.startswith(demo_norm_prefix):

            suffix = demo_key[len(demo_norm_prefix):]

            qwen_key = (
                qwen_norm_prefix
                + suffix
            )

            mapping[demo_key] = qwen_key

    # =========================================================================
    # 5. Decision normalization
    #
    # ART model.decision_norm
    #       <- Qwen model.norm
    #
    # This creates an independent copy of the Qwen final RMSNorm.
    # =========================================================================

    demo_decision_norm_prefix = "model.decision_norm."
    qwen_norm_prefix = "model.norm."

    for demo_key in demo_state:

        if demo_key.startswith(
            demo_decision_norm_prefix
        ):

            suffix = demo_key[
                len(demo_decision_norm_prefix):
            ]

            qwen_key = (
                qwen_norm_prefix
                + suffix
            )

            mapping[demo_key] = qwen_key

    # =========================================================================
    # 6. LM head
    # =========================================================================

    for demo_key in demo_state:

        if demo_key.startswith("lm_head."):

            suffix = demo_key[len("lm_head."):]

            qwen_key = (
                "lm_head."
                + suffix
            )

            mapping[demo_key] = qwen_key

    return mapping


# =============================================================================
# Verify transfer mapping
# =============================================================================

def verify_transfer_mapping(
    demo_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
    mapping: dict[str, str],
) -> None:
    """
    Verify that every expected Qwen-backed ART parameter has a valid
    source tensor with exactly the same shape.

    ART-only parameters such as:

        retrieval_decision_head.network.*

    are intentionally excluded.
    """

    print()
    print("=" * 80)
    print("ART <-> QWEN WEIGHT TRANSFER REPORT")
    print("=" * 80)

    # =========================================================================
    # ART mapped keys that do not exist in the demo
    # =========================================================================

    missing_demo_keys = sorted(
        demo_key
        for demo_key in mapping
        if demo_key not in demo_state
    )

    # =========================================================================
    # Qwen source keys that do not exist
    # =========================================================================

    missing_qwen_keys = sorted(
        qwen_key
        for qwen_key in mapping.values()
        if qwen_key not in qwen_state
    )

    # =========================================================================
    # Shape mismatches
    # =========================================================================

    shape_mismatches = []

    for demo_key, qwen_key in mapping.items():

        if (
            demo_key not in demo_state
            or qwen_key not in qwen_state
        ):

            continue

        demo_shape = tuple(
            demo_state[demo_key].shape
        )

        qwen_shape = tuple(
            qwen_state[qwen_key].shape
        )

        if demo_shape != qwen_shape:

            shape_mismatches.append(
                (
                    demo_key,
                    qwen_key,
                    qwen_shape,
                    demo_shape,
                )
            )

    # =========================================================================
    # ART-specific parameters
    # =========================================================================

    mapped_demo_keys = set(mapping)

    additional_art_keys = sorted(
        set(demo_state)
        - mapped_demo_keys
    )

    # =========================================================================
    # Print report
    # =========================================================================

    print()

    print_names(
        "Mapped ART parameters missing from demo",
        missing_demo_keys,
    )

    print()

    print_names(
        "Qwen source parameters missing",
        missing_qwen_keys,
    )

    print()

    print(
        f"Shape mismatches: "
        f"{len(shape_mismatches)}"
    )

    for (
        demo_key,
        qwen_key,
        qwen_shape,
        demo_shape,
    ) in shape_mismatches[:30]:

        print(
            f"  - ART:  {demo_key}"
        )

        print(
            f"    Qwen: {qwen_key}"
        )

        print(
            f"    Qwen shape: {qwen_shape}"
        )

        print(
            f"    ART shape : {demo_shape}"
        )

    if len(shape_mismatches) > 30:

        print(
            f"  ... and "
            f"{len(shape_mismatches) - 30} more"
        )

    print()

    print_names(
        "ART-specific parameters "
        "(will keep initialized values)",
        additional_art_keys,
    )

    # =========================================================================
    # Abort on errors
    # =========================================================================

    if (
        missing_demo_keys
        or missing_qwen_keys
        or shape_mismatches
    ):

        raise RuntimeError(
            "ART and Qwen weights are not compatible "
            "for the configured transfer mapping; "
            "nothing was uploaded."
        )

    print()
    print(
        f"✓ Valid transfer mapping contains "
        f"{len(mapping)} ART parameter tensors."
    )

    print(
        "✓ All mapped Qwen source tensors exist."
    )

    print(
        "✓ All mapped tensors have matching shapes."
    )


# =============================================================================
# Verify transferred values
# =============================================================================

def verify_transferred_values(
    demo_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
    mapping: dict[str, str],
) -> None:
    """
    Verify that every mapped ART tensor is exactly equal to its Qwen source.

    ART-specific parameters are intentionally ignored.
    """

    unequal = []

    for demo_key, qwen_key in mapping.items():

        if not torch.equal(
            demo_state[demo_key].cpu(),
            qwen_state[qwen_key].cpu(),
        ):

            unequal.append(
                (
                    demo_key,
                    qwen_key,
                )
            )

    if unequal:

        print()
        print(
            "ART tensors with unequal values after transfer:"
        )

        for demo_key, qwen_key in unequal[:30]:

            print(
                f"  - ART : {demo_key}"
            )

            print(
                f"    Qwen: {qwen_key}"
            )

        if len(unequal) > 30:

            print(
                f"  ... and "
                f"{len(unequal) - 30} more"
            )

        raise RuntimeError(
            "Weight transfer verification failed; "
            "nothing was uploaded."
        )

    print()
    print(
        f"✓ All {len(mapping)} mapped ART tensors "
        "match their Qwen source tensors exactly."
    )


# =============================================================================
# Verify decision branch duplication
# =============================================================================

def verify_decision_branch_copy(
    demo_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
) -> None:
    """
    Explicitly verify that:

        decision_layers.0...13

    are exact copies of:

        Qwen layers.22...35
    """

    print()
    print("=" * 80)
    print("VERIFYING DECISION BRANCH INITIALIZATION")
    print("=" * 80)

    for decision_idx in range(NUM_DECISION_LAYERS):

        qwen_layer_idx = (
            DECISION_BRANCH_START_LAYER
            + decision_idx
        )

        demo_prefix = (
            f"model.decision_layers.{decision_idx}."
        )

        qwen_prefix = (
            f"model.layers.{qwen_layer_idx}."
        )

        matching_keys = [
            key
            for key in demo_state
            if key.startswith(demo_prefix)
        ]

        if not matching_keys:

            raise RuntimeError(
                f"No ART parameters found for "
                f"decision layer {decision_idx}."
            )

        for demo_key in matching_keys:

            suffix = demo_key[
                len(demo_prefix):
            ]

            qwen_key = (
                qwen_prefix
                + suffix
            )

            if qwen_key not in qwen_state:

                raise RuntimeError(
                    f"Missing Qwen source tensor "
                    f"for decision layer {decision_idx}: "
                    f"{qwen_key}"
                )

            if not torch.equal(
                demo_state[demo_key].cpu(),
                qwen_state[qwen_key].cpu(),
            ):

                raise RuntimeError(
                    "Decision branch transfer verification failed:\n"
                    f"  ART : {demo_key}\n"
                    f"  Qwen: {qwen_key}"
                )

        print(
            f"  ✓ decision_layers.{decision_idx:02d} "
            f"<- model.layers.{qwen_layer_idx:02d}"
        )

    print()
    print(
        "✓ All 14 decision layers were copied "
        "from Qwen layers 22-35."
    )


# =============================================================================
# Verify decision branch is physically separate
# =============================================================================

def verify_decision_branch_is_separate(
    demo,
) -> None:
    """
    Verify that the decision branch has its own Parameter objects.

    The values should initially be equal to the corresponding Qwen layers,
    but the Parameter objects must be different from the LM branch.
    """

    print()
    print("=" * 80)
    print("VERIFYING DECISION BRANCH PARAMETER SEPARATION")
    print("=" * 80)

    for decision_idx in range(NUM_DECISION_LAYERS):

        qwen_layer_idx = (
            DECISION_BRANCH_START_LAYER
            + decision_idx
        )

        decision_layer = (
            demo.model.decision_layers[
                decision_idx
            ]
        )

        lm_layer = (
            demo.model.layers[
                qwen_layer_idx
            ]
        )

        decision_parameters = dict(
            decision_layer.named_parameters()
        )

        lm_parameters = dict(
            lm_layer.named_parameters()
        )

        for name, decision_parameter in (
            decision_parameters.items()
        ):

            if name not in lm_parameters:

                raise RuntimeError(
                    f"Parameter {name} exists in "
                    f"decision layer but not in LM layer."
                )

            lm_parameter = lm_parameters[name]

            if decision_parameter is lm_parameter:

                raise RuntimeError(
                    "Decision branch is sharing a Parameter "
                    "object with the LM branch:\n"
                    f"  decision layer: {decision_idx}\n"
                    f"  LM layer:       {qwen_layer_idx}\n"
                    f"  parameter:      {name}"
                )

    print(
        "✓ Decision branch parameters are physically "
        "separate from the LM branch."
    )


# =============================================================================
# Verify decision normalization
# =============================================================================

def verify_decision_norm(
    demo_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
    demo,
) -> None:
    """
    Verify that:

        ART model.norm
            <- Qwen model.norm

        ART model.decision_norm
            <- Qwen model.norm

    and verify that model.norm and decision_norm are physically
    separate Parameter objects.
    """

    print()
    print("=" * 80)
    print("VERIFYING DECISION NORMALIZATION")
    print("=" * 80)

    demo_norm_key = "model.norm.weight"
    demo_decision_norm_key = "model.decision_norm.weight"
    qwen_norm_key = "model.norm.weight"

    # =========================================================================
    # Verify state-dict keys
    # =========================================================================

    for key in (
        demo_norm_key,
        demo_decision_norm_key,
    ):

        if key not in demo_state:

            raise RuntimeError(
                f"Missing ART normalization parameter: {key}"
            )

    if qwen_norm_key not in qwen_state:

        raise RuntimeError(
            f"Missing Qwen normalization parameter: "
            f"{qwen_norm_key}"
        )

    # =========================================================================
    # Verify values
    # =========================================================================

    if not torch.equal(
        demo_state[demo_norm_key].cpu(),
        qwen_state[qwen_norm_key].cpu(),
    ):

        raise RuntimeError(
            "ART model.norm does not match "
            "Qwen model.norm."
        )

    if not torch.equal(
        demo_state[demo_decision_norm_key].cpu(),
        qwen_state[qwen_norm_key].cpu(),
    ):

        raise RuntimeError(
            "ART model.decision_norm does not match "
            "Qwen model.norm."
        )

    print(
        "✓ model.norm <- Qwen model.norm"
    )

    print(
        "✓ model.decision_norm <- Qwen model.norm"
    )

    # =========================================================================
    # Verify physical parameter separation
    # =========================================================================

    lm_norm_parameter = demo.model.norm.weight

    decision_norm_parameter = (
        demo.model.decision_norm.weight
    )

    if lm_norm_parameter is decision_norm_parameter:

        raise RuntimeError(
            "Decision norm shares the same Parameter object "
            "with LM norm."
        )

    if (
        lm_norm_parameter.data_ptr()
        == decision_norm_parameter.data_ptr()
    ):

        raise RuntimeError(
            "Decision norm shares the same underlying storage "
            "with LM norm."
        )

    print(
        "✓ model.norm and model.decision_norm "
        "are physically separate parameters."
    )

    print()
    print(
        f"Normalization shape: "
        f"{tuple(demo_decision_norm_parameter.shape) if False else tuple(decision_norm_parameter.shape)}"
    )

    print(
        "✓ Decision normalization is initialized "
        "from the original Qwen RMSNorm."
    )


# =============================================================================
# ART INITIALIZATION
# =============================================================================

def initialize_retrieval_decision_head(
    demo,
) -> None:
    """
    Randomly initialize the complete ART retrieval decision head.

    Architecture:

        hidden_size
            ↓
            1
            ↓
          sigmoid

    For Qwen2.5-3B:

        2048 → 1

    The Linear layer is initialized using the same initialization as
    torch.nn.Linear.reset_parameters():

        weight -> Kaiming uniform
        bias   -> uniform based on fan-in
    """

    print()
    print("=" * 80)
    print("INITIALIZING ART RETRIEVAL DECISION HEAD")
    print("=" * 80)

    head = demo.retrieval_decision_head

    network = head.network

    hidden_size = demo.config.hidden_size

    # =========================================================================
    # Report architecture
    # =========================================================================

    print()
    print("Retrieval decision head architecture:")

    print(
        f"  Input hidden size : "
        f"{hidden_size}"
    )

    print(
        f"  Output layer      : "
        f"{hidden_size} → 1"
    )

    print(
        "  Output activation : "
        "sigmoid"
    )

    # =========================================================================
    # Collect Linear layers
    # =========================================================================

    linear_layers = [
        module
        for module in network.modules()
        if isinstance(
            module,
            torch.nn.Linear,
        )
    ]

    if not linear_layers:

        raise RuntimeError(
            "No nn.Linear layers were found inside "
            "retrieval_decision_head.network."
        )

    print()
    print(
        f"Found {len(linear_layers)} Linear layer "
        "inside the retrieval decision head."
    )

    # =========================================================================
    # Verify expected number of Linear layers
    # =========================================================================

    if len(linear_layers) != 1:

        raise RuntimeError(
            "Expected exactly 1 Linear layer in the "
            "retrieval decision head."
        )

    # =========================================================================
    # Verify expected dimensions
    # =========================================================================

    expected_dimensions = [
        (
            hidden_size,
            1,
        ),
    ]

    actual_dimensions = [
        (
            layer.in_features,
            layer.out_features,
        )
        for layer in linear_layers
    ]

    if actual_dimensions != expected_dimensions:

        raise RuntimeError(
            "Retrieval decision head architecture mismatch.\n"
            f"Expected: {expected_dimensions}\n"
            f"Actual:   {actual_dimensions}"
        )

    print()
    print("Linear architecture verified:")

    for in_features, out_features in actual_dimensions:

        print(
            f"  {in_features} → {out_features}"
        )

    # =========================================================================
    # Before initialization
    # =========================================================================

    print()
    print("Before random initialization:")

    for index, layer in enumerate(linear_layers):

        print()
        print(
            f"Linear layer {index + 1}: "
            f"{layer.in_features} → "
            f"{layer.out_features}"
        )

        print(
            f"  Weight mean     : "
            f"{layer.weight.detach().float().mean().item():+.6e}"
        )

        print(
            f"  Weight std      : "
            f"{layer.weight.detach().float().std().item():.6e}"
        )

        print(
            f"  Weight abs mean : "
            f"{layer.weight.detach().float().abs().mean().item():.6e}"
        )

        if layer.bias is not None:

            print(
                f"  Bias mean       : "
                f"{layer.bias.detach().float().mean().item():+.6e}"
            )

            print(
                f"  Bias std        : "
                f"{layer.bias.detach().float().std().item():.6e}"
            )

    # =========================================================================
    # Random initialization
    #
    # Match torch.nn.Linear.reset_parameters().
    # =========================================================================

    with torch.no_grad():

        for layer in linear_layers:

            torch.nn.init.kaiming_uniform_(
                layer.weight,
                a=5 ** 0.5,
            )

            if layer.bias is not None:

                fan_in, _ = (
                    torch.nn.init._calculate_fan_in_and_fan_out(
                        layer.weight
                    )
                )

                bound = 1 / (fan_in ** 0.5)

                torch.nn.init.uniform_(
                    layer.bias,
                    -bound,
                    bound,
                )

    # =========================================================================
    # After initialization
    # =========================================================================

    print()
    print("After random initialization:")

    for index, layer in enumerate(linear_layers):

        print()
        print(
            f"Linear layer {index + 1}: "
            f"{layer.in_features} → "
            f"{layer.out_features}"
        )

        print(
            f"  Weight mean     : "
            f"{layer.weight.detach().float().mean().item():+.6e}"
        )

        print(
            f"  Weight std      : "
            f"{layer.weight.detach().float().std().item():.6e}"
        )

        print(
            f"  Weight abs mean : "
            f"{layer.weight.detach().float().abs().mean().item():.6e}"
        )

        print(
            f"  Weight min      : "
            f"{layer.weight.detach().float().min().item():+.6e}"
        )

        print(
            f"  Weight max      : "
            f"{layer.weight.detach().float().max().item():+.6e}"
        )

        if layer.bias is not None:

            print(
                f"  Bias mean       : "
                f"{layer.bias.detach().float().mean().item():+.6e}"
            )

            print(
                f"  Bias std        : "
                f"{layer.bias.detach().float().std().item():.6e}"
            )

    print()
    print("Initialization method:")

    print(
        "  weight = Kaiming uniform"
    )

    print(
        "  bias   = uniform(-1/sqrt(fan_in), +1/sqrt(fan_in))"
    )

    print(
        "  score = sigmoid(Linear(hidden_state))"
    )

    print("=" * 80)


# =============================================================================
# VERIFY RANDOM DECISION HEAD
# =============================================================================

def verify_retrieval_decision_head(
    demo,
) -> None:
    """
    Verify that the complete ART retrieval decision head is randomly
    initialized and produces valid decision scores.
    """

    print()
    print("=" * 80)
    print("VERIFYING RANDOM RETRIEVAL DECISION HEAD")
    print("=" * 80)

    head = demo.retrieval_decision_head

    network = head.network

    hidden_size = demo.config.hidden_size

    # =========================================================================
    # Find Linear layers
    # =========================================================================

    linear_layers = [
        module
        for module in network.modules()
        if isinstance(
            module,
            torch.nn.Linear,
        )
    ]

    if not linear_layers:

        raise RuntimeError(
            "No nn.Linear layers were found inside "
            "retrieval_decision_head.network."
        )

    # =========================================================================
    # Verify number of Linear layers
    # =========================================================================

    if len(linear_layers) != 1:

        raise RuntimeError(
            "Expected exactly 1 Linear layer in the "
            "retrieval decision head."
        )

    # =========================================================================
    # Verify architecture
    # =========================================================================

    expected_dimensions = [
        (
            hidden_size,
            1,
        ),
    ]

    actual_dimensions = [
        (
            layer.in_features,
            layer.out_features,
        )
        for layer in linear_layers
    ]

    print()
    print("Expected Linear architecture:")

    for in_features, out_features in expected_dimensions:

        print(
            f"  {in_features} → {out_features}"
        )

    print()
    print("Actual Linear architecture:")

    for in_features, out_features in actual_dimensions:

        print(
            f"  {in_features} → {out_features}"
        )

    if actual_dimensions != expected_dimensions:

        raise RuntimeError(
            "Retrieval decision head architecture does not "
            "match the expected structure."
        )

    print()
    print(
        "✓ Retrieval decision head architecture is correct."
    )

    # =========================================================================
    # Weight verification
    # =========================================================================

    print()
    print("Checking Linear layer weights:")

    for index, layer in enumerate(linear_layers):

        weight_abs_mean = (
            layer.weight.detach()
            .float()
            .abs()
            .mean()
            .item()
        )

        weight_std = (
            layer.weight.detach()
            .float()
            .std()
            .item()
        )

        print()
        print(
            f"Linear layer {index + 1}:"
        )

        print(
            f"  Shape           : "
            f"{tuple(layer.weight.shape)}"
        )

        print(
            f"  Weight abs mean : "
            f"{weight_abs_mean:.6e}"
        )

        print(
            f"  Weight std      : "
            f"{weight_std:.6e}"
        )

        if weight_abs_mean == 0.0:

            raise RuntimeError(
                f"Retrieval decision head Linear layer "
                f"{index + 1} has zero weights."
            )

        print(
            "  ✓ Weight is non-zero."
        )

    # =========================================================================
    # Random hidden states
    # =========================================================================

    device = linear_layers[0].weight.device

    dtype = linear_layers[0].weight.dtype

    test_hidden = torch.randn(
        8,
        1,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():

        decision_scores = head(
            test_hidden
        )

    scores = (
        decision_scores
        .detach()
        .float()
        .cpu()
    )

    # =========================================================================
    # Verify output shape
    # =========================================================================

    expected_shape = (
        test_hidden.shape[0],
    )

    print()
    print(
        f"Decision output shape: "
        f"{tuple(scores.shape)}"
    )

    if tuple(scores.shape) != expected_shape:

        raise RuntimeError(
            f"Unexpected retrieval decision output shape: "
            f"{tuple(scores.shape)}; "
            f"expected {expected_shape}."
        )

    print(
        "✓ Decision output shape is correct."
    )

    # =========================================================================
    # Verify sigmoid range
    # =========================================================================

    if not torch.all(
        (scores >= 0.0)
        & (scores <= 1.0)
    ):

        raise RuntimeError(
            "Retrieval decision scores are outside "
            "the [0, 1] sigmoid range."
        )

    print(
        "✓ Decision scores are within [0, 1]."
    )

    # =========================================================================
    # Decision score statistics
    # =========================================================================

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
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():

        zero_decision_score = head(
            zero_hidden
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
        "This is the actual output of the complete "
        "Linear → sigmoid pipeline "
        "for a zero hidden state."
    )

    print()
    print(
        "✓ Retrieval decision head is randomly initialized."
    )

    print("=" * 80)


# =============================================================================
# Report architecture
# =============================================================================

def report_art_architecture(
    demo,
) -> None:
    """
    Print the final ART architecture relevant to initialization.
    """

    print()
    print("=" * 80)
    print("ART ARCHITECTURE")
    print("=" * 80)

    print()
    print(
        f"Hidden size: "
        f"{demo.config.hidden_size}"
    )

    print(
        f"Vocabulary size: "
        f"{demo.config.vocab_size}"
    )

    print(
        f"Total Qwen layers: "
        f"{demo.config.num_hidden_layers}"
    )

    print(
        f"Shared layers: "
        f"0 ... {DECISION_BRANCH_START_LAYER - 1}"
    )

    print(
        f"LM branch: "
        f"{DECISION_BRANCH_START_LAYER} ... "
        f"{QWEN_NUM_LAYERS - 1}"
    )

    print(
        f"Decision branch: "
        f"{NUM_DECISION_LAYERS} layers"
    )

    print(
        f"Decision branch initialization: "
        f"Qwen layers "
        f"{DECISION_BRANCH_START_LAYER} ... "
        f"{QWEN_NUM_LAYERS - 1}"
    )

    print(
        "LM normalization: "
        "Qwen model.norm"
    )

    print(
        "Decision normalization: "
        "independent copy of Qwen model.norm"
    )

    print(
        "Retrieval decision head: "
        f"{demo.config.hidden_size} → 1 → sigmoid"
    )

    print()
    print(
        "Parameter relationship:"
    )

    print(
        "  model.layers.22"
        "  <-> independent from"
        "  model.decision_layers.0"
    )

    print(
        "  model.layers.23"
        "  <-> independent from"
        "  model.decision_layers.1"
    )

    print(
        "  ..."
    )

    print(
        "  model.layers.35"
        "  <-> independent from"
        "  model.decision_layers.13"
    )

    print()
    print(
        "  model.norm"
        "  <-> independent from"
        "  model.decision_norm"
    )

    print()
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
        "INITIALIZE ART: "
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
        "\n1. Loading custom ART config "
        "and creating an empty model..."
    )

    config = AutoConfig.from_pretrained(
        CUSTOM_MODEL_DIR,
        trust_remote_code=True,
    )

    # -------------------------------------------------------------------------
    # Verify architecture configuration
    # -------------------------------------------------------------------------

    if config.num_hidden_layers != QWEN_NUM_LAYERS:

        raise RuntimeError(
            "Unexpected number of Qwen layers in ART config.\n"
            f"Expected: {QWEN_NUM_LAYERS}\n"
            f"Actual:   {config.num_hidden_layers}"
        )

    if (
        config.decision_branch_start_layer
        != DECISION_BRANCH_START_LAYER
    ):

        raise RuntimeError(
            "Unexpected decision branch start layer.\n"
            f"Expected: {DECISION_BRANCH_START_LAYER}\n"
            f"Actual:   {config.decision_branch_start_layer}"
        )

    if (
        config.num_decision_layers
        != NUM_DECISION_LAYERS
    ):

        raise RuntimeError(
            "Unexpected number of decision layers.\n"
            f"Expected: {NUM_DECISION_LAYERS}\n"
            f"Actual:   {config.num_decision_layers}"
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

    print(
        f"   Hidden size: "
        f"{demo.config.hidden_size}"
    )

    # =========================================================================
    # Verify normalization modules exist
    # =========================================================================

    if not hasattr(
        demo.model,
        "norm",
    ):

        raise RuntimeError(
            "ART model is missing model.norm."
        )

    if not hasattr(
        demo.model,
        "decision_norm",
    ):

        raise RuntimeError(
            "ART model is missing model.decision_norm."
        )

    print(
        "   ✓ model.norm exists."
    )

    print(
        "   ✓ model.decision_norm exists."
    )

    # =========================================================================
    # 1.1 Report architecture
    # =========================================================================

    report_art_architecture(
        demo=demo,
    )

    # =========================================================================
    # 1.2 Randomly initialize ART retrieval decision head
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

    print(
        f"   Original layers: "
        f"{qwen.config.num_hidden_layers}"
    )

    if qwen.config.num_hidden_layers != QWEN_NUM_LAYERS:

        raise RuntimeError(
            "Unexpected number of layers in original Qwen model.\n"
            f"Expected: {QWEN_NUM_LAYERS}\n"
            f"Actual:   {qwen.config.num_hidden_layers}"
        )

    # =========================================================================
    # 3. Collect state dictionaries
    # =========================================================================

    print(
        "\n3. Building explicit ART <- Qwen "
        "weight transfer mapping..."
    )

    demo_state = demo.state_dict()

    qwen_state = qwen.state_dict()

    transfer_mapping = build_qwen_transfer_mapping(
        demo_state=demo_state,
        qwen_state=qwen_state,
    )

    print(
        f"   Transfer mapping contains "
        f"{len(transfer_mapping)} ART tensors."
    )

    # =========================================================================
    # 3.1 Verify mapping
    # =========================================================================

    print(
        "\n3.1 Checking transfer compatibility..."
    )

    verify_transfer_mapping(
        demo_state=demo_state,
        qwen_state=qwen_state,
        mapping=transfer_mapping,
    )

    # =========================================================================
    # 4. Transfer Qwen weights
    # =========================================================================

    print(
        "\n4. Transferring Qwen weights into ART demo..."
    )

    merged_state = demo.state_dict()

    # -------------------------------------------------------------------------
    # Transfer every explicitly mapped ART parameter.
    #
    # Important:
    #
    #     model.decision_layers.*
    #
    # receives a COPY of Qwen's last 14 layers.
    #
    #     model.decision_norm.*
    #
    # receives a COPY of Qwen's final RMSNorm.
    #
    #     retrieval_decision_head.*
    #
    # is NOT touched by the Qwen transfer.
    # -------------------------------------------------------------------------

    for demo_key, qwen_key in transfer_mapping.items():

        merged_state[demo_key] = (
            qwen_state[qwen_key]
        )

    demo.load_state_dict(
        merged_state,
        strict=True,
    )

    print(
        f"   Transferred "
        f"{len(transfer_mapping)} ART parameter tensors."
    )

    print()
    print(
        "   Qwen weights transferred:"
    )

    print(
        "     - model.embed_tokens"
    )

    print(
        "     - model.layers.0...35"
    )

    print(
        "     - model.norm"
    )

    print(
        "     - lm_head"
    )

    print()
    print(
        "   Decision branch transferred:"
    )

    print(
        "     - decision_layers.0"
        "  <- Qwen model.layers.22"
    )

    print(
        "     - decision_layers.1"
        "  <- Qwen model.layers.23"
    )

    print(
        "     - ..."
    )

    print(
        "     - decision_layers.13"
        " <- Qwen model.layers.35"
    )

    print()
    print(
        "   Decision normalization transferred:"
    )

    print(
        "     - decision_norm"
        " <- Qwen model.norm"
    )

    print()
    print(
        "   ART retrieval decision head was preserved "
        "with its random initialization."
    )

    # =========================================================================
    # 5. Verify transferred values
    # =========================================================================

    print(
        "\n5. Verifying transferred Qwen weights..."
    )

    verify_transferred_values(
        demo_state=demo.state_dict(),
        qwen_state=qwen_state,
        mapping=transfer_mapping,
    )

    # =========================================================================
    # 5.1 Explicitly verify decision branch
    # =========================================================================

    verify_decision_branch_copy(
        demo_state=demo.state_dict(),
        qwen_state=qwen_state,
    )

    # =========================================================================
    # 5.2 Verify decision branch is physically separate
    # =========================================================================

    verify_decision_branch_is_separate(
        demo=demo,
    )

    # =========================================================================
    # 5.3 Verify decision normalization
    # =========================================================================

    verify_decision_norm(
        demo_state=demo.state_dict(),
        qwen_state=qwen_state,
        demo=demo,
    )

    # =========================================================================
    # 5.4 Verify ART decision head after Qwen transfer
    # =========================================================================

    print(
        "\n5.4 Verifying ART retrieval decision head..."
    )

    verify_retrieval_decision_head(
        demo=demo,
    )

    # =========================================================================
    # 6. Report ART-specific parameters
    # =========================================================================

    additional_keys = sorted(
        set(demo.state_dict())
        - set(transfer_mapping)
    )

    print()
    print("=" * 80)
    print("ART-SPECIFIC PARAMETERS")
    print("=" * 80)

    print()

    for name in additional_keys:

        print(
            f"  - {name}"
        )

    print()
    print(
        f"Total ART-specific parameter tensors: "
        f"{len(additional_keys)}"
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
        prefix="art-qwen2.5-3b-"
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
                "Initialize ART from Qwen2.5-3B-Instruct with "
                "22 shared + 14-layer LM branch + "
                "14-layer decision branch, "
                "independent decision norm, and "
                "random 2048-1 retrieval head"
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
        "ART architecture:"
    )

    print(
        "  Shared: "
        "Qwen layers 0-21"
    )

    print(
        "  LM branch: "
        "Qwen layers 22-35"
    )

    print(
        "  LM norm: "
        "Qwen model.norm"
    )

    print(
        "  Decision branch: "
        "copy of Qwen layers 22-35"
    )

    print(
        "  Decision norm: "
        "independent copy of Qwen model.norm"
    )

    print(
        "  Decision head: "
        "random 2048 → 1 → sigmoid"
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
# Entry point
# =============================================================================

if __name__ == "__main__":

    main()