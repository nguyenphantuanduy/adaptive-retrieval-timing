# ============================================================
# STAGE 1 — HYPOTHESIS VERIFICATION
# QWEN3-4B WEIGHT TRANSFER → ATTENTION MIXING PROBE
#
# Purpose:
#   Construct AttentionMixingProbe from the custom config,
#   load Qwen3-4B weights, transfer all compatible weights,
#   and dispatch the resulting probe across 2 GPUs.
#
# This script prepares the model for Stage 1 hypothesis tests.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import gc

import torch

from huggingface_hub import snapshot_download

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from accelerate import (
    infer_auto_device_map,
    dispatch_model,
)


# ============================================================
# 2. CONFIG
# ============================================================

DTYPE = torch.bfloat16

PROBE_REPO = (
    "adaptive-retrieval-timing/"
    "ART-Attention-Mixing-Probe"
)

PROBE_REVISION = "dev"

QWEN_REPO = "Qwen/Qwen3-4B"

EXPECTED_GPUS = 2

MAX_MEMORY = {
    0: "13GiB",
    1: "13GiB",
    "cpu": "30GiB",
}


# ============================================================
# 3. CLEAR ENVIRONMENT
# ============================================================

print("=" * 80)
print("STAGE 1 — HYPOTHESIS VERIFICATION")
print("QWEN3-4B → ATTENTION MIXING PROBE")
print("=" * 80)

print()
print("Clearing environment...")

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print()


# ============================================================
# 4. CHECK GPUs
# ============================================================

print("=" * 80)
print("0. GPU CHECK")
print("=" * 80)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

gpu_count = torch.cuda.device_count()

print(
    f"Available GPUs: {gpu_count}"
)

for i in range(gpu_count):

    props = torch.cuda.get_device_properties(i)

    print(
        f"GPU {i}: "
        f"{props.name} | "
        f"{props.total_memory / 1024**3:.2f} GB"
    )

print()

if gpu_count < EXPECTED_GPUS:
    raise RuntimeError(
        f"Expected at least {EXPECTED_GPUS} GPUs, "
        f"but found {gpu_count}."
    )


# ============================================================
# 5. DOWNLOAD PROBE REPOSITORY
# ============================================================

print("=" * 80)
print("1. DOWNLOAD ATTENTION MIXING PROBE")
print("=" * 80)

local_path = snapshot_download(
    repo_id=PROBE_REPO,
    revision=PROBE_REVISION,
)

print()
print("Probe repository:")
print(local_path)

print()


# ============================================================
# 6. LOAD PROBE CONFIG
# ============================================================

print("=" * 80)
print("2. LOAD PROBE CONFIG")
print("=" * 80)

config = AutoConfig.from_pretrained(
    local_path,
    trust_remote_code=True,
)

print(
    "Config class:",
    type(config),
)

print()


# ============================================================
# 7. CREATE PROBE ON CPU
# ============================================================

print("=" * 80)
print("3. CREATE ATTENTION MIXING PROBE")
print("=" * 80)

probe = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
)

print(
    "Model:",
    type(probe),
)

print(
    "Decoder layer:",
    type(probe.model.layers[0]),
)

print(
    "Attention:",
    type(
        probe.model.layers[0].self_attn
    ),
)

print(
    "Initial dtype:",
    probe.model.embed_tokens.weight.dtype,
)

print()


# ============================================================
# 8. CONVERT PROBE TO BF16
#
# Important:
#   Keep the probe on CPU while converting dtype.
#   This avoids unnecessary GPU memory usage.
# ============================================================

print("=" * 80)
print("4. CONVERT PROBE FP32 → BF16")
print("=" * 80)

probe = probe.to(
    dtype=DTYPE
)

print(
    "Probe dtype:",
    probe.model.embed_tokens.weight.dtype,
)

print(
    "Probe device:",
    probe.model.embed_tokens.weight.device,
)

print()


# ============================================================
# 9. LOAD QWEN3-4B ON CPU
# ============================================================

print("=" * 80)
print("5. LOAD QWEN3-4B")
print("=" * 80)

qwen = AutoModelForCausalLM.from_pretrained(
    QWEN_REPO,
    torch_dtype=DTYPE,
    device_map="cpu",
)

print(
    "Qwen loaded."
)

print(
    "Embedding device:",
    qwen.model.embed_tokens.weight.device,
)

print(
    "Qwen dtype:",
    qwen.model.embed_tokens.weight.dtype,
)

print()


# ============================================================
# 10. TRANSFER MATCHING WEIGHTS
# ============================================================

print("=" * 80)
print("6. TRANSFER MATCHING WEIGHTS")
print("=" * 80)

probe_state = probe.state_dict()
qwen_state = qwen.state_dict()

copied = []
skipped = []
shape_mismatch = []

with torch.no_grad():

    for name, probe_param in probe_state.items():

        # ----------------------------------------------------
        # Parameter does not exist in Qwen
        # ----------------------------------------------------

        if name not in qwen_state:

            skipped.append(name)

            continue

        qwen_param = qwen_state[name]

        # ----------------------------------------------------
        # Parameter exists but shapes differ
        # ----------------------------------------------------

        if probe_param.shape != qwen_param.shape:

            shape_mismatch.append(
                (
                    name,
                    tuple(probe_param.shape),
                    tuple(qwen_param.shape),
                )
            )

            continue

        # ----------------------------------------------------
        # Compatible parameter
        # ----------------------------------------------------

        probe_param.copy_(
            qwen_param
        )

        copied.append(name)


# ============================================================
# 11. TRANSFER REPORT
# ============================================================

print()
print("-" * 80)
print("TRANSFER REPORT")
print("-" * 80)

print(
    f"Copied parameters : {len(copied)}"
)

print(
    f"Skipped parameters: {len(skipped)}"
)

print(
    f"Shape mismatches  : {len(shape_mismatch)}"
)

print()


# ============================================================
# 12. PRINT SKIPPED PARAMETERS
# ============================================================

if skipped:

    print(
        "Skipped parameters:"
    )

    for name in skipped[:30]:

        print(
            "  ",
            name,
        )

    if len(skipped) > 30:

        print(
            f"  ... and "
            f"{len(skipped) - 30} more"
        )

    print()


# ============================================================
# 13. PRINT SHAPE MISMATCHES
# ============================================================

if shape_mismatch:

    print(
        "Shape mismatches:"
    )

    for (
        name,
        probe_shape,
        qwen_shape,
    ) in shape_mismatch[:30]:

        print(
            f"  {name}: "
            f"probe={probe_shape}, "
            f"qwen={qwen_shape}"
        )

    if len(shape_mismatch) > 30:

        print(
            f"  ... and "
            f"{len(shape_mismatch) - 30} more"
        )

    print()


# ============================================================
# 14. DELETE QWEN
#
# Qwen is no longer needed after weight transfer.
# Free CPU RAM before GPU dispatch.
# ============================================================

print("=" * 80)
print("7. DELETE ORIGINAL QWEN MODEL")
print("=" * 80)

del qwen
del qwen_state

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print(
    "Qwen deleted."
)

print()


# ============================================================
# 15. VERIFY CPU PROBE
# ============================================================

print("=" * 80)
print("8. VERIFY CPU PROBE")
print("=" * 80)

print(
    "Device:",
    probe.model.embed_tokens.weight.device,
)

print(
    "Dtype:",
    probe.model.embed_tokens.weight.dtype,
)

print()


# ============================================================
# 16. CREATE 2-GPU DEVICE MAP
# ============================================================

print("=" * 80)
print("9. CREATE 2-GPU DEVICE MAP")
print("=" * 80)

device_map = infer_auto_device_map(
    probe,
    max_memory=MAX_MEMORY,
    dtype=DTYPE,
    no_split_module_classes=[
        "AttentionMixingProbeDecoderLayer",
    ],
)

print(
    "Device map:"
)

device_counts = {}

for module_name, device in device_map.items():

    device = str(device)

    device_counts[device] = (
        device_counts.get(device, 0)
        + 1
    )

for device, count in sorted(
    device_counts.items()
):

    print(
        f"  {device}: "
        f"{count} modules"
    )

print()


# ============================================================
# 17. DISPATCH PROBE
# ============================================================

print("=" * 80)
print("10. DISPATCH PROBE ACROSS GPUs")
print("=" * 80)

probe = dispatch_model(
    probe,
    device_map=device_map,
)

print(
    "✓ Probe dispatched."
)

print()


# ============================================================
# 18. GPU MEMORY REPORT
# ============================================================

print("=" * 80)
print("11. GPU MEMORY")
print("=" * 80)

for i in range(
    torch.cuda.device_count()
):

    allocated = (
        torch.cuda.memory_allocated(i)
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(i)
        / 1024**3
    )

    free, total = (
        torch.cuda.mem_get_info(i)
    )

    print(
        f"GPU {i}:"
    )

    print(
        f"  Allocated: "
        f"{allocated:.2f} GB"
    )

    print(
        f"  Reserved : "
        f"{reserved:.2f} GB"
    )

    print(
        f"  Free     : "
        f"{free / 1024**3:.2f} GB"
    )

    print(
        f"  Total    : "
        f"{total / 1024**3:.2f} GB"
    )

    print()


# ============================================================
# 19. FINAL MODEL VERIFICATION
# ============================================================

print("=" * 80)
print("12. FINAL VERIFICATION")
print("=" * 80)

print(
    "Model:",
    type(probe),
)

print(
    "Layer 0:",
    type(probe.model.layers[0]),
)

print(
    "Layer 0 attention:",
    type(
        probe.model.layers[0].self_attn
    ),
)

print(
    "Embedding device:",
    probe.model.embed_tokens.weight.device,
)

print(
    "Embedding dtype:",
    probe.model.embed_tokens.weight.dtype,
)

print(
    "LM head device:",
    probe.lm_head.weight.device,
)

print()


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("=" * 80)
print("STAGE 1 MODEL SETUP COMPLETE")
print("=" * 80)

print()

print(
    "Probe repository:",
    PROBE_REPO,
)

print(
    "Probe revision:",
    PROBE_REVISION,
)

print(
    "Base model:",
    QWEN_REPO,
)

print(
    "Dtype:",
    DTYPE,
)

print(
    "Copied parameters:",
    len(copied),
)

print(
    "Skipped parameters:",
    len(skipped),
)

print(
    "Shape mismatches:",
    len(shape_mismatch),
)

print(
    "GPU count:",
    torch.cuda.device_count(),
)

print()

print(
    "✓ Qwen3-4B weights transferred"
)

print(
    "✓ AttentionMixingProbe initialized"
)

print(
    "✓ Probe dispatched across GPUs"
)

print(
    "✓ Ready for Stage 1 hypothesis verification"
)

print()

print("=" * 80)
print("DONE")
print("=" * 80)