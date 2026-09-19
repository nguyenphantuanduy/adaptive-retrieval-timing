"""Build and publish the Qwen2.5-3B ART attention-mixing probe.

Run this file from the repository root:

    .\\.venv\\Scripts\\python.exe \
        shared/ART_Models/attention_mixing_probe/Qwen2.5_3B/initialize_art_probe_qwen2_5_3b.py

The script refuses to publish unless the original Qwen checkpoint and the
custom probe have exactly the same state-dict keys and tensor shapes.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM


# The directory containing this script, config.json, and all custom modules.
CUSTOM_MODEL_DIR = Path(__file__).resolve().parent

QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct"
TARGET_REPO = "adaptive-retrieval-timing/ART-Attention-Mixing-Probe"
DTYPE = torch.bfloat16


def print_names(title: str, names: list[str]) -> None:
    """Print a concise diagnostic before aborting on an incompatibility."""
    print(f"{title}: {len(names)}")
    for name in names[:30]:
        print(f"  - {name}")
    if len(names) > 30:
        print(f"  ... and {len(names) - 30} more")


def verify_state_dicts(
    probe_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
) -> None:
    """Require an exact key and shape match before any weight transfer."""
    probe_keys = set(probe_state)
    qwen_keys = set(qwen_state)

    only_in_qwen = sorted(qwen_keys - probe_keys)
    only_in_probe = sorted(probe_keys - qwen_keys)
    shape_mismatches = sorted(
        (
            name,
            tuple(qwen_state[name].shape),
            tuple(probe_state[name].shape),
        )
        for name in qwen_keys & probe_keys
        if qwen_state[name].shape != probe_state[name].shape
    )

    print("\nState-dict compatibility report")
    print("-" * 80)
    print_names("Only in original Qwen", only_in_qwen)
    print_names("Only in custom probe", only_in_probe)
    print(f"Shape mismatches: {len(shape_mismatches)}")
    for name, qwen_shape, probe_shape in shape_mismatches[:30]:
        print(f"  - {name}: qwen={qwen_shape}, probe={probe_shape}")
    if len(shape_mismatches) > 30:
        print(f"  ... and {len(shape_mismatches) - 30} more")

    if only_in_qwen or only_in_probe or shape_mismatches:
        raise RuntimeError(
            "State dicts are not exactly compatible; nothing was uploaded."
        )

    print(f"All {len(qwen_keys)} state-dict tensors have matching names and shapes.")


def verify_transferred_values(
    probe_state: dict[str, torch.Tensor],
    qwen_state: dict[str, torch.Tensor],
) -> None:
    """Ensure strict loading copied every tensor exactly, not just structurally."""
    unequal = [
        name
        for name in qwen_state
        if not torch.equal(probe_state[name].cpu(), qwen_state[name].cpu())
    ]

    if unequal:
        print_names("Tensors with unequal values after transfer", unequal)
        raise RuntimeError("Weight transfer verification failed; nothing was uploaded.")

    print(f"All {len(qwen_state)} tensors match the original Qwen checkpoint exactly.")


def copy_custom_code(export_dir: Path) -> None:
    """Place all local custom-model source files beside the saved weights."""
    for source_file in CUSTOM_MODEL_DIR.glob("*.py"):
        shutil.copy2(source_file, export_dir / source_file.name)

    # Preserve the hand-authored auto_map that makes AutoModel load this class.
    shutil.copy2(CUSTOM_MODEL_DIR / "config.json", export_dir / "config.json")


def main() -> None:
    print("=" * 80)
    print("INITIALIZE ART ATTENTION-MIXING PROBE: QWEN2.5-3B-INSTRUCT")
    print("=" * 80)

    print("\n1. Loading custom probe config and creating an empty probe...")
    config = AutoConfig.from_pretrained(
        CUSTOM_MODEL_DIR,
        trust_remote_code=True,
    )
    probe = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=DTYPE,
    )
    print(f"   Probe class: {type(probe).__name__}")
    print(f"   Probe dtype: {probe.model.embed_tokens.weight.dtype}")

    print("\n2. Loading original Qwen2.5-3B-Instruct checkpoint on CPU...")
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_REPO,
        torch_dtype=DTYPE,
        device_map="cpu",
    )
    print(f"   Original class: {type(qwen).__name__}")
    print(f"   Original dtype: {qwen.model.embed_tokens.weight.dtype}")

    probe_state = probe.state_dict()
    qwen_state = qwen.state_dict()
    verify_state_dicts(probe_state, qwen_state)

    print("\n3. Transferring weights with strict loading...")
    probe.load_state_dict(qwen_state, strict=True)
    verify_transferred_values(probe.state_dict(), qwen_state)

    # Free the original model before serializing the probe and uploading it.
    del qwen
    del qwen_state
    gc.collect()

    print("\n4. Saving model weights and custom source into a publish directory...")
    with tempfile.TemporaryDirectory(prefix="art-qwen2.5-3b-") as temp_dir:
        export_dir = Path(temp_dir)
        probe.save_pretrained(export_dir, safe_serialization=True)
        copy_custom_code(export_dir)

        saved_files = sorted(path.name for path in export_dir.iterdir())
        print("   Files to publish:")
        for filename in saved_files:
            print(f"   - {filename}")

        print(f"\n5. Uploading and overwriting {TARGET_REPO}...")
        api = HfApi()
        api.create_repo(TARGET_REPO, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            repo_id=TARGET_REPO,
            repo_type="model",
            folder_path=export_dir,
            commit_message="Publish Qwen2.5-3B ART attention-mixing probe",
        )

    print("\n" + "=" * 80)
    print("PUBLISH COMPLETE")
    print("=" * 80)
    print(f"Base model: {QWEN_REPO}")
    print(f"Published model: https://huggingface.co/{TARGET_REPO}")
    print(f"Commit: {commit}")


if __name__ == "__main__":
    main()
