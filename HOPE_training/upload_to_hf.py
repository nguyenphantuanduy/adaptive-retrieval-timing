from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


# ============================================================
# CONFIG
# ============================================================

LOCAL_MODEL_DIR = Path("./art-training-output")

HF_REPO_ID = "adaptive-retrieval-timing/ART-HOPE"

REPO_TYPE = "model"

WEIGHT_FILE = "model.safetensors"


# ============================================================
# UPLOAD
# ============================================================

def upload_model_to_hf() -> None:
    """
    Upload only the final trained model weights.

    The Hugging Face repository already contains:
        - ART model code
        - initial model weights
        - config
        - tokenizer

    Therefore this script only replaces/uploads:
        model.safetensors
    """

    print()
    print("=" * 80)
    print("UPLOADING FINAL ART MODEL WEIGHTS")
    print("=" * 80)

    # ========================================================
    # 1. Check local model
    # ========================================================

    weight_path = LOCAL_MODEL_DIR / WEIGHT_FILE

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Final model weights not found:\n"
            f"{weight_path}"
        )

    print(
        f"Local weights : {weight_path}"
    )

    print(
        f"HF repository : {HF_REPO_ID}"
    )

    # ========================================================
    # 2. Print file information
    # ========================================================

    file_size_gb = (
        weight_path.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"Weight size   : "
        f"{file_size_gb:.2f} GB"
    )

    # ========================================================
    # 3. Connect to Hugging Face
    # ========================================================

    print()
    print("Connecting to Hugging Face...")

    api = HfApi()

    # ========================================================
    # 4. Ensure repository exists
    # ========================================================

    print()
    print("Ensuring repository exists...")

    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type=REPO_TYPE,
        exist_ok=True,
    )

    print(
        "Repository ready."
    )

    # ========================================================
    # 5. Upload final weights
    # ========================================================

    print()
    print("=" * 80)
    print("UPLOADING MODEL.SAFETENSORS")
    print("=" * 80)

    api.upload_file(
        path_or_fileobj=str(weight_path),
        path_in_repo=WEIGHT_FILE,
        repo_id=HF_REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update final trained ART model weights",
    )

    print()
    print(
        "✓ model.safetensors uploaded successfully."
    )

    # ========================================================
    # 6. Done
    # ========================================================

    print()
    print("=" * 80)
    print("UPLOAD COMPLETE")
    print("=" * 80)

    print()
    print(
        "Repository:"
    )

    print(
        f"https://huggingface.co/{HF_REPO_ID}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    upload_model_to_hf()