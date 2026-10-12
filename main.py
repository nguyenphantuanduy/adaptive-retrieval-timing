import os
import tempfile
from pathlib import Path
from getpass import getpass

# ============================================================
# 1. Chuyển cache và thư mục tạm sang ổ D
# Phải thực hiện trước khi import huggingface_hub
# ============================================================

HF_HOME = Path(r"D:\HF_HOME")
HF_HUB_CACHE = HF_HOME / "hub"
HF_XET_CACHE = HF_HOME / "xet"
HF_ASSETS_CACHE = HF_HOME / "assets"
HF_TEMP = Path(r"D:\HF_TEMP")

for folder in (
    HF_HOME,
    HF_HUB_CACHE,
    HF_XET_CACHE,
    HF_ASSETS_CACHE,
    HF_TEMP,
):
    folder.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_XET_CACHE"] = str(HF_XET_CACHE)
os.environ["HF_ASSETS_CACHE"] = str(HF_ASSETS_CACHE)

os.environ["TEMP"] = str(HF_TEMP)
os.environ["TMP"] = str(HF_TEMP)
tempfile.tempdir = str(HF_TEMP)

# ============================================================
# 2. Download đúng một file vào thư mục chứa main.py
# ============================================================

from huggingface_hub import hf_hub_download

REPO_ID = "adaptive-retrieval-timing/ART-REVIVE"
FILENAME = "model.safetensors"

SAVE_DIR = Path(__file__).resolve().parent
DESTINATION = SAVE_DIR / FILENAME

HF_TOKEN = getpass("Nhap Hugging Face access token: ")

print(f"Destination: {DESTINATION}")
print(f"HF cache:    {HF_HUB_CACHE}")
print(f"Xet cache:   {HF_XET_CACHE}")
print(f"Temp:        {HF_TEMP}")

path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    revision="main",
    token=HF_TOKEN,
    local_dir=str(SAVE_DIR),
)

print(f"\nDownload completed: {path}")
print(f"File exists: {Path(path).is_file()}")
print(f"File size: {Path(path).stat().st_size / (1024**3):.3f} GB")
