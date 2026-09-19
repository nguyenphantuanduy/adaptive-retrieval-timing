from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
FILENAME = "config.json"


def main():
    # Download only config.json from Hugging Face.
    downloaded_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=FILENAME,
    )

    # Save a copy next to this main.py.
    output_path = Path(__file__).resolve().parent / FILENAME
    output_path.write_bytes(Path(downloaded_path).read_bytes())

    print(f"Downloaded: {MODEL_ID}/{FILENAME}")
    print(f"Saved to:   {output_path}")


if __name__ == "__main__":
    main()
