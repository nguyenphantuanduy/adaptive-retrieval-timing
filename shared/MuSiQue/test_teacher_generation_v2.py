# ============================================================
# TEST MUSIQUE TEACHER GENERATION (v2)
# QWEN3-8B
# SINGLE SAMPLE
# ============================================================

import gc
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.MuSiQue.load_dataset import load_musique_ans
    from shared.MuSiQue.musique_sample import MuSiQueSampleConverter
    from shared.MuSiQue.musique_teacher_prompt_v2 import (
        DEFAULT_DEMONSTRATION_FILE,
        create_teacher_prompt,
        create_teacher_prompts,
        load_teacher_demonstrations,
    )
except ImportError:
    from .load_dataset import load_musique_ans
    from .musique_sample import MuSiQueSampleConverter
    from .musique_teacher_prompt_v2 import (
        DEFAULT_DEMONSTRATION_FILE,
        create_teacher_prompt,
        create_teacher_prompts,
        load_teacher_demonstrations,
    )


# ============================================================
# CONFIG (Qwen3-8B Non-Thinking Mode Recommended)
# ============================================================

MODEL_NAME = "Qwen/Qwen3-8B"

MAX_NEW_TOKENS = 2048

TEMPERATURE = 0.7

TOP_P = 0.8

TOP_K = 20

MIN_P = 0.0

REPETITION_PENALTY = 1.0

# ------------------------------------------------------------
# Which sample to test
# ------------------------------------------------------------

SAMPLE_INDEX = 0

CURRENT_FILE_DIR = Path(__file__).resolve().parent
LOCAL_DEMONSTRATION_FILE = (
    CURRENT_FILE_DIR / "teacher_reasoning_demonstrations.json"
)


# ============================================================
# LOAD TEACHER MODEL
# ============================================================

def load_teacher_model():

    print("=" * 80)
    print("LOADING TEACHER MODEL (v2)")
    print("=" * 80)

    print()
    print(f"Model: {MODEL_NAME}")

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # MODEL
    # ========================================================

    model_kwargs = {
        "device_map": "auto",
    }

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    model.eval()

    print()
    print("✓ Teacher model loaded")
    print(f"  Dtype      : {model.dtype}")
    print("  Device map : auto")

    if torch.cuda.is_available():
        print()
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    return model, tokenizer


# ============================================================
# MAIN TEST
# ============================================================

def test_single_sample():

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    model, tokenizer = load_teacher_model()

    # ========================================================
    # 2. LOAD DEMONSTRATIONS FOR PROMPT V2
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING TEACHER DEMONSTRATIONS")
    print("=" * 80)

    demo_file = (
        LOCAL_DEMONSTRATION_FILE
        if LOCAL_DEMONSTRATION_FILE.exists()
        else DEFAULT_DEMONSTRATION_FILE
    )
    demonstrations = load_teacher_demonstrations(demo_file)

    print()
    print(f"✓ Loaded {len(demonstrations)} demonstrations from:")
    print(f"  {demo_file}")

    # ========================================================
    # 3. LOAD MUSIQUE
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING MUSIQUE")
    print("=" * 80)

    dataset = load_musique_ans()
    dev_dataset = dataset["dev"]

    print()
    print(f"✓ Dev dataset loaded: {len(dev_dataset):,} samples")

    # ========================================================
    # 4. SELECT ONE SAMPLE
    # ========================================================

    if SAMPLE_INDEX >= len(dev_dataset):
        raise IndexError(
            f"SAMPLE_INDEX={SAMPLE_INDEX} is out of range. "
            f"Dataset size: {len(dev_dataset)}"
        )

    raw_sample = dev_dataset[SAMPLE_INDEX]

    # ========================================================
    # 5. CONVERT SAMPLE
    # ========================================================

    sample = MuSiQueSampleConverter.from_musique(raw_sample)

    # ========================================================
    # 6. PRINT SAMPLE INFORMATION
    # ========================================================

    print()
    print("=" * 80)
    print("SELECTED SAMPLE")
    print("=" * 80)

    print()
    print(f"Sample index: {SAMPLE_INDEX}")
    print()
    print("Question:")
    print(sample.question)
    print()
    print("Target answer:")
    print(sample.answer)
    print()
    print(f"Number of golden documents   : {len(sample.golden_document)}")
    print(f"Number of distract documents : {len(sample.distract_document)}")
    print(f"Number of reasoning steps    : {len(sample.question_decomposition)}")

    # ========================================================
    # 7. CREATE TEACHER PROMPT (v2)
    # ========================================================

    system_prompt, user_prompt = create_teacher_prompts(
        sample=sample,
        demonstrations=demonstrations,
    )

    prompt_text = create_teacher_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tokenizer=tokenizer,
        enable_thinking=False,
    )

    # ========================================================
    # 8. PRINT TEACHER PROMPT
    # ========================================================

    print()
    print("=" * 80)
    print("SYSTEM PROMPT (v2)")
    print("=" * 80)
    print(system_prompt)

    print()
    print("=" * 80)
    print("FINAL CHAT-TEMPLATE INPUT (v2)")
    print("=" * 80)
    print(prompt_text)

    # ========================================================
    # 9. TOKENIZE
    # ========================================================

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=False,
    )

    model_device = model.device
    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    print()
    print("=" * 80)
    print("TOKENIZATION")
    print("=" * 80)
    print(f"Input tokens: {input_length:,}")

    # ========================================================
    # 10. GENERATE
    # ========================================================

    print()
    print("=" * 80)
    print("GENERATING TEACHER OUTPUT (v2)")
    print("=" * 80)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            min_p=MIN_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    # ========================================================
    # 11. EXTRACT GENERATED TOKENS
    # ========================================================

    generated_tokens = outputs[0, input_length:]

    # Remove EOS
    eos_positions = generated_tokens == tokenizer.eos_token_id
    eos_indices = torch.nonzero(eos_positions, as_tuple=False)

    if eos_indices.numel() > 0:
        eos_index = int(eos_indices[0].item())
        generated_tokens = generated_tokens[:eos_index]

    # Decode
    teacher_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    # ========================================================
    # 12. PRINT RESULT
    # ========================================================

    print()
    print("=" * 80)
    print("TEACHER GENERATION OUTPUT (v2)")
    print("=" * 80)
    print()
    print(teacher_output)

    print()
    print("=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)
    print(f"Generated tokens : {generated_tokens.shape[0]:,}")
    print(f"Max new tokens   : {MAX_NEW_TOKENS:,}")

    print()
    print("=" * 80)
    print("✓ SINGLE-SAMPLE TEST (v2) COMPLETED")
    print("=" * 80)

    # ========================================================
    # 13. CLEANUP
    # ========================================================

    del outputs
    del inputs
    del generated_tokens
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            with torch.cuda.device(device_index):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass


# ============================================================
# OFFICIAL ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_single_sample()
