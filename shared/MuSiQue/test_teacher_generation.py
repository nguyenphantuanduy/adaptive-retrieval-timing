# ============================================================
# TEST MUSIQUE TEACHER GENERATION
# QWEN3-4B
# SINGLE SAMPLE
# ============================================================

import gc
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .load_dataset import load_musique_ans
from .musique_sample import MuSiQueSampleConverter
from .musique_teacher_prompt import create_teacher_prompt


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-4B"

MAX_NEW_TOKENS = 1024

TEMPERATURE = 0.6

TOP_P = 0.95

# ------------------------------------------------------------
# Which sample to test
# ------------------------------------------------------------

SAMPLE_INDEX = 0


# ============================================================
# LOAD TEACHER MODEL
# ============================================================

def load_teacher_model():

    print("=" * 80)
    print("LOADING TEACHER MODEL")
    print("=" * 80)

    print()
    print(
        f"Model: {MODEL_NAME}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ========================================================
    # MODEL
    # ========================================================

    model_kwargs = {
        "device_map": "auto",
    }

    if torch.cuda.is_available():

        if torch.cuda.is_bf16_supported():

            model_kwargs[
                "torch_dtype"
            ] = torch.bfloat16

        else:

            model_kwargs[
                "torch_dtype"
            ] = torch.float16

    else:

        model_kwargs[
            "torch_dtype"
        ] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    model.eval()

    print()
    print(
        "✓ Teacher model loaded"
    )

    print(
        f"  Dtype: {model.dtype}"
    )

    print(
        f"  Device map: auto"
    )

    if torch.cuda.is_available():

        print()
        print(
            f"CUDA devices: "
            f"{torch.cuda.device_count()}"
        )

        for i in range(
            torch.cuda.device_count()
        ):

            print(
                f"  GPU {i}: "
                f"{torch.cuda.get_device_name(i)}"
            )

    return model, tokenizer


# ============================================================
# MAIN TEST
# ============================================================

def test_single_sample():

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    model, tokenizer = (
        load_teacher_model()
    )

    # ========================================================
    # 2. LOAD MUSIQUE
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING MUSIQUE")
    print("=" * 80)

    dataset = load_musique_ans()

    dev_dataset = dataset[
        "dev"
    ]

    print()
    print(
        f"✓ Dev dataset loaded: "
        f"{len(dev_dataset):,} samples"
    )

    # ========================================================
    # 3. SELECT ONE SAMPLE
    # ========================================================

    if SAMPLE_INDEX >= len(
        dev_dataset
    ):

        raise IndexError(
            f"SAMPLE_INDEX={SAMPLE_INDEX} "
            f"is out of range. "
            f"Dataset size: {len(dev_dataset)}"
        )

    raw_sample = dev_dataset[
        SAMPLE_INDEX
    ]

    # ========================================================
    # 4. CONVERT SAMPLE
    # ========================================================

    sample = (
        MuSiQueSampleConverter.from_musique(
            raw_sample
        )
    )

    # ========================================================
    # 5. PRINT SAMPLE INFORMATION
    # ========================================================

    print()
    print("=" * 80)
    print("SELECTED SAMPLE")
    print("=" * 80)

    print()
    print(
        f"Sample index: "
        f"{SAMPLE_INDEX}"
    )

    print()
    print(
        "Question:"
    )

    print(
        sample.question
    )

    print()
    print(
        "Target answer:"
    )

    print(
        sample.answer
    )

    print()
    print(
        "Number of golden documents:"
    )

    print(
        len(
            sample.golden_document
        )
    )

    print()
    print(
        "Number of distract documents:"
    )

    print(
        len(
            sample.distract_document
        )
    )

    print()
    print(
        "Number of reasoning steps:"
    )

    print(
        len(
            sample.question_decomposition
        )
    )

    # ========================================================
    # 6. CREATE TEACHER PROMPT
    # ========================================================

    teacher_prompt = (
        create_teacher_prompt(
            sample=sample
        )
    )

    # ========================================================
    # 7. PRINT SYSTEM PROMPT
    # ========================================================

    print()
    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)

    print()

    print(
        teacher_prompt[
            "system"
        ]
    )

    # ========================================================
    # 8. PRINT USER PROMPT
    # ========================================================

    print()
    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)

    print()

    print(
        teacher_prompt[
            "user"
        ]
    )

    # ========================================================
    # 9. BUILD CHAT MESSAGES
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": teacher_prompt[
                "system"
            ],
        },
        {
            "role": "user",
            "content": teacher_prompt[
                "user"
            ],
        },
    ]

    # ========================================================
    # 10. APPLY CHAT TEMPLATE
    # ========================================================

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # ========================================================
    # 11. PRINT FINAL MODEL INPUT
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL CHAT-TEMPLATE INPUT")
    print("=" * 80)

    print()

    print(text)

    # ========================================================
    # 12. TOKENIZE
    # ========================================================

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
    )

    model_device = model.device

    inputs = {
        key: value.to(
            model_device
        )
        for key, value
        in inputs.items()
    }

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    print()
    print("=" * 80)
    print("TOKENIZATION")
    print("=" * 80)

    print()
    print(
        f"Input tokens: "
        f"{input_length:,}"
    )

    # ========================================================
    # 13. GENERATE
    # ========================================================

    print()
    print("=" * 80)
    print("GENERATING TEACHER OUTPUT")
    print("=" * 80)

    print()

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=True,

            temperature=TEMPERATURE,

            top_p=TOP_P,

            pad_token_id=(
                tokenizer.eos_token_id
            ),
        )

    # ========================================================
    # 14. EXTRACT GENERATED TOKENS
    # ========================================================

    generated_tokens = (
        outputs[
            0,
            input_length:
        ]
    )

    # ========================================================
    # 15. REMOVE EOS
    # ========================================================

    eos_positions = (
        generated_tokens
        == tokenizer.eos_token_id
    )

    eos_indices = torch.nonzero(
        eos_positions,
        as_tuple=False,
    )

    if eos_indices.numel() > 0:

        eos_index = int(
            eos_indices[
                0
            ].item()
        )

        generated_tokens = (
            generated_tokens[
                :eos_index
            ]
        )

    # ========================================================
    # 16. DECODE
    # ========================================================

    teacher_output = (
        tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
    )

    # ========================================================
    # 17. PRINT RESULT
    # ========================================================

    print()
    print("=" * 80)
    print("TEACHER GENERATION OUTPUT")
    print("=" * 80)

    print()

    print(
        teacher_output
    )

    print()
    print("=" * 80)
    print("GENERATION STATISTICS")
    print("=" * 80)

    print()

    print(
        f"Generated tokens: "
        f"{generated_tokens.shape[0]:,}"
    )

    print(
        f"Max new tokens: "
        f"{MAX_NEW_TOKENS:,}"
    )

    print()
    print("=" * 80)
    print("✓ SINGLE-SAMPLE TEST COMPLETED")
    print("=" * 80)

    # ========================================================
    # 18. CLEANUP
    # ========================================================

    del outputs
    del inputs
    del generated_tokens
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():

        for device_index in range(
            torch.cuda.device_count()
        ):

            with torch.cuda.device(
                device_index
            ):

                torch.cuda.empty_cache()

                torch.cuda.ipc_collect()


# ============================================================
# OFFICIAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_single_sample()