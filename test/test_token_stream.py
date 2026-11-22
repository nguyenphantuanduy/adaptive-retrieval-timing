from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TEXTS = [
    "The capital of France is Paris.",
    "The retrieval mechanism is adaptive.",
    "Internationalization and characterization are important.",
    "unbelievable, misunderstanding, automatically, representation.",
    "autoregressive, tokenization, Transformer, reinforcement.",
    "The value is 3.14159265358979.",
    "I spoke with Dr. Smith yesterday.",
    "Mr. John Smith works at the U.S. office.",
    "Really?! That was unexpected...",
    "This is a state-of-the-art retrieval system.",
    "The function is called get_completed_sentences().",
    "Visit https://example.com or email test@example.com.",
    "Use StreamSegmenter and get_completed_sentences().",
    "Vietnamese text: Thời điểm truy hồi thích ứng.",
    "Qwen2.5-3B-Instruct uses a tokenizer with a vocabulary of many subword tokens.",
]


def analyze(text, tokenizer):
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    pieces = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in token_ids
    ]

    reconstructed = "".join(pieces)

    print("=" * 100)
    print(f"TEXT: {text}")
    print("-" * 100)

    for i, (token_id, piece) in enumerate(zip(token_ids, pieces)):
        print(f"{i:3d} | {token_id:6d} | {piece!r}")

    print("-" * 100)
    print(f"Tokens       : {len(token_ids)}")
    print(f"Reconstructed: {reconstructed!r}")
    print(f"Match        : {reconstructed == text}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for text in TEXTS:
        analyze(text, tokenizer)


if __name__ == "__main__":
    main()