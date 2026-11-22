from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    question = "Who founded Microsoft?"

    retrieved_passages = """Passage 1:
Microsoft was founded by Bill Gates and Paul Allen in 1975.

Passage 2:
Bill Gates and Paul Allen established Microsoft, originally named Micro-Soft,
in Albuquerque, New Mexico."""

    messages = [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "tool",
            "content": retrieved_passages,
        },
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("=" * 80)
    print("FORMATTED CHAT TEMPLATE")
    print("=" * 80)
    print(formatted_text)
    print("=" * 80)


if __name__ == "__main__":
    main()