from transformers import AutoTokenizer
from sentencesplit import StreamSegmenter


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TEXT = (
    "The capital of France is Paris. "
    "Paris is a beautiful city with a long history. "
    "It is also an important cultural and economic center in Europe. "
    "The retrieval mechanism is adaptive and decides when additional information is needed. "
    "For example, a language model may generate several tokens before deciding to retrieve new documents. "
    "This process should ideally avoid unnecessary retrieval while still providing useful evidence. "
    "In our experiment, the model generates text continuously and the sentence boundary detector runs online. "
    "When a sentence is completed, the ART module can make a retrieval decision. "
    "The goal is to learn when retrieval is useful rather than retrieving after every fixed number of tokens."
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---------------------------------------------------------
    # 1. Tokenize the complete text
    # ---------------------------------------------------------
    token_ids = tokenizer(
        TEXT,
        add_special_tokens=False,
    )["input_ids"]

    print("=" * 100)
    print("ORIGINAL TEXT")
    print("=" * 100)
    print(TEXT)

    print("\n" + "=" * 100)
    print("TOKEN STREAM + STREAM SEGMENTER")
    print("=" * 100)

    # ---------------------------------------------------------
    # 2. Create streaming sentence segmenter
    # ---------------------------------------------------------
    segmenter = StreamSegmenter(language="en")

    # ---------------------------------------------------------
    # 3. Feed tokens one by one
    # ---------------------------------------------------------
    for i, token_id in enumerate(token_ids):

        # Decode exactly ONE token
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        # Feed this token piece into the streaming segmenter
        segmenter.feed(piece)

        print(f"\nToken {i:3d} | id={token_id:6d} | piece={piece!r}")

        # Check whether one or more complete sentences
        # became available after this token.
        completed = segmenter.get_completed_sentences()

        if completed:
            print("  >>> COMPLETED SENTENCE:")

            for sentence in completed:
                print(f"      {sentence!r}")

    # ---------------------------------------------------------
    # 4. Flush remaining unfinished text
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("FLUSH")
    print("=" * 100)

    remaining = segmenter.flush()

    if remaining:
        for sentence in remaining:
            print(f"  >>> {sentence!r}")
    else:
        print("  No remaining text.")


if __name__ == "__main__":
    main()