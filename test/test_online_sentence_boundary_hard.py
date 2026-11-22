from transformers import AutoTokenizer
from sentencesplit import StreamSegmenter


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TEXT = (
    "I spoke with Dr. Smith yesterday. "
    "The value is 3.14159265358979, which is approximately pi. "
    "Mr. John Smith works at the U.S. office. "
    "The U.S. economy is large, and the U.S. is an important abbreviation. "
    "Really?! That was unexpected... "
    "This is a state-of-the-art retrieval system. "
    "The function is called get_completed_sentences(). "
    "For example, version 3.5 is newer than version 3. "
    "The model is called Qwen2.5-3B-Instruct. "
    "Visit https://example.com or email test@example.com. "
    "The researcher said, \"This method works well.\" Then the experiment continued. "
    "The abbreviation e.g. should not create a sentence boundary. "
    "The abbreviation i.e. should also not create a sentence boundary. "
    "Initials such as J. K. Rowling should not be split incorrectly. "
    "The model achieved 95.7% accuracy in the final experiment. "
    "The result was surprising! However, the experiment was repeated. "
    "Is this really necessary? Yes, it is. "
    "Wait... What happened next? Nobody knew."
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---------------------------------------------------------
    # Tokenize complete text
    # ---------------------------------------------------------
    token_ids = tokenizer(
        TEXT,
        add_special_tokens=False,
    )["input_ids"]

    # ---------------------------------------------------------
    # Streaming sentence segmenter
    # ---------------------------------------------------------
    segmenter = StreamSegmenter(language="en")

    completed_sentences = []

    print("=" * 100)
    print("HARD TEST: ONLINE SENTENCE BOUNDARY DETECTION")
    print("=" * 100)

    print("\nORIGINAL TEXT:")
    print("-" * 100)
    print(TEXT)

    print("\n" + "=" * 100)
    print("TOKEN STREAM")
    print("=" * 100)

    # ---------------------------------------------------------
    # Feed one token at a time
    # ---------------------------------------------------------
    for i, token_id in enumerate(token_ids):

        # Decode exactly one token
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        # Feed token piece into streaming SBD
        segmenter.feed(piece)

        # Check for completed sentences
        sentences = segmenter.get_completed_sentences()

        if sentences:
            for sentence in sentences:
                completed_sentences.append(sentence)

                print(
                    f"\n"
                    f"Token {i:3d} | "
                    f"id={token_id:6d} | "
                    f"piece={piece!r}"
                )

                print("  >>> BOUNDARY DETECTED")
                print(f"      {sentence!r}")

    # ---------------------------------------------------------
    # Flush remaining buffer
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("FLUSH")
    print("=" * 100)

    remaining = segmenter.flush()

    if remaining:
        for sentence in remaining:
            completed_sentences.append(sentence)
            print(f"  >>> {sentence!r}")
    else:
        print("  No remaining text.")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"Total tokens    : {len(token_ids)}")
    print(f"Sentences found : {len(completed_sentences)}")

    print("\nDetected sentences:")
    print("-" * 100)

    for i, sentence in enumerate(completed_sentences, start=1):
        print(f"{i:2d}. {sentence!r}")


if __name__ == "__main__":
    main()