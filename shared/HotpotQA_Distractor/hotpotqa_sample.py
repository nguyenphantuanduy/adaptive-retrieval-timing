from dataclasses import dataclass
from typing import List, Dict, Any


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Document:
    title: str
    content: str


@dataclass
class HotpotQASample:
    question: str
    answer: str
    golden_document: List[Document]
    distract_document: List[Document]


# ============================================================
# HOTPOTQA CONVERTER
# ============================================================

class HotpotQASampleConverter:
    """
    Convert one raw HotpotQA Distractor sample into a structured
    HotpotQASample.

    The raw HotpotQA sample is expected to have:

        question
        answer
        supporting_facts
        context

    Golden documents:
        Documents whose titles appear in supporting_facts["title"].

    Distractor documents:
        All remaining documents.

    Supporting sentences inside golden documents are explicitly
    marked with:

        [SUPPORTING] ... [/SUPPORTING]
    """

    SUPPORTING_START = "[SUPPORTING]"
    SUPPORTING_END = "[/SUPPORTING]"

    @classmethod
    def from_hotpotqa(
        cls,
        sample: Dict[str, Any],
    ) -> HotpotQASample:

        # ====================================================
        # 1. BASIC FIELDS
        # ====================================================

        question = sample["question"]
        answer = sample["answer"]

        context = sample["context"]

        titles = context["title"]
        sentences = context["sentences"]

        supporting_facts = sample["supporting_facts"]

        supporting_titles = (
            supporting_facts["title"]
        )

        supporting_sent_ids = (
            supporting_facts["sent_id"]
        )

        # ====================================================
        # 2. BUILD SUPPORTING-SENTENCE MAP
        #
        # Example:
        #
        # {
        #     "Scott Derrickson": {0},
        #     "Ed Wood": {0}
        # }
        # ====================================================

        supporting_sentence_map = {}

        for title, sent_id in zip(
            supporting_titles,
            supporting_sent_ids,
        ):

            if title not in supporting_sentence_map:
                supporting_sentence_map[title] = set()

            supporting_sentence_map[title].add(
                sent_id
            )

        # ====================================================
        # 3. CONVERT DOCUMENTS
        # ====================================================

        golden_document = []
        distract_document = []

        for title, document_sentences in zip(
            titles,
            sentences,
        ):

            # ------------------------------------------------
            # Determine whether this document is useful.
            # ------------------------------------------------

            is_golden = (
                title in supporting_sentence_map
            )

            # ------------------------------------------------
            # Build document content.
            # ------------------------------------------------

            processed_sentences = []

            for sent_id, sentence in enumerate(
                document_sentences
            ):

                # Supporting sentence.
                if (
                    is_golden
                    and sent_id
                    in supporting_sentence_map[title]
                ):

                    sentence = (
                        f"{cls.SUPPORTING_START} "
                        f"{sentence.strip()} "
                        f"{cls.SUPPORTING_END}"
                    )

                else:
                    sentence = sentence.strip()

                processed_sentences.append(
                    sentence
                )

            content = " ".join(
                processed_sentences
            )

            document = Document(
                title=title,
                content=content,
            )

            # ------------------------------------------------
            # Separate golden / distractor documents.
            # ------------------------------------------------

            if is_golden:
                golden_document.append(
                    document
                )
            else:
                distract_document.append(
                    document
                )

        # ====================================================
        # 4. RETURN STRUCTURED SAMPLE
        # ====================================================

        return HotpotQASample(
            question=question,
            answer=answer,
            golden_document=golden_document,
            distract_document=distract_document,
        )


# ============================================================
# TEST
# ============================================================

def test_hotpotqa_sample_converter():

    from datasets import load_dataset

    print("=" * 80)
    print("TEST — HOTPOTQA SAMPLE CONVERTER")
    print("=" * 80)

    # --------------------------------------------------------
    # Load validation sample
    # --------------------------------------------------------

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        streaming=True,
    )

    raw_sample = next(iter(dataset))

    print()
    print("✓ HotpotQA sample loaded")

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    converted = (
        HotpotQASampleConverter.from_hotpotqa(
            raw_sample
        )
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    assert isinstance(
        converted.question,
        str,
    )

    assert isinstance(
        converted.answer,
        str,
    )

    assert isinstance(
        converted.golden_document,
        list,
    )

    assert isinstance(
        converted.distract_document,
        list,
    )

    assert (
        len(converted.golden_document) > 0
    )

    assert (
        len(converted.distract_document) > 0
    )

    # --------------------------------------------------------
    # Print question / answer
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(converted.question)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(converted.answer)

    # --------------------------------------------------------
    # Print golden documents
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"GOLDEN DOCUMENTS "
        f"({len(converted.golden_document)})"
    )
    print("=" * 80)

    for i, document in enumerate(
        converted.golden_document,
        start=1,
    ):

        print()
        print(
            f"--- GOLDEN DOCUMENT {i} ---"
        )

        print(
            f"Title: {document.title}"
        )

        print(
            document.content
        )

        # Supporting marker must exist.
        assert (
            HotpotQASampleConverter.SUPPORTING_START
            in document.content
        )

        assert (
            HotpotQASampleConverter.SUPPORTING_END
            in document.content
        )

    # --------------------------------------------------------
    # Print distractor documents
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"DISTRACTOR DOCUMENTS "
        f"({len(converted.distract_document)})"
    )
    print("=" * 80)

    for i, document in enumerate(
        converted.distract_document,
        start=1,
    ):

        print()
        print(
            f"--- DISTRACTOR DOCUMENT {i} ---"
        )

        print(
            f"Title: {document.title}"
        )

        print(
            document.content
        )

        # Distractors must NOT contain
        # supporting markers.
        assert (
            HotpotQASampleConverter.SUPPORTING_START
            not in document.content
        )

        assert (
            HotpotQASampleConverter.SUPPORTING_END
            not in document.content
        )

    # --------------------------------------------------------
    # Document count check
    # --------------------------------------------------------

    total_documents = (
        len(converted.golden_document)
        + len(converted.distract_document)
    )

    assert (
        total_documents
        == len(raw_sample["context"]["title"])
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Question             : "
        f"{converted.question}"
    )

    print(
        f"Answer               : "
        f"{converted.answer}"
    )

    print(
        f"Golden documents     : "
        f"{len(converted.golden_document)}"
    )

    print(
        f"Distractor documents : "
        f"{len(converted.distract_document)}"
    )

    print(
        f"Total documents      : "
        f"{total_documents}"
    )

    print()
    print(
        "✓ Supporting sentences correctly marked"
    )

    print(
        "✓ Golden/distractor documents correctly separated"
    )

    print()
    print(
        "✓ HOTPOTQA SAMPLE CONVERTER TEST PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_hotpotqa_sample_converter()