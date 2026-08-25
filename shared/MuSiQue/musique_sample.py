# ============================================================
# MuSiQue Sample Converter
# ============================================================

from dataclasses import dataclass
from typing import List, Dict, Any


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Document:
    """
    Structured representation of one MuSiQue paragraph.
    """

    title: str
    content: str
    paragraph_idx: int


@dataclass
class QuestionDecompositionStep:
    """
    One reasoning step from MuSiQue question decomposition.

    paragraph_support_idx identifies the paragraph containing
    the evidence for this reasoning step.
    """

    step_id: int
    question: str
    answer: str
    paragraph_support_idx: int


@dataclass
class MuSiQueSample:
    """
    Structured representation of one MuSiQue-Ans sample.
    """

    question: str
    answer: str

    golden_document: List[Document]
    distract_document: List[Document]

    question_decomposition: List[
        QuestionDecompositionStep
    ]

    reasoning_instruction: str


# ============================================================
# MUSIQUE CONVERTER
# ============================================================

class MuSiQueSampleConverter:
    """
    Convert one raw MuSiQue-Ans sample into a structured
    MuSiQueSample.

    Expected raw fields:

        question
        answer
        paragraphs
        question_decomposition

    Each paragraph has:

        idx
        title
        paragraph_text
        is_supporting

    Golden documents:
        paragraphs whose is_supporting == True

    Distractor documents:
        paragraphs whose is_supporting == False

    The question decomposition is converted into structured
    reasoning steps.

    A teacher-oriented reasoning instruction is generated from
    the decomposition.

    The reasoning instruction contains:

        - reasoning step
        - sub-question
        - intermediate answer
        - exact supporting document title
        - final answer

    Paragraph indices are retained internally for dataset
    mapping and validation, but are NOT exposed in the
    teacher-facing reasoning instruction.
    """

    # ========================================================
    # MAIN CONVERTER
    # ========================================================

    @classmethod
    def from_musique(
        cls,
        sample: Dict[str, Any],
    ) -> MuSiQueSample:

        # ====================================================
        # 1. BASIC FIELDS
        # ====================================================

        question = sample["question"]

        answer = sample["answer"]

        paragraphs = sample["paragraphs"]

        raw_decomposition = (
            sample.get(
                "question_decomposition",
                [],
            )
        )

        # ====================================================
        # 2. QUESTION DECOMPOSITION
        # ====================================================

        question_decomposition = []

        for step_idx, step in enumerate(
            raw_decomposition,
            start=1,
        ):

            decomposition_step = (
                QuestionDecompositionStep(
                    step_id=step_idx,

                    question=step["question"],

                    answer=step["answer"],

                    paragraph_support_idx=(
                        step["paragraph_support_idx"]
                    ),
                )
            )

            question_decomposition.append(
                decomposition_step
            )

        # ====================================================
        # 3. BUILD GOLDEN / DISTRACTOR DOCUMENTS
        # ====================================================

        golden_document = []

        distract_document = []

        for paragraph in paragraphs:

            paragraph_idx = paragraph["idx"]

            title = paragraph["title"]

            content = (
                paragraph["paragraph_text"]
                .strip()
            )

            is_supporting = (
                paragraph["is_supporting"]
            )

            # ------------------------------------------------
            # Create document.
            # ------------------------------------------------

            document = Document(
                title=title,
                content=content,
                paragraph_idx=paragraph_idx,
            )

            # ------------------------------------------------
            # Separate golden / distractor.
            # ------------------------------------------------

            if is_supporting:

                golden_document.append(
                    document
                )

            else:

                distract_document.append(
                    document
                )

        # ====================================================
        # 4. BUILD TEACHER REASONING INSTRUCTION
        # ====================================================

        reasoning_instruction = (
            cls._build_reasoning_instruction(
                question=question,
                decomposition=question_decomposition,
                golden_documents=golden_document,
                answer=answer,
            )
        )

        # ====================================================
        # 5. RETURN STRUCTURED SAMPLE
        # ====================================================

        return MuSiQueSample(

            question=question,

            answer=answer,

            golden_document=golden_document,

            distract_document=distract_document,

            question_decomposition=(
                question_decomposition
            ),

            reasoning_instruction=(
                reasoning_instruction
            ),
        )

    # ========================================================
    # BUILD REASONING INSTRUCTION
    # ========================================================

    @classmethod
    def _build_reasoning_instruction(
        cls,
        question: str,
        decomposition: List[
            QuestionDecompositionStep
        ],
        golden_documents: List[Document],
        answer: str,
    ) -> str:
        """
        Build the complete teacher-oriented reasoning scaffold.

        Each reasoning step contains:

            - sub-question
            - intermediate answer
            - exact supporting document title

        The final answer is also explicitly provided.

        Paragraph indices are intentionally excluded because
        they are internal dataset metadata and are not useful
        for teacher reasoning generation.

        The resulting instruction is designed to replace the
        separate question_decomposition section in the teacher
        prompt.
        """

        # ----------------------------------------------------
        # Build paragraph-index -> document map.
        #
        # This mapping is used internally to resolve the exact
        # supporting document title.
        # ----------------------------------------------------

        document_map = {
            document.paragraph_idx: document
            for document in golden_documents
        }

        lines = []

        # ----------------------------------------------------
        # Introduction.
        # ----------------------------------------------------

        lines.append(
            "Use the following reasoning structure to solve "
            "the main question."
        )

        lines.append(
            "The information below is provided as teacher "
            "supervision."
        )

        lines.append(
            "Reconstruct a natural reasoning process from it "
            "rather than copying it verbatim."
        )

        lines.append("")

        lines.append(
            f"Main question: {question}"
        )

        lines.append("")

        # ----------------------------------------------------
        # Reasoning steps.
        # ----------------------------------------------------

        if decomposition:

            lines.append(
                "Reasoning steps:"
            )

            lines.append("")

            for step in decomposition:

                # --------------------------------------------
                # Resolve supporting document.
                # --------------------------------------------

                supporting_document = (
                    document_map.get(
                        step.paragraph_support_idx
                    )
                )

                if supporting_document is not None:

                    document_title = (
                        supporting_document.title
                    )

                else:

                    document_title = (
                        "UNKNOWN DOCUMENT"
                    )

                # --------------------------------------------
                # Step.
                # --------------------------------------------

                lines.append(
                    f"Step {step.step_id}:"
                )

                lines.append(
                    f"  Sub-question: "
                    f"{step.question}"
                )

                lines.append(
                    f"  Intermediate answer: "
                    f"{step.answer}"
                )

                lines.append(
                    f"  Supporting document: "
                    f"{document_title}"
                )

                lines.append("")

        else:

            lines.append(
                "No explicit question decomposition "
                "is available."
            )

            lines.append("")

        # ----------------------------------------------------
        # Final answer.
        # ----------------------------------------------------

        lines.append(
            f"Final answer: {answer}"
        )

        return "\n".join(lines)

    # ========================================================
    # HELPER — FIND DOCUMENT
    # ========================================================

    @classmethod
    def find_document_by_paragraph_idx(
        cls,
        sample: MuSiQueSample,
        paragraph_idx: int,
    ) -> Document | None:
        """
        Find a document using its original MuSiQue paragraph
        index.

        This helper is used internally for validation and
        dataset mapping.
        """

        for document in (
            sample.golden_document
            + sample.distract_document
        ):

            if (
                document.paragraph_idx
                == paragraph_idx
            ):

                return document

        return None


# ============================================================
# TEST
# ============================================================

def test_musique_sample_converter():

    print("=" * 80)
    print("TEST — MUSIQUE SAMPLE CONVERTER")
    print("=" * 80)

    # --------------------------------------------------------
    # Load one sample using the project's dataset utility.
    # --------------------------------------------------------

    from .load_dataset import load_musique_ans

    dataset = load_musique_ans()

    # --------------------------------------------------------
    # Use first DEV sample.
    # --------------------------------------------------------

    raw_sample = dataset["dev"][0]

    print()
    print("✓ MuSiQue sample loaded")

    # --------------------------------------------------------
    # Convert.
    # --------------------------------------------------------

    converted = (
        MuSiQueSampleConverter.from_musique(
            raw_sample
        )
    )

    # ========================================================
    # BASIC CHECKS
    # ========================================================

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

    assert isinstance(
        converted.question_decomposition,
        list,
    )

    assert isinstance(
        converted.reasoning_instruction,
        str,
    )

    assert (
        len(converted.golden_document) > 0
    )

    assert (
        len(converted.distract_document) > 0
    )

    assert (
        len(converted.question_decomposition)
        > 0
    )

    # ========================================================
    # QUESTION
    # ========================================================

    print()
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(
        converted.question
    )

    # ========================================================
    # ANSWER
    # ========================================================

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(
        converted.answer
    )

    # ========================================================
    # QUESTION DECOMPOSITION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "QUESTION DECOMPOSITION "
        f"({len(converted.question_decomposition)})"
    )
    print("=" * 80)

    for step in (
        converted.question_decomposition
    ):

        print()

        print(
            f"--- STEP {step.step_id} ---"
        )

        print(
            "Sub-question:"
        )

        print(
            f"  {step.question}"
        )

        print(
            "Intermediate answer:"
        )

        print(
            f"  {step.answer}"
        )

        print(
            "Supporting paragraph index:"
        )

        print(
            f"  {step.paragraph_support_idx}"
        )

    # ========================================================
    # GOLDEN DOCUMENTS
    # ========================================================

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
            f"Paragraph index: "
            f"{document.paragraph_idx}"
        )

        print(
            f"Title: "
            f"{document.title}"
        )

        print(
            document.content
        )

    # ========================================================
    # DISTRACTOR DOCUMENTS
    # ========================================================

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
            f"Paragraph index: "
            f"{document.paragraph_idx}"
        )

        print(
            f"Title: "
            f"{document.title}"
        )

        print(
            document.content
        )

    # ========================================================
    # REASONING INSTRUCTION
    # ========================================================

    print()
    print("=" * 80)
    print("TEACHER REASONING INSTRUCTION")
    print("=" * 80)

    print(
        converted.reasoning_instruction
    )

    # ========================================================
    # REASONING INSTRUCTION VALIDATION
    # ========================================================

    # --------------------------------------------------------
    # Main question must exist.
    # --------------------------------------------------------

    assert (
        converted.question
        in converted.reasoning_instruction
    )

    # --------------------------------------------------------
    # Final answer must exist.
    # --------------------------------------------------------

    assert (
        converted.answer
        in converted.reasoning_instruction
    )

    # --------------------------------------------------------
    # Every decomposition step must appear.
    # --------------------------------------------------------

    for step in (
        converted.question_decomposition
    ):

        assert (
            step.question
            in converted.reasoning_instruction
        )

        assert (
            step.answer
            in converted.reasoning_instruction
        )

        # ----------------------------------------------------
        # Resolve supporting document.
        # ----------------------------------------------------

        document = (
            MuSiQueSampleConverter
            .find_document_by_paragraph_idx(
                converted,
                step.paragraph_support_idx,
            )
        )

        assert document is not None

        assert (
            document.title
            in converted.reasoning_instruction
        )

        # ----------------------------------------------------
        # Paragraph index MUST NOT appear as a field in the
        # teacher-facing reasoning instruction.
        #
        # We specifically check that the old field format is
        # gone.
        # ----------------------------------------------------

        assert (
            "Supporting paragraph index:"
            not in converted.reasoning_instruction
        )

    # ========================================================
    # DOCUMENT COUNT CHECK
    # ========================================================

    total_documents = (
        len(converted.golden_document)
        + len(converted.distract_document)
    )

    assert (
        total_documents
        == len(raw_sample["paragraphs"])
    )

    # ========================================================
    # GOLDEN DOCUMENT COUNT CHECK
    # ========================================================

    expected_golden = sum(
        1
        for paragraph in raw_sample["paragraphs"]
        if paragraph["is_supporting"]
    )

    assert (
        len(converted.golden_document)
        == expected_golden
    )

    # ========================================================
    # DISTRACTOR DOCUMENT COUNT CHECK
    # ========================================================

    expected_distractor = sum(
        1
        for paragraph in raw_sample["paragraphs"]
        if not paragraph["is_supporting"]
    )

    assert (
        len(converted.distract_document)
        == expected_distractor
    )

    # ========================================================
    # PARAGRAPH INDEX CHECK
    # ========================================================

    raw_indices = {
        paragraph["idx"]
        for paragraph in raw_sample["paragraphs"]
    }

    converted_indices = {
        document.paragraph_idx
        for document in (
            converted.golden_document
            + converted.distract_document
        )
    }

    assert (
        raw_indices
        == converted_indices
    )

    # ========================================================
    # REASONING DOCUMENT MAPPING CHECK
    # ========================================================

    for step in (
        converted.question_decomposition
    ):

        supporting_document = (
            MuSiQueSampleConverter
            .find_document_by_paragraph_idx(
                converted,
                step.paragraph_support_idx,
            )
        )

        assert (
            supporting_document is not None
        )

        # Supporting paragraph must actually be golden.

        assert (
            supporting_document
            in converted.golden_document
        )

    # ========================================================
    # SUMMARY
    # ========================================================

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

    print(
        f"Reasoning steps      : "
        f"{len(converted.question_decomposition)}"
    )

    print()
    print(
        "✓ Question decomposition correctly parsed"
    )

    print(
        "✓ Paragraph indices correctly preserved internally"
    )

    print(
        "✓ Golden/distractor documents correctly separated"
    )

    print(
        "✓ Reasoning instruction contains document titles"
    )

    print(
        "✓ Reasoning instruction contains intermediate answers"
    )

    print(
        "✓ Reasoning instruction excludes paragraph indices"
    )

    print(
        "✓ Reasoning step → document mapping verified"
    )

    print()
    print(
        "✓ MUSIQUE SAMPLE CONVERTER TEST PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_musique_sample_converter()