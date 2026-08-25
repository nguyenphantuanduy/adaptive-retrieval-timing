from hotpotqa_sample import (
    HotpotQASample,
    Document,
    HotpotQASampleConverter,
)


# ============================================================
# TEACHER PROMPT BUILDER
# ============================================================

def create_teacher_prompt(
    sample: HotpotQASample,
) -> str:
    """
    Create an instruction prompt for Qwen3-8B teacher generation.

    The teacher receives:
        - question
        - target answer
        - golden documents only

    Golden documents contain explicitly marked supporting
    sentences.

    The teacher is instructed to reconstruct a natural reasoning
    process that starts from the question, identifies useful
    evidence, reasons over the relevant documents, and reaches
    the given answer.

    Document references MUST use the exact document title.
    """

    # ========================================================
    # 1. BUILD GOLDEN DOCUMENT SECTION
    # ========================================================

    document_sections = []

    for document in sample.golden_document:

        document_sections.append(
            f"[GOLDEN DOCUMENT]\n"
            f"Title: {document.title}\n"
            f"{document.content}\n"
            f"[END DOCUMENT]"
        )

    documents_text = "\n\n".join(
        document_sections
    )

    # ========================================================
    # 2. BUILD PROMPT
    # ========================================================

    prompt = f"""
You are a reasoning teacher generating high-quality reasoning
data for a retrieval-augmented language model.

Your task is to solve the question below by reasoning over the
provided documents.

IMPORTANT:
- You are given the final answer only as a target.
- You must reconstruct the reasoning that naturally leads from
  the question to that answer.
- Do NOT simply explain the answer directly.
- Start from the question and determine what information is
  needed to solve it.
- Inspect the provided documents and identify the specific facts
  that are useful for answering the question.
- Build the reasoning chain by connecting the relevant evidence
  from the documents.
- The reasoning should follow a natural information-seeking
  process.
- Do not mention that you were given the answer.
- Do not mention the GOLDEN DOCUMENT label in the reasoning.
- Do not simply repeat the target answer.
- Every factual claim should be supported by information from
  the provided documents when possible.
- When referring to a document, ALWAYS use its exact Title.
- NEVER refer to documents as "Document 1", "Document 2", etc.

The supporting sentences inside the documents are marked with:

[SUPPORTING] ... [/SUPPORTING]

These markers indicate the annotated supporting evidence from
HotpotQA. Use the information naturally as evidence, but DO NOT
mention the [SUPPORTING] markers in your reasoning.

The final answer should be concise and appear at the end in
exactly this form:

Answer: <short answer>


============================================================
QUESTION
============================================================

{sample.question}


============================================================
TARGET ANSWER
============================================================

{sample.answer}


============================================================
GOLDEN DOCUMENTS
============================================================

{documents_text}


============================================================
TASK
============================================================

Reason through the question step by step.

Start from the question itself.

Determine what information is required to answer it.

Identify the relevant evidence in the provided documents.

Connect the relevant facts together and derive the answer.

When referring to evidence, identify the source using its exact
document Title.

Do not refer to documents using positional names such as
"Document 1" or "Document 2".

At the end, provide the concise final answer using exactly:

Answer: <short answer>
""".strip()

    return prompt


# ============================================================
# TEST
# ============================================================

def test_create_teacher_prompt():
    """
    Test teacher prompt generation using the first HotpotQA
    validation sample.
    """

    from datasets import load_dataset

    print("=" * 80)
    print("TEST — TEACHER PROMPT GENERATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load one HotpotQA sample
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
    # Convert sample
    # --------------------------------------------------------

    sample = (
        HotpotQASampleConverter.from_hotpotqa(
            raw_sample
        )
    )

    print("✓ Sample converted")

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = create_teacher_prompt(
        sample=sample,
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    assert isinstance(
        prompt,
        str,
    )

    assert len(prompt) > 0

    # --------------------------------------------------------
    # Question must exist
    # --------------------------------------------------------

    assert sample.question in prompt

    # --------------------------------------------------------
    # Answer must exist
    # --------------------------------------------------------

    assert sample.answer in prompt

    # --------------------------------------------------------
    # Every golden document must exist
    # --------------------------------------------------------

    for document in sample.golden_document:

        assert document.title in prompt

        assert document.content in prompt

    # --------------------------------------------------------
    # Number of golden documents
    # --------------------------------------------------------

    golden_marker_count = prompt.count(
        "[GOLDEN DOCUMENT]"
    )

    assert golden_marker_count == (
        len(sample.golden_document)
    )

    # --------------------------------------------------------
    # No distractor documents
    # --------------------------------------------------------

    assert (
        "[DISTRACTOR DOCUMENT]"
        not in prompt
    )

    # --------------------------------------------------------
    # Title-based reference instruction
    # --------------------------------------------------------

    assert (
        "ALWAYS use its exact Title"
        in prompt
    )

    assert (
        'NEVER refer to documents as "Document 1", '
        '"Document 2", etc.'
        in prompt
    )

    # --------------------------------------------------------
    # Supporting marker instruction
    # --------------------------------------------------------

    assert (
        "[SUPPORTING]"
        in prompt
    )

    assert (
        "[/SUPPORTING]"
        in prompt
    )

    # --------------------------------------------------------
    # Print prompt
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GENERATED TEACHER PROMPT")
    print("=" * 80)

    print(prompt)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Golden documents : "
        f"{len(sample.golden_document)}"
    )

    print(
        f"Prompt length    : "
        f"{len(prompt)} characters"
    )

    print()
    print(
        "✓ Teacher prompt generated successfully"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_create_teacher_prompt()