from .musique_sample import (
    MuSiQueSample,
    Document,
    MuSiQueSampleConverter,
)


# ============================================================
# TEACHER PROMPT BUILDER
# ============================================================

def create_teacher_prompt(
    sample: MuSiQueSample,
) -> str:
    """
    Create an instruction prompt for teacher-model reasoning generation.

    The teacher receives:
        - question
        - target answer
        - golden documents
        - reasoning instruction

    The reasoning instruction is provided by the MuSiQueSample
    itself.

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
    # 2. GET REASONING INSTRUCTION FROM SAMPLE
    # ========================================================

    reasoning_instruction = sample.reasoning_instruction

    # ========================================================
    # 3. BUILD PROMPT
    # ========================================================

    prompt = f"""
You are a reasoning teacher generating high-quality reasoning
data for a retrieval-augmented language model.

Your task is to solve the question below by reasoning over the
provided documents.

The target answer is provided intentionally as supervision for
this teacher-generation process.

IMPORTANT:

- You are given the final answer as a target.
- Reconstruct a natural reasoning process that leads to that
  answer.
- Do NOT simply explain the answer directly.
- Start from the original question.
- Determine what information is required.
- Identify the relevant evidence.
- Connect intermediate facts when the question requires
  multi-hop reasoning.
- Every factual claim should be grounded in the provided
  documents whenever possible.
- When referring to a document, ALWAYS use its exact Title.
- NEVER refer to documents as "Document 1", "Document 2", etc.
- Do not mention that you were given the target answer.
- Do not mention internal dataset annotations.
- Do not mention document labels such as "[GOLDEN DOCUMENT]"
  or "[END DOCUMENT]" in the reasoning.
- Do not mention this instruction.
- Do not introduce information that is not supported by the
  provided documents.

The final answer should be concise and appear at the end in
exactly this form:

<answer>
short answer
</answer>


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

The following documents provide the evidence needed
to solve the question.

{documents_text}


============================================================
REASONING INSTRUCTION
============================================================

{reasoning_instruction}


============================================================
TASK
============================================================

Reason through the question step by step.

Use the provided reasoning instruction to guide the reasoning.

Start from the original question.

Determine what information is required.

Identify the relevant evidence in the provided documents.

Follow the necessary information dependencies between
intermediate facts.

For multi-hop questions, clearly connect the intermediate
answers so that the final answer follows naturally from them.

When referring to evidence, identify the source using its exact
document Title.

Do not refer to documents using positional names such as
"Document 1" or "Document 2".

Do not mention that the answer was provided as supervision.
Do not mention the dataset, annotations, or these instructions
in the reasoning.

Write a natural, coherent reasoning process.

At the end, provide the concise final answer using exactly:

<answer>
short answer
</answer>
""".strip()

    return prompt


# ============================================================
# TEST
# ============================================================

def test_create_teacher_prompt():
    """
    Test teacher prompt generation using the first MuSiQue-Ans
    development sample.
    """

    print("=" * 80)
    print("TEST — MUSIQUE TEACHER PROMPT GENERATION")
    print("=" * 80)

    # ========================================================
    # 1. LOAD ONE MUSIQUE SAMPLE
    # ========================================================

    from .load_dataset import load_musique_ans

    dataset = load_musique_ans()

    raw_sample = dataset["dev"][0]

    print()
    print("✓ MuSiQue sample loaded")

    # ========================================================
    # 2. CONVERT SAMPLE
    # ========================================================

    sample = (
        MuSiQueSampleConverter.from_musique(
            raw_sample
        )
    )

    print("✓ Sample converted")

    # ========================================================
    # 3. CREATE TEACHER PROMPT
    # ========================================================

    prompt = create_teacher_prompt(
        sample=sample,
    )

    assert isinstance(
        prompt,
        str,
    )

    assert len(prompt) > 0

    print("✓ Teacher prompt created")

    # ========================================================
    # 4. QUESTION
    # ========================================================

    assert (
        sample.question
        in prompt
    )

    # ========================================================
    # 5. TARGET ANSWER
    # ========================================================

    assert (
        sample.answer
        in prompt
    )

    # ========================================================
    # 6. REASONING INSTRUCTION
    # ========================================================

    assert (
        "REASONING INSTRUCTION"
        in prompt
    )

    assert (
        sample.reasoning_instruction
        in prompt
    )

    # ========================================================
    # 7. QUESTION DECOMPOSITION MUST NOT BE INCLUDED
    # ========================================================

    assert (
        "QUESTION DECOMPOSITION"
        not in prompt
    )

    assert (
        "decomposition_text"
        not in prompt
    )

    # ========================================================
    # 8. GOLDEN DOCUMENTS
    # ========================================================

    for document in sample.golden_document:

        # ----------------------------------------------------
        # Title must exist
        # ----------------------------------------------------

        assert (
            document.title
            in prompt
        )

        # ----------------------------------------------------
        # Content must exist
        # ----------------------------------------------------

        assert (
            document.content
            in prompt
        )

        # ----------------------------------------------------
        # Exact document section must exist
        # ----------------------------------------------------

        expected_section = (
            "[GOLDEN DOCUMENT]\n"
            f"Title: {document.title}\n"
            f"{document.content}\n"
            "[END DOCUMENT]"
        )

        assert (
            expected_section
            in prompt
        )

    # ========================================================
    # 9. NO DISTRACTOR DOCUMENTS
    # ========================================================

    assert (
        "[DISTRACTOR DOCUMENT]"
        not in prompt
    )

    # ========================================================
    # 10. DOCUMENT REFERENCE INSTRUCTIONS
    # ========================================================

    assert (
        "ALWAYS use its exact Title"
        in prompt
    )

    assert (
        'NEVER refer to documents as "Document 1", '
        '"Document 2", etc.'
        in prompt
    )

    # ========================================================
    # 11. ANSWER FORMAT
    # ========================================================

    assert (
        "Answer: <short answer>"
        in prompt
    )

    # ========================================================
    # 12. SECTION ORDER
    # ========================================================

    question_pos = prompt.index(
        "QUESTION"
    )

    target_answer_pos = prompt.index(
        "TARGET ANSWER"
    )

    golden_documents_pos = prompt.index(
        "GOLDEN DOCUMENTS"
    )

    reasoning_instruction_pos = prompt.index(
        "REASONING INSTRUCTION"
    )

    task_pos = prompt.index(
        "TASK"
    )

    assert (
        question_pos
        < target_answer_pos
        < golden_documents_pos
        < reasoning_instruction_pos
        < task_pos
    )

    # ========================================================
    # 13. PRINT GENERATED PROMPT
    # ========================================================

    print()
    print("=" * 80)
    print("GENERATED TEACHER PROMPT")
    print("=" * 80)

    print(prompt)

    # ========================================================
    # 14. SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Question             : "
        f"{sample.question}"
    )

    print(
        f"Answer               : "
        f"{sample.answer}"
    )

    print(
        f"Reasoning steps      : "
        f"{len(sample.question_decomposition)}"
    )

    print(
        f"Golden documents     : "
        f"{len(sample.golden_document)}"
    )

    print(
        f"Prompt length        : "
        f"{len(prompt)} characters"
    )

    print()

    print(
        "✓ Reasoning instruction loaded from MuSiQueSample"
    )

    print(
        "✓ Question decomposition NOT included"
    )

    print(
        "✓ Golden documents included and validated"
    )

    print(
        "✓ No distractor documents included"
    )

    print(
        "✓ Document reference instructions validated"
    )

    print(
        "✓ Reasoning instruction included"
    )

    print(
        "✓ Target answer included"
    )

    print(
        "✓ Section ordering validated"
    )

    print()
    print(
        "✓ MUSIQUE TEACHER PROMPT TEST PASSED"
    )

if __name__ == "__main__":
    test_create_teacher_prompt()