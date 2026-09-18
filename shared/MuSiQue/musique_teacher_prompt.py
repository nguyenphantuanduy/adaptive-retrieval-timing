# ============================================================
# Teacher Prompt Builder
# ============================================================

from .musique_sample import (
    MuSiQueSample,
    Document,
    MuSiQueSampleConverter,
)


def create_teacher_prompt(
    sample: MuSiQueSample,
) -> dict[str, str]:

    # ========================================================
    # System Prompt
    # ========================================================

    system_prompt = """
You are a reasoning teacher generating high-quality reasoning
data for a retrieval-augmented language model.

Your task is to solve questions by reasoning over the
provided documents.

IMPORTANT:

- You are given the final answer as a target.
- Reconstruct a natural reasoning process that leads to that
  answer.
- Do NOT simply explain the answer directly.
- Use the original question as the starting point for your
  reasoning.
- Do NOT restate, repeat, or paraphrase the original question.
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
- Write a natural and coherent reasoning process.
- The reasoning should contain enough detail to clearly show
  how the intermediate facts lead to the final answer.
- Do not unnecessarily shorten or omit important reasoning
  steps.

The final answer should be concise and appear at the end.

The final answer MUST use exactly this format:

<answer>
YOUR_FINAL_ANSWER
</answer>

Do not include any additional text inside the <answer> tags.
Do not write "<short answer>" or any other placeholder inside
the <answer> tags.
""".strip()

    # ========================================================
    # Golden Documents
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
    # Reasoning Instruction
    # ========================================================

    reasoning_instruction = (
        sample.reasoning_instruction
    )

    # ========================================================
    # User Prompt
    # ========================================================

    user_prompt = f"""
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

Follow the necessary information dependencies between
intermediate facts.

For multi-hop questions, clearly connect the intermediate
answers so that the final answer follows naturally from them.

At the end, provide the concise final answer.
""".strip()

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


# ============================================================
# MAIN — TEST
# ============================================================

if __name__ == "__main__":

    from .load_dataset import load_musique_ans

    # ========================================================
    # LOAD MuSiQue
    # ========================================================

    dataset = load_musique_ans()

    # ========================================================
    # GET FIRST DEV SAMPLE
    # ========================================================

    raw_sample = dataset["dev"][0]

    # ========================================================
    # CONVERT SAMPLE
    # ========================================================

    sample = MuSiQueSampleConverter.from_musique(
        raw_sample
    )

    # ========================================================
    # CREATE TEACHER PROMPT
    # ========================================================

    prompts = create_teacher_prompt(
        sample
    )

    # ========================================================
    # PRINT SAMPLE INFO
    # ========================================================

    print("\n" + "=" * 80)
    print("SAMPLE")
    print("=" * 80)

    print(
        f"Question       : {sample.question}"
    )

    print(
        f"Target answer  : {sample.answer}"
    )

    print(
        f"Golden docs    : "
        f"{len(sample.golden_document)}"
    )

    print(
        f"Distractor     : "
        f"{len(sample.distract_document)}"
    )

    print(
        f"Reasoning steps: "
        f"{len(sample.question_decomposition)}"
    )

    # ========================================================
    # PRINT SYSTEM PROMPT
    # ========================================================

    print("\n" + "=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)

    print(
        prompts["system"]
    )

    # ========================================================
    # PRINT USER PROMPT
    # ========================================================

    print("\n" + "=" * 80)
    print("USER PROMPT")
    print("=" * 80)

    print(
        prompts["user"]
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)