import json
from pathlib import Path
from typing import Any

from .musique_sample import (
    MuSiQueSample,
    MuSiQueSampleConverter,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DEMONSTRATION_FILE = (
    "shared/MuSiQue/teacher_reasoning_demonstrations.json"
)

QWEN_REPO = "Qwen/Qwen3-8B"


# ============================================================
# LOAD TEACHER DEMONSTRATIONS
# ============================================================

def load_teacher_demonstrations(
    file_path: str | Path = DEFAULT_DEMONSTRATION_FILE,
) -> list[dict]:
    """
    Load manually verified teacher reasoning demonstrations.

    Returns:
        A list of dictionaries containing:
        - question
        - dataset_answer
        - golden_documents
        - reasoning
    """

    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    demonstrations = []

    for item in data["demonstrations"]:
        demonstrations.append({
            "question": item["question"],
            "dataset_answer": item["dataset_answer"],
            "golden_documents": item["golden_documents"],
            "reasoning": item["reasoning"],
        })

    return demonstrations


# ============================================================
# SYSTEM PROMPT
# ============================================================

def create_system_prompt() -> str:
    """
    Create the system prompt containing the teacher role
    and general reasoning rules.
    """

    return """
You are an expert reasoning teacher generating high-quality
reasoning data for a retrieval-augmented language model.

Solve the given multi-hop question using the provided passages
and your prior knowledge where appropriate.

Your reasoning must be natural, coherent, well-structured,
and faithful to the available information.

Follow these rules:

- Ground every factual claim in the provided passages or your own prior knowledge.

- Whenever referring to evidence from a passage, use its exact title. Never refer to passages using positional names such as "Document 1", "Document 2", or "Passage 1".

- Follow the information dependencies required to answer the question. Clearly connect intermediate facts and explain how each piece of evidence supports the next reasoning step.

- Reasonable assumptions are allowed when supported by context, logic, or common human language usage. Explicitly state any assumption that a conclusion depends on.

- Do not present an assumption as a directly established fact. Do not fabricate missing facts or introduce increasingly speculative assumptions to bridge gaps in the reasoning.

- If a required relationship cannot be established through the provided passages, prior knowledge, or a reasonable assumption, stop at the first unsupported step. Do not proceed with subsequent reasoning steps that depend on that relationship.

- When the available information is insufficient to establish the answer, explicitly identify the missing relationship and use the exact answer: "Insufficient evidence in the provided documents."

- Explain the reasoning process rather than merely stating the final answer. Connect evidence and intermediate conclusions naturally, avoid unnecessary repetition, and do not skip necessary reasoning steps.

- Dataset answers are provided only as supplementary reference information. Do not trust them unconditionally or use them as definitive evidence. Independently evaluate their correctness against the provided passages, prior knowledge, and reasoning. Never mention or refer to dataset answers in your reasoning or final answer.

- Reasoning instructions are human-provided supplementary guidance, not mandatory procedures or authoritative evidence. You may follow, adapt, reorder, partially use, deviate from, or reject them as appropriate. Independently determine the most appropriate reasoning approach and the order in which to use the provided documents, relevant parametric knowledge, and logical inference. Always prioritize factual accuracy and evidential support over adherence to the suggested instructions. Never mention or refer to reasoning instructions in your reasoning or final answer.

- Do not mention internal dataset annotations, demonstrations, teacher supervision, or hidden target answers.

- Always distinguish between different types of relationships. Never assume that one relationship implies another or treat distinct relationships as interchangeable.

Return the reasoning first, followed by the concise final answer
using exactly the following format:

<answer>
short answer
</answer>

If the available information is insufficient to establish
the answer, use:

<answer>
Insufficient evidence in the provided documents.
</answer>
""".strip()


# ============================================================
# DOCUMENT FORMATTER
# ============================================================

def _get_document_field(
    document: Any,
    field: str,
    default: str = "",
) -> str:
    """
    Read a document field from either a dictionary or an object.
    """

    if isinstance(document, dict):
        value = document.get(field, default)
    else:
        value = getattr(document, field, default)

    return str(value)


def format_golden_passages(
    documents: list,
) -> str:
    """
    Format golden documents without XML tags.

    Format:
        Title: ...

        Content:
        ...
    """

    passage_sections = []

    for document in documents:
        title = _get_document_field(
            document,
            "title",
        )

        content = _get_document_field(
            document,
            "content",
        )

        passage_sections.append(
            f"Title: {title}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(passage_sections)


# ============================================================
# QUESTION FORMATTER
# ============================================================

def format_question(
    question: str,
    golden_documents: list | None = None,
) -> str:
    """
    Format a question.

    If golden_documents is provided, append the passages.
    """

    result = (
        f"Question:\n"
        f"{question.strip()}"
    )

    if golden_documents is not None:
        passages_text = format_golden_passages(
            golden_documents
        )

        result += (
            f"\n\n"
            f"Passages:\n"
            f"{passages_text}"
        )

    return result


# ============================================================
# DEMONSTRATION FORMATTER
# ============================================================

def format_demonstration(
    demonstration: dict,
    index: int,
) -> str:
    """
    Format one manually verified demonstration.

    Demonstrations intentionally exclude golden passages.

    Format:
        Demonstration N:

        Question:
        ...

        Dataset answer:
        ...

        Reasoning:
        ...
    """

    question = demonstration["question"].strip()

    dataset_answer = str(
        demonstration["dataset_answer"]
    ).strip()

    reasoning = demonstration["reasoning"].strip()

    return (
        f"Demonstration {index}:\n\n"
        f"Question:\n"
        f"{question}\n\n"
        f"Dataset answer:\n"
        f"{dataset_answer}\n\n"
        f"Reasoning:\n"
        f"{reasoning}"
    )


# ============================================================
# CURRENT SAMPLE FORMATTER
# ============================================================

def format_current_sample(
    sample: MuSiQueSample,
) -> str:
    """
    Format the current sample with:
        - Question
        - Golden passages
        - Dataset answer
        - Reasoning instruction

    The current sample intentionally excludes the raw
    question decomposition.
    """

    question_text = format_question(
        question=sample.question,
        golden_documents=sample.golden_document,
    )

    dataset_answer = str(
        sample.answer
    ).strip()

    reasoning_instruction = str(
        sample.reasoning_instruction
    ).strip()

    return (
        f"{question_text}\n\n"
        f"Dataset answer:\n"
        f"{dataset_answer}\n\n"
        f"Reasoning instruction:\n"
        f"{reasoning_instruction}\n\n"
        f"Reasoning:"
    )


# ============================================================
# USER PROMPT BUILDER
# ============================================================

def create_user_prompt(
    sample: MuSiQueSample,
    demonstrations: list[dict],
) -> str:
    """
    Build the user prompt containing few-shot demonstrations
    followed by the current sample.
    """

    sections = []

    # ========================================================
    # 1. FEW-SHOT DEMONSTRATIONS
    # ========================================================

    for index, demonstration in enumerate(
        demonstrations,
        start=1,
    ):
        sections.append(
            format_demonstration(
                demonstration=demonstration,
                index=index,
            )
        )

    # ========================================================
    # 2. CURRENT SAMPLE
    # ========================================================

    current_sample_text = format_current_sample(
        sample=sample,
    )

    sections.append(
        "Now solve the following question.\n\n"
        f"{current_sample_text}"
    )

    return "\n\n\n".join(sections)


# ============================================================
# TEACHER PROMPTS BUILDER
# ============================================================

def create_teacher_prompts(
    sample: MuSiQueSample,
    demonstrations: list[dict] | None = None,
    demonstration_file_path: str | Path = DEFAULT_DEMONSTRATION_FILE,
) -> tuple[str, str]:
    """
    Create separate system and user prompts.

    Returns:
        tuple[str, str]:
            - system_prompt
            - user_prompt
    """

    # Load demonstrations if not explicitly provided.
    if demonstrations is None:
        demonstrations = load_teacher_demonstrations(
            file_path=demonstration_file_path,
        )

    # Create system prompt.
    system_prompt = create_system_prompt()

    # Create user prompt.
    user_prompt = create_user_prompt(
        sample=sample,
        demonstrations=demonstrations,
    )

    return system_prompt, user_prompt


# ============================================================
# QWEN3 CHAT TEMPLATE
# ============================================================

def create_teacher_prompt(
    system_prompt: str,
    user_prompt: str,
    tokenizer,
    enable_thinking: bool = True,
) -> str:
    """
    Serialize system and user prompts using the Qwen3 chat template.

    Args:
        system_prompt:
            General teacher instructions.

        user_prompt:
            Few-shot demonstrations and current question.

        tokenizer:
            Qwen3 tokenizer.

        enable_thinking:
            Whether to enable Qwen3 thinking mode.

    Returns:
        Serialized ChatML prompt ready for generation.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


# ============================================================
# TEST
# ============================================================

def test_create_teacher_prompts():
    """
    Test teacher prompt generation using the first MuSiQue-Ans
    development sample.
    """

    from .load_dataset import load_musique_ans

    print("=" * 80)
    print("TEST — MUSIQUE TEACHER PROMPT GENERATION")
    print("=" * 80)

    # ========================================================
    # 1. LOAD SAMPLE
    # ========================================================

    dataset = load_musique_ans()

    raw_sample = dataset["dev"][0]

    sample = MuSiQueSampleConverter.from_musique(
        raw_sample
    )

    print("✓ MuSiQue sample loaded and converted")

    # ========================================================
    # 2. LOAD DEMONSTRATIONS
    # ========================================================

    demonstrations = load_teacher_demonstrations()

    assert len(demonstrations) > 0

    print(
        f"✓ Loaded {len(demonstrations)} demonstrations"
    )

    # ========================================================
    # 3. CREATE PROMPTS
    # ========================================================

    system_prompt, user_prompt = create_teacher_prompts(
        sample=sample,
        demonstrations=demonstrations,
    )

    assert isinstance(system_prompt, str)
    assert isinstance(user_prompt, str)

    print("✓ System and user prompts created")

    # ========================================================
    # 4. VALIDATE SYSTEM PROMPT
    # ========================================================

    assert "expert reasoning teacher" in system_prompt
    assert "prior knowledge" in system_prompt
    assert "reasonable assumptions" in system_prompt
    assert "first unsupported step" in system_prompt
    assert "dataset answer" in system_prompt
    assert "reasoning instructions" in system_prompt
    assert "Do not follow them blindly" in system_prompt
    assert "<answer>" in system_prompt
    assert "</answer>" in system_prompt

    print("✓ System prompt validated")

    # ========================================================
    # 5. VALIDATE DEMONSTRATIONS
    # ========================================================

    for index, demonstration in enumerate(
        demonstrations,
        start=1,
    ):
        formatted_demo = format_demonstration(
            demonstration=demonstration,
            index=index,
        )

        assert formatted_demo in user_prompt

        assert f"Demonstration {index}:" in formatted_demo

        assert demonstration["question"].strip() in formatted_demo

        assert demonstration["dataset_answer"].strip() in formatted_demo

        assert demonstration["reasoning"].strip() in formatted_demo

        # Demonstrations must not contain passages.
        assert "Passages:" not in formatted_demo
        assert "Title:" not in formatted_demo
        assert "Content:" not in formatted_demo

        # Demonstration reasoning must contain final answer tags.
        assert "<answer>" in demonstration["reasoning"]
        assert "</answer>" in demonstration["reasoning"]

    # No redundant rules sections.
    assert "\nRules:\n" not in user_prompt

    print("✓ Demonstrations validated")

    # ========================================================
    # 6. VALIDATE CURRENT SAMPLE
    # ========================================================

    current_sample_text = format_current_sample(
        sample
    )

    assert current_sample_text in user_prompt

    assert sample.question.strip() in current_sample_text

    # Verify current sample passages.
    assert "Passages:" in current_sample_text

    for document in sample.golden_document:
        title = _get_document_field(
            document,
            "title",
        )

        content = _get_document_field(
            document,
            "content",
        )

        assert f"Title: {title}" in current_sample_text
        assert f"Content:\n{content}" in current_sample_text

    # Verify dataset answer is included.
    assert "Dataset answer:" in current_sample_text
    assert str(sample.answer).strip() in current_sample_text

    # Verify reasoning instruction is included.
    assert "Reasoning instruction:" in current_sample_text

    assert (
        str(sample.reasoning_instruction).strip()
        in current_sample_text
    )

    # Verify XML tags are not used.
    assert "<passage>" not in current_sample_text
    assert "<title>" not in current_sample_text
    assert "<golden_passages>" not in current_sample_text

    # Current sample must not expose the raw question decomposition.
    assert "QUESTION DECOMPOSITION" not in current_sample_text
    assert "decomposition_text" not in current_sample_text

    # No distractor documents.
    assert "[DISTRACTOR DOCUMENT]" not in current_sample_text

    # The current sample must end with the answer-generation cue.
    assert current_sample_text.rstrip().endswith("Answer:")

    print("✓ Current sample validated")

    # ========================================================
    # 7. PRINT PROMPTS
    # ========================================================

    print("\n" + "=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)

    print(system_prompt)

    print("\n" + "=" * 80)
    print("USER PROMPT")
    print("=" * 80)

    print(user_prompt)

    # ========================================================
    # 8. SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Demonstrations       : {len(demonstrations)}"
    )

    print(
        f"Golden documents     : {len(sample.golden_document)}"
    )

    print(
        f"System prompt length : {len(system_prompt)} characters"
    )

    print(
        f"User prompt length   : {len(user_prompt)} characters"
    )

    print("\n✓ Demonstrations contain questions, dataset answers, and reasoning")
    print("✓ Demonstrations exclude golden passages")
    print("✓ Current sample contains formatted passages and dataset answer")
    print("✓ Current sample contains reasoning instruction")
    print("✓ Dataset answers are reference information, not absolute ground truth")
    print("✓ Reasoning instructions are reference information, not definitive reasoning")
    print("✓ Raw question decomposition is not separately exposed")

    print("\n✓ MUSIQUE TEACHER PROMPTS TEST PASSED")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from transformers import AutoTokenizer

    from .load_dataset import load_musique_ans

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    dataset = load_musique_ans()

    target_question = "What year saw the creation of the region where the county of Hertfordshire is located?"

    # ========================================================
    # 2. FIND TARGET SAMPLE
    # ========================================================

    raw_sample = None

    for split in ["dev", "train", "test"]:
        for item in dataset[split]:

            if item["question"] == target_question:
                raw_sample = item

                print(
                    f"✓ Found sample in split: {split}"
                )

                break

        if raw_sample is not None:
            break

    if raw_sample is None:
        raise ValueError(
            f"Could not find question:\n{target_question}"
        )

    # ========================================================
    # 3. CONVERT SAMPLE
    # ========================================================

    sample = MuSiQueSampleConverter.from_musique(
        raw_sample
    )

    print("✓ Sample converted")

    # ========================================================
    # 4. LOAD DEMONSTRATIONS
    # ========================================================

    demonstrations = load_teacher_demonstrations()

    print(
        f"✓ Loaded {len(demonstrations)} demonstrations"
    )

    # ========================================================
    # 5. CREATE SYSTEM AND USER PROMPTS
    # ========================================================

    system_prompt, user_prompt = create_teacher_prompts(
        sample=sample,
        demonstrations=demonstrations,
    )

    # ========================================================
    # 6. LOAD QWEN3 TOKENIZER
    # ========================================================

    print()
    print(f"Loading tokenizer: {QWEN_REPO}")

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_REPO,
        trust_remote_code=True,
    )

    print("✓ Qwen3 tokenizer loaded")

    # ========================================================
    # 7. APPLY QWEN3 CHAT TEMPLATE
    # ========================================================

    prompt = create_teacher_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tokenizer=tokenizer,
        enable_thinking=False,
    )

    # ========================================================
    # 8. PRINT SERIALIZED QWEN3 PROMPT
    # ========================================================

    print("\n" + "=" * 80)
    print("SERIALIZED QWEN3 CHAT TEMPLATE")
    print("=" * 80)

    print(prompt)

    # ========================================================
    # 9. PROMPT STATISTICS
    # ========================================================

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    print("\n" + "=" * 80)
    print("PROMPT STATISTICS")
    print("=" * 80)

    print(
        f"Demonstrations       : {len(demonstrations)}"
    )

    print(
        f"System prompt        : {len(system_prompt)} characters"
    )

    print(
        f"User prompt          : {len(user_prompt)} characters"
    )

    print(
        f"Serialized prompt    : {len(prompt)} characters"
    )

    print(
        f"Token count          : {len(token_ids)}"
    )

    print("\n" + "=" * 80)
    print("✓ QWEN3-8B TEACHER PROMPT CREATED")
    print("=" * 80)