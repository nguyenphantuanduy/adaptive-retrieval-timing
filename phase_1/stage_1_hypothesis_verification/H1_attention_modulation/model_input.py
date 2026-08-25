from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

QWEN_REPO = "Qwen/Qwen3-4B"
INSTRUCTION = (
    "Reason about the question as needed, and at the end "
    "provide a concise final answer in the form "
    "\"Answer: <short answer>\"."
)

# Decision values:
#
#   0/32, 1/32, 2/32, ..., 32/32
#
# Therefore:
#
#   number of intervals = 32
#   number of decision values = 33
#
NUM_DECISION_LEVELS = 32


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Document:
    title: str
    content: str


@dataclass
class SampleObject:
    question: str
    documents: List[Document]


# ============================================================
# INPUT BUILDER
# ============================================================

def create_model_input(
    sample: SampleObject,
    tokenizer,
) -> Dict[str, Any]:
    """
    Create batched model inputs for one SampleObject.

    Input structure:

        QUESTION
            ↓
        chat template
            ↓
        [Retrieved document]
        Title: ...
        Content: ...
        ...
        [End retrieved document]

    There is NO simulated reasoning/generation between the
    question and the documents.

    The input is duplicated for every retrieval decision:

        0/32
        1/32
        2/32
        ...
        32/32

    Therefore:

        B = 33

    Returns
    -------
    Dict[str, torch.Tensor]

        input_ids:
            shape = (33, L)

        document_attention_mask:
            shape = (33, L)

        retrieval_decision:
            shape = (33,)
    """

    # ========================================================
    # 1. BUILD QUESTION / CHAT CONTEXT
    # ========================================================

    messages = [
        {
            "role": "user",
            "content": (
                sample.question
                + " "
                + INSTRUCTION
            ),
        }
    ]

    base_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # ========================================================
    # 2. BUILD DOCUMENT SECTION
    #
    # Same document ordering as sample.documents.
    # ========================================================

    document_parts = [
        "[Retrieved document]\n"
    ]

    for document in sample.documents:
        document_parts.append(
            f"Title: {document.title}\n"
            f"{document.content}\n"
        )

    document_parts.append(
        "[End retrieved document]\n"
    )

    document_text = "\n".join(document_parts)

    # Question/chat + documents.
    full_text = (
        base_text
        + document_text
    )

    # ========================================================
    # 3. DOCUMENT CHARACTER SPAN
    # ========================================================

    document_start = full_text.find(
        document_text
    )

    if document_start == -1:
        raise RuntimeError(
            "Could not locate document_text inside full_text."
        )

    document_end = (
        document_start
        + len(document_text)
    )

    # ========================================================
    # 4. TOKENIZE BEFORE / DOCUMENT SEPARATELY
    #
    # This is important because we need to know exactly which
    # token positions correspond to the document.
    # ========================================================

    text_before = full_text[:document_start]

    text_document = full_text[
        document_start:document_end
    ]

    before_ids = tokenizer(
        text_before,
        add_special_tokens=False,
    )["input_ids"]

    document_ids = tokenizer(
        text_document,
        add_special_tokens=False,
    )["input_ids"]

    input_id_list = (
        before_ids
        + document_ids
    )

    # ========================================================
    # 5. SINGLE INPUT
    # ========================================================

    input_ids_single = torch.tensor(
        [input_id_list],
        dtype=torch.long,
    )

    L = input_ids_single.shape[1]

    # ========================================================
    # 6. DOCUMENT TOKEN SPAN
    # ========================================================

    document_start_token = len(
        before_ids
    )

    document_end_token = (
        document_start_token
        + len(document_ids)
    )

    # Sanity checks.
    assert (
        document_start_token
        >= 0
    )

    assert (
        document_end_token
        <= L
    )

    assert (
        document_end_token
        > document_start_token
    )

    # ========================================================
    # 7. DOCUMENT ATTENTION MASK
    #
    # 0      -> normal / visible
    # -inf   -> document span
    #
    # Shape before duplication:
    #
    #     (L,)
    #
    # Shape after duplication:
    #
    #     (33, L)
    # ========================================================

    document_attention_mask_single = torch.zeros(
        L,
        dtype=torch.float32,
    )

    document_attention_mask_single[
        document_start_token:document_end_token
    ] = float("-inf")

    # ========================================================
    # 8. DECISION VALUES
    #
    #     0/32, 1/32, ..., 32/32
    #
    # Shape:
    #
    #     (33,)
    # ========================================================

    retrieval_decision = (
        torch.arange(
            NUM_DECISION_LEVELS + 1,
            dtype=torch.float32,
        )
        / NUM_DECISION_LEVELS
    )

    batch_size = (
        NUM_DECISION_LEVELS + 1
    )

    # ========================================================
    # 9. DUPLICATE INPUT
    # ========================================================

    input_ids = input_ids_single.repeat(
        batch_size,
        1,
    )

    document_attention_mask = (
        document_attention_mask_single
        .unsqueeze(0)
        .repeat(
            batch_size,
            1,
        )
    )

    # ========================================================
    # 10. FINAL SANITY CHECKS
    # ========================================================

    assert input_ids.shape == (
        batch_size,
        L,
    )

    assert document_attention_mask.shape == (
        batch_size,
        L,
    )

    assert retrieval_decision.shape == (
        batch_size,
    )

    return {
        "input_ids": input_ids,
        "document_attention_mask": document_attention_mask,
        "retrieval_decision": retrieval_decision,
        "full_text": full_text,
    }


# ============================================================
# TEST HELPERS
# ============================================================

def _print_test_header(name: str):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)


def _check_model_input(
    sample: SampleObject,
    tokenizer,
    result: Dict[str, torch.Tensor],
):
    """
    Deep validation of create_model_input().
    """

    input_ids = result["input_ids"]
    document_attention_mask = (
        result["document_attention_mask"]
    )
    retrieval_decision = (
        result["retrieval_decision"]
    )

    # --------------------------------------------------------
    # Expected batch size
    # --------------------------------------------------------

    expected_B = (
        NUM_DECISION_LEVELS + 1
    )

    assert (
        retrieval_decision.shape
        == (expected_B,)
    )

    assert (
        input_ids.shape[0]
        == expected_B
    )

    assert (
        document_attention_mask.shape[0]
        == expected_B
    )

    # --------------------------------------------------------
    # Same sequence length
    # --------------------------------------------------------

    B, L = input_ids.shape

    assert (
        document_attention_mask.shape
        == (B, L)
    )

    # --------------------------------------------------------
    # Decision values
    # --------------------------------------------------------

    expected_decisions = (
        torch.arange(
            NUM_DECISION_LEVELS + 1,
            dtype=torch.float32,
        )
        / NUM_DECISION_LEVELS
    )

    assert torch.allclose(
        retrieval_decision,
        expected_decisions,
    )

    assert retrieval_decision[0].item() == 0.0

    assert retrieval_decision[-1].item() == 1.0

    # --------------------------------------------------------
    # Monotonically increasing
    # --------------------------------------------------------

    assert torch.all(
        retrieval_decision[1:]
        > retrieval_decision[:-1]
    )

    # --------------------------------------------------------
    # Step size = 1/32
    # --------------------------------------------------------

    expected_step = (
        1.0 / NUM_DECISION_LEVELS
    )

    steps = (
        retrieval_decision[1:]
        - retrieval_decision[:-1]
    )

    assert torch.allclose(
        steps,
        torch.full_like(
            steps,
            expected_step,
        ),
    )

    # --------------------------------------------------------
    # All input rows must be identical.
    #
    # Only retrieval_decision changes between rows.
    # --------------------------------------------------------

    for i in range(1, B):
        assert torch.equal(
            input_ids[0],
            input_ids[i],
        )

        assert torch.equal(
            document_attention_mask[0],
            document_attention_mask[i],
        )

    # --------------------------------------------------------
    # Mask should contain only:
    #
    #     0
    #     -inf
    # --------------------------------------------------------

    valid_mask_values = (
        (document_attention_mask == 0)
        | torch.isneginf(
            document_attention_mask
        )
    )

    assert valid_mask_values.all()

    # --------------------------------------------------------
    # There must be a masked document span.
    # --------------------------------------------------------

    masked_positions = torch.isneginf(
        document_attention_mask[0]
    )

    assert masked_positions.any()

    masked_indices = torch.nonzero(
        masked_positions,
        as_tuple=False,
    ).flatten()

    masked_start = (
        masked_indices[0].item()
    )

    masked_end = (
        masked_indices[-1].item()
        + 1
    )

    # --------------------------------------------------------
    # Masked span must be contiguous.
    # --------------------------------------------------------

    expected_masked_indices = torch.arange(
        masked_start,
        masked_end,
        dtype=masked_indices.dtype,
    )

    assert torch.equal(
        masked_indices,
        expected_masked_indices,
    )

    # --------------------------------------------------------
    # Everything before / after document is visible.
    # --------------------------------------------------------

    assert torch.all(
        document_attention_mask[
            0,
            :masked_start,
        ] == 0
    )

    assert torch.all(
        document_attention_mask[
            0,
            masked_end:,
        ] == 0
    )

    # --------------------------------------------------------
    # Decode document span.
    # --------------------------------------------------------

    decoded_document = tokenizer.decode(
        input_ids[
            0,
            masked_start:masked_end,
        ].tolist(),
        skip_special_tokens=False,
    )

    assert "[Retrieved document]" in (
        decoded_document
    )

    assert "[End retrieved document]" in (
        decoded_document
    )

    # --------------------------------------------------------
    # Every document title/content must exist in the
    # decoded document span.
    # --------------------------------------------------------

    for document in sample.documents:

        assert document.title in (
            decoded_document
        )

        assert document.content in (
            decoded_document
        )

    return {
        "B": B,
        "L": L,
        "masked_start": masked_start,
        "masked_end": masked_end,
        "masked_tokens": (
            masked_end - masked_start
        ),
        "decoded_document": decoded_document,
    }


# ============================================================
# MAIN TEST
# ============================================================

def main():

    _print_test_header(
        "LOAD TOKENIZER"
    )

    print(
        f"Repository: {QWEN_REPO}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            QWEN_REPO,
        )
    )

    print("✓ Tokenizer loaded")

    # ========================================================
    # TEST 1
    # Single document
    # ========================================================

    _print_test_header(
        "TEST 1 — SINGLE DOCUMENT"
    )

    sample_1 = SampleObject(
        question=(
            "What is the energy density of "
            "the experimental battery?"
        ),
        documents=[
            Document(
                title="Aurora Research Institute",
                content=(
                    "The Aurora Research Institute's "
                    "experimental battery achieved an "
                    "energy density of 412 Wh/kg."
                ),
            )
        ],
    )

    result_1 = create_model_input(
        sample_1,
        tokenizer,
    )

    info_1 = _check_model_input(
        sample_1,
        tokenizer,
        result_1,
    )

    print(
        f"B: {info_1['B']}"
    )

    print(
        f"L: {info_1['L']}"
    )

    print(
        f"Masked span: "
        f"{info_1['masked_start']}"
        f" → "
        f"{info_1['masked_end']}"
    )

    print(
        f"Masked tokens: "
        f"{info_1['masked_tokens']}"
    )

    print(
        "Decisions:"
    )

    print(
        result_1[
            "retrieval_decision"
        ]
    )

    print(
        "✓ Single-document test passed"
    )

    # ========================================================
    # TEST 2
    # Multiple documents
    # ========================================================

    _print_test_header(
        "TEST 2 — MULTIPLE DOCUMENTS"
    )

    sample_2 = SampleObject(
        question=(
            "Which institute developed the "
            "experimental battery?"
        ),
        documents=[
            Document(
                title="Institute A",
                content=(
                    "Institute A researches "
                    "solar panels."
                ),
            ),
            Document(
                title="Institute B",
                content=(
                    "Institute B developed an "
                    "experimental battery."
                ),
            ),
            Document(
                title="Institute C",
                content=(
                    "Institute C studies "
                    "quantum materials."
                ),
            ),
        ],
    )

    result_2 = create_model_input(
        sample_2,
        tokenizer,
    )

    info_2 = _check_model_input(
        sample_2,
        tokenizer,
        result_2,
    )

    print(
        f"B: {info_2['B']}"
    )

    print(
        f"L: {info_2['L']}"
    )

    print(
        f"Masked tokens: "
        f"{info_2['masked_tokens']}"
    )

    print(
        "✓ Multiple-document test passed"
    )

    # ========================================================
    # TEST 3
    # Unicode / special characters
    # ========================================================

    _print_test_header(
        "TEST 3 — UNICODE / SPECIAL CHARACTERS"
    )

    sample_3 = SampleObject(
        question=(
            "Thông số của thiết bị là bao nhiêu?"
        ),
        documents=[
            Document(
                title="Viện Nghiên cứu Ánh Dương — 2026",
                content=(
                    "Thiết bị đạt hiệu suất 99.5%, "
                    "năng lượng 412 Wh/kg và "
                    "nhiệt độ hoạt động −20°C."
                ),
            )
        ],
    )

    result_3 = create_model_input(
        sample_3,
        tokenizer,
    )

    info_3 = _check_model_input(
        sample_3,
        tokenizer,
        result_3,
    )

    print(
        f"B: {info_3['B']}"
    )

    print(
        f"L: {info_3['L']}"
    )

    print(
        info_3[
            "decoded_document"
        ]
    )

    print(
        "✓ Unicode/special-character test passed"
    )

    # ========================================================
    # TEST 4
    # Empty document content
    # ========================================================

    _print_test_header(
        "TEST 4 — EMPTY DOCUMENT CONTENT"
    )

    sample_4 = SampleObject(
        question="What is the title?",
        documents=[
            Document(
                title="Empty Content Document",
                content="",
            )
        ],
    )

    result_4 = create_model_input(
        sample_4,
        tokenizer,
    )

    info_4 = _check_model_input(
        sample_4,
        tokenizer,
        result_4,
    )

    print(
        f"B: {info_4['B']}"
    )

    print(
        f"L: {info_4['L']}"
    )

    print(
        "✓ Empty-content test passed"
    )

    # ========================================================
    # TEST 5
    # Empty documents list
    #
    # The function still creates a document section:
    #
    #   [Retrieved document]
    #   [End retrieved document]
    #
    # This verifies that the function does not crash when
    # documents is empty.
    # ========================================================

    _print_test_header(
        "TEST 5 — EMPTY DOCUMENT LIST"
    )

    sample_5 = SampleObject(
        question="This sample has no document.",
        documents=[],
    )

    result_5 = create_model_input(
        sample_5,
        tokenizer,
    )

    info_5 = _check_model_input(
        sample_5,
        tokenizer,
        result_5,
    )

    print(
        f"B: {info_5['B']}"
    )

    print(
        f"L: {info_5['L']}"
    )

    print(
        "✓ Empty-documents test passed"
    )

    # ========================================================
    # TEST 6
    # Explicit decision boundary checks
    # ========================================================

    _print_test_header(
        "TEST 6 — DECISION BOUNDARIES"
    )

    decisions = result_1[
        "retrieval_decision"
    ]

    assert decisions[0].item() == 0.0

    assert decisions[-1].item() == 1.0

    assert abs(
        decisions[1].item()
        - (1.0 / 32.0)
    ) < 1e-7

    assert abs(
        decisions[16].item()
        - 0.5
    ) < 1e-7

    assert abs(
        decisions[32].item()
        - 1.0
    ) < 1e-7

    print(
        "decision[0]  =",
        decisions[0].item(),
    )

    print(
        "decision[1]  =",
        decisions[1].item(),
    )

    print(
        "decision[16] =",
        decisions[16].item(),
    )

    print(
        "decision[32] =",
        decisions[32].item(),
    )

    print(
        "✓ Decision boundary test passed"
    )

    # ========================================================
    # TEST 7
    # Batch duplication integrity
    # ========================================================

    _print_test_header(
        "TEST 7 — BATCH DUPLICATION INTEGRITY"
    )

    input_ids = result_1[
        "input_ids"
    ]

    mask = result_1[
        "document_attention_mask"
    ]

    for i in range(
        NUM_DECISION_LEVELS + 1
    ):

        assert torch.equal(
            input_ids[0],
            input_ids[i],
        )

        assert torch.equal(
            mask[0],
            mask[i],
        )

    print(
        "✓ All 33 input rows are identical"
    )

    print(
        "✓ All 33 attention-mask rows are identical"
    )

    # ========================================================
    # TEST 8
    # Ensure different calls do not share mutable tensors
    # ========================================================

    _print_test_header(
        "TEST 8 — OUTPUT INDEPENDENCE"
    )

    result_a = create_model_input(
        sample_1,
        tokenizer,
    )

    result_b = create_model_input(
        sample_1,
        tokenizer,
    )

    result_a[
        "input_ids"
    ][0, 0] += 1

    assert not torch.equal(
        result_a["input_ids"],
        result_b["input_ids"],
    )

    # The second result must remain intact.
    fresh = create_model_input(
        sample_1,
        tokenizer,
    )

    assert torch.equal(
        result_b["input_ids"],
        fresh["input_ids"],
    )

    assert torch.equal(
        result_b[
            "document_attention_mask"
        ],
        fresh[
            "document_attention_mask"
        ],
    )

    assert torch.equal(
        result_b[
            "retrieval_decision"
        ],
        fresh[
            "retrieval_decision"
        ],
    )

    print(
        "✓ Separate function calls are independent"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    _print_test_header(
        "ALL TESTS PASSED"
    )

    print(
        "✓ Single document"
    )

    print(
        "✓ Multiple documents"
    )

    print(
        "✓ Unicode / special characters"
    )

    print(
        "✓ Empty document content"
    )

    print(
        "✓ Empty document list"
    )

    print(
        "✓ Decision values: 0/32 → 32/32"
    )

    print(
        "✓ Batch duplication"
    )

    print(
        "✓ Document attention mask"
    )

    print(
        "✓ Output independence"
    )

    print()

    print(
        f"NUM_DECISION_LEVELS = "
        f"{NUM_DECISION_LEVELS}"
    )

    print(
        f"Actual batch size B = "
        f"{NUM_DECISION_LEVELS + 1}"
    )

    print(
        "Decision step = "
        f"1/{NUM_DECISION_LEVELS}"
    )


def debug_multi_document_input():
    """
    Debug create_model_input() using Qwen3-4B tokenizer.

    Prints:
        1. Output tensor shapes
        2. Full model input text
        3. Masked token range
        4. Decoded masked document span
        5. Mask correctness checks
    """

    _print_test_header(
        "DEBUG TEST — MULTI-DOCUMENT INPUT"
    )

    # ========================================================
    # 1. LOAD QWEN3-4B TOKENIZER
    # ========================================================

    print(
        f"Repository: {QWEN_REPO}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_REPO,
    )

    print("✓ Tokenizer loaded")

    # ========================================================
    # 2. MULTI-DOCUMENT SAMPLE
    # ========================================================

    debug_sample = SampleObject(
        question=(
            "Which institute developed the experimental "
            "battery, and what was its energy density?"
        ),
        documents=[
            Document(
                title="Institute A",
                content=(
                    "Institute A researches advanced "
                    "solar panel materials."
                ),
            ),
            Document(
                title="Aurora Research Institute",
                content=(
                    "The Aurora Research Institute developed "
                    "an experimental battery that achieved "
                    "an energy density of 412 Wh/kg."
                ),
            ),
            Document(
                title="Institute C",
                content=(
                    "Institute C studies quantum materials "
                    "and low-temperature superconductors."
                ),
            ),
            Document(
                title="Institute D",
                content=(
                    "Institute D focuses on atmospheric "
                    "monitoring systems."
                ),
            ),
        ],
    )

    # ========================================================
    # 3. CREATE MODEL INPUT
    # ========================================================

    result = create_model_input(
        debug_sample,
        tokenizer,
    )

    input_ids = result["input_ids"]

    document_attention_mask = (
        result["document_attention_mask"]
    )

    retrieval_decision = (
        result["retrieval_decision"]
    )

    full_text = result["full_text"]

    # ========================================================
    # 4. PRINT OUTPUT SHAPES
    # ========================================================

    print()
    print("-" * 80)
    print("OUTPUT SHAPES")
    print("-" * 80)

    print(
        f"input_ids                : "
        f"shape={tuple(input_ids.shape)}, "
        f"dtype={input_ids.dtype}"
    )

    print(
        f"document_attention_mask : "
        f"shape={tuple(document_attention_mask.shape)}, "
        f"dtype={document_attention_mask.dtype}"
    )

    print(
        f"retrieval_decision      : "
        f"shape={tuple(retrieval_decision.shape)}, "
        f"dtype={retrieval_decision.dtype}"
    )

    print(
        f"full_text               : "
        f"type={type(full_text).__name__}"
    )

    # ========================================================
    # 5. PRINT FULL TEXT
    # ========================================================

    print()
    print("=" * 80)
    print("FULL TEXT USED BY MODEL")
    print("=" * 80)

    print(full_text)

    # ========================================================
    # 6. FIND MASKED DOCUMENT SPAN
    # ========================================================

    masked_positions = torch.isneginf(
        document_attention_mask[0]
    )

    assert masked_positions.any(), (
        "No masked document positions found."
    )

    masked_indices = torch.nonzero(
        masked_positions,
        as_tuple=False,
    ).flatten()

    masked_start = masked_indices[0].item()

    masked_end = (
        masked_indices[-1].item()
        + 1
    )

    # ========================================================
    # 7. DECODE MASKED SPAN
    # ========================================================

    masked_text = tokenizer.decode(
        input_ids[
            0,
            masked_start:masked_end,
        ].tolist(),
        skip_special_tokens=False,
    )

    # ========================================================
    # 8. PRINT MASKED SPAN
    # ========================================================

    print()
    print("=" * 80)
    print("MASKED DOCUMENT SPAN")
    print("=" * 80)

    print(
        f"Masked token range : "
        f"{masked_start} → {masked_end}"
    )

    print(
        f"Masked token count : "
        f"{masked_end - masked_start}"
    )

    print()
    print("-" * 80)
    print("DECODED MASKED CONTENT")
    print("-" * 80)

    print(masked_text)

    # ========================================================
    # 9. VERIFY MASK
    # ========================================================

    print()
    print("=" * 80)
    print("MASK VERIFICATION")
    print("=" * 80)

    # Document span must be -inf.
    assert torch.isneginf(
        document_attention_mask[
            0,
            masked_start:masked_end,
        ]
    ).all()

    print(
        "✓ Document span is fully masked (-inf)"
    )

    # Everything before document must be visible.
    assert torch.all(
        document_attention_mask[
            0,
            :masked_start,
        ] == 0
    )

    print(
        "✓ Tokens before document are visible"
    )

    # Everything after document must be visible.
    assert torch.all(
        document_attention_mask[
            0,
            masked_end:,
        ] == 0
    )

    print(
        "✓ Tokens after document are visible"
    )

    # Every document must exist in masked span.
    for document in debug_sample.documents:

        assert document.title in masked_text

        assert document.content in masked_text

        print(
            f"✓ Document found: "
            f"{document.title}"
        )

    # ========================================================
    # 10. VERIFY BATCH DUPLICATION
    # ========================================================

    for i in range(
        NUM_DECISION_LEVELS + 1
    ):

        assert torch.equal(
            input_ids[0],
            input_ids[i],
        )

        assert torch.equal(
            document_attention_mask[0],
            document_attention_mask[i],
        )

    print(
        "✓ All 33 input rows are identical"
    )

    # ========================================================
    # 11. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)

    print(
        f"Batch size B       : "
        f"{input_ids.shape[0]}"
    )

    print(
        f"Sequence length L  : "
        f"{input_ids.shape[1]}"
    )

    print(
        f"Decision values    : "
        f"{retrieval_decision.shape[0]}"
    )

    print(
        f"Masked token range : "
        f"{masked_start} → {masked_end}"
    )

    print(
        f"Masked token count : "
        f"{masked_end - masked_start}"
    )

    print()
    print(
        "✓ MULTI-DOCUMENT DEBUG TEST PASSED"
    )

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    # debug_multi_document_input()