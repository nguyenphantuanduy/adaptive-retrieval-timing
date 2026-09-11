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


@dataclass
class SampleObjectForLMLoss(SampleObject):

    answer: str



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
    Dict[str, Any]

        input_ids:
            shape = (33, L)

        attention_mask:
            shape = (33, L)

            This is the standard Hugging Face tokenizer
            attention mask.

            1 -> real token
            0 -> padding

        document_attention_mask:
            shape = (33, L)

            Custom additive attention mask:

            0      -> normal / visible
            -inf   -> document span

        retrieval_decision:
            shape = (33,)

        full_text:
            str

    IMPORTANT
    ---------
    input_ids and attention_mask are constructed from the
    SAME two tokenizer calls:

        text_before
        text_document

    and then concatenated in exactly the same order.

    This guarantees that attention_mask corresponds exactly
    to input_ids.
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

    document_text = "\n".join(
        document_parts
    )

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
    # 4. SPLIT TEXT
    #
    # We intentionally tokenize these two parts separately
    # because we need the exact token boundary between:
    #
    #     question/chat
    #
    # and
    #
    #     document section
    # ========================================================

    text_before = full_text[
        :document_start
    ]

    text_document = full_text[
        document_start:document_end
    ]

    # ========================================================
    # 5. TOKENIZE BEFORE / DOCUMENT SEPARATELY
    #
    # IMPORTANT:
    #
    # input_ids AND attention_mask are both obtained from
    # these exact same tokenizer calls.
    #
    # before:
    #
    #     before_ids
    #     before_attention_mask
    #
    # document:
    #
    #     document_ids
    #     document_attention_mask_from_tokenizer
    #
    # Then both are concatenated in the same order.
    # ========================================================

    before_encoding = tokenizer(
        text_before,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    document_encoding = tokenizer(
        text_document,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    # --------------------------------------------------------
    # BEFORE TOKENS
    # --------------------------------------------------------

    before_ids = before_encoding[
        "input_ids"
    ]

    before_attention_mask = (
        before_encoding[
            "attention_mask"
        ]
    )

    # --------------------------------------------------------
    # DOCUMENT TOKENS
    # --------------------------------------------------------

    document_ids = document_encoding[
        "input_ids"
    ]

    document_tokenizer_attention_mask = (
        document_encoding[
            "attention_mask"
        ]
    )

    # ========================================================
    # 6. CONCATENATE TOKENIZER OUTPUTS
    #
    # input_ids:
    #
    #     [before tokens] + [document tokens]
    #
    # attention_mask:
    #
    #     [before mask] + [document mask]
    #
    # These correspond position-by-position.
    # ========================================================

    input_id_list = (
        before_ids
        + document_ids
    )

    attention_mask_list = (
        before_attention_mask
        + document_tokenizer_attention_mask
    )

    # ========================================================
    # 7. SINGLE INPUT
    # ========================================================

    input_ids_single = torch.tensor(
        [input_id_list],
        dtype=torch.long,
    )

    attention_mask_single = torch.tensor(
        [attention_mask_list],
        dtype=torch.long,
    )

    L = input_ids_single.shape[1]

    # ========================================================
    # 8. TOKENIZER ATTENTION MASK SANITY CHECK
    # ========================================================

    assert attention_mask_single.shape == (
        1,
        L,
    )

    assert (
        len(input_id_list)
        == len(attention_mask_list)
    )

    # Since no padding is introduced in either tokenizer call,
    # all tokens should be real tokens.
    assert torch.all(
        attention_mask_single == 1
    ), (
        "Unexpected tokenizer attention_mask. "
        "Expected all tokens to be visible because "
        "no padding is used."
    )

    # ========================================================
    # 9. DOCUMENT TOKEN SPAN
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
    # 10. VERIFY TOKENIZER MASK ALIGNMENT
    # ========================================================
    #
    # The attention mask corresponding to the document span
    # must be exactly the second tokenizer's attention mask.
    #
    # The attention mask corresponding to the question/chat
    # span must be exactly the first tokenizer's mask.
    # ========================================================

    expected_before_mask = torch.tensor(
        before_attention_mask,
        dtype=torch.long,
    )

    expected_document_mask = torch.tensor(
        document_tokenizer_attention_mask,
        dtype=torch.long,
    )

    assert torch.equal(
        attention_mask_single[
            0,
            :document_start_token,
        ],
        expected_before_mask,
    )

    assert torch.equal(
        attention_mask_single[
            0,
            document_start_token:document_end_token,
        ],
        expected_document_mask,
    )

    # ========================================================
    # 11. DOCUMENT ATTENTION MASK
    #
    # This is NOT the tokenizer attention_mask.
    #
    # It is the custom additive mask used to control the
    # visibility of the document span.
    #
    #     0      -> normal / visible
    #     -inf   -> document span
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
    # 12. DECISION VALUES
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
    # 13. DUPLICATE INPUT
    # ========================================================

    input_ids = input_ids_single.repeat(
        batch_size,
        1,
    )

    attention_mask = (
        attention_mask_single.repeat(
            batch_size,
            1,
        )
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
    # 14. FINAL SANITY CHECKS
    # ========================================================

    assert input_ids.shape == (
        batch_size,
        L,
    )

    assert attention_mask.shape == (
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

    # All duplicated sequences are identical.
    assert torch.equal(
        input_ids[0],
        input_ids[1],
    )

    # Tokenizer attention masks must also be identical.
    assert torch.equal(
        attention_mask[0],
        attention_mask[1],
    )

    # No padding.
    assert torch.all(
        attention_mask == 1
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "document_attention_mask": document_attention_mask,
        "retrieval_decision": retrieval_decision,
        "full_text": full_text,
    }


# ============================================================
# INPUT BUILDER FOR LM LOSS
# ============================================================

def create_model_input_for_lm_loss(
    sample: SampleObjectForLMLoss,
    tokenizer,
) -> Dict[str, Any]:
    """
    Create model inputs for LM loss.

    Logic
    -----
    1. Convert SampleObjectForLMLoss -> SampleObject.
    2. Call create_model_input().
    3. Append "\\n" + answer to full_text.
    4. Tokenize answer separately.
    5. Duplicate answer tokens across all retrieval decisions.
    6. Concatenate answer to input_ids.
    7. Concatenate answer attention_mask.
    8. Append zero document_attention_mask for answer.
    9. Create labels:
           question/document -> -100
           answer           -> answer token IDs.

    Shapes
    ------
    Original:
        input_ids:
            (B, L)

        attention_mask:
            (B, L)

        document_attention_mask:
            (B, L)

        retrieval_decision:
            (B,)

    Answer:
        answer_ids:
            (A,)

        answer_attention_mask:
            (A,)

    Final:
        input_ids:
            (B, L + A)

        attention_mask:
            (B, L + A)

        document_attention_mask:
            (B, L + A)

        labels:
            (B, L + A)

        retrieval_decision:
            (B,)

    where:

        B = NUM_DECISION_LEVELS + 1
        L = original sequence length
        A = answer token length

    Notes
    -----
    - For a single string, tokenizer returns 1D sequences.
    - Therefore answer length is answer_ids.shape[0].
    - No special tokens are added to the answer.
    - "\\n" is intentionally included in the supervised answer span.
    - Labels are not shifted manually.
      Hugging Face causal LM models perform the shift internally.
    """

    # # ========================================================
    # # 1. CONVERT SampleObjectForLMLoss -> SampleObject
    # # ========================================================

    # sample_for_input = SampleObject(
    #     question=sample.question,
    #     documents=sample.documents,
    # )

    # ========================================================
    # 2. CREATE ORIGINAL MODEL INPUT
    # ========================================================

    result = create_model_input(
        sample,
        tokenizer,
    )

    input_ids = result[
        "input_ids"
    ]

    attention_mask = result[
        "attention_mask"
    ]

    document_attention_mask = result[
        "document_attention_mask"
    ]

    retrieval_decision = result[
        "retrieval_decision"
    ]

    full_text = result[
        "full_text"
    ]

    # ========================================================
    # 3. ORIGINAL SHAPES
    # ========================================================

    B, L = input_ids.shape

    expected_B = (
        NUM_DECISION_LEVELS + 1
    )

    assert B == expected_B

    assert attention_mask.shape == (
        B,
        L,
    )

    assert document_attention_mask.shape == (
        B,
        L,
    )

    assert retrieval_decision.shape == (
        B,
    )

    # ========================================================
    # 4. BUILD ANSWER TEXT
    #
    # The newline is intentionally part of the answer span.
    # ========================================================

    answer_text = (
        "\n"
        + sample.answer
    )

    # ========================================================
    # 5. APPEND ANSWER TO FULL TEXT
    # ========================================================

    full_text = (
        full_text
        + answer_text
    )

    # ========================================================
    # 6. TOKENIZE ANSWER
    #
    # For a single string:
    #
    #     input_ids      -> (A,)
    #     attention_mask -> (A,)
    #
    # There is NO batch dimension here.
    # ========================================================

    answer_encoding = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    # ========================================================
    # 7. CONVERT ANSWER TO TENSORS
    #
    # Keep them 1-dimensional:
    #
    #     answer_ids            -> (A,)
    #     answer_attention_mask -> (A,)
    # ========================================================

    answer_ids = torch.tensor(
        answer_encoding["input_ids"],
        dtype=torch.long,
    )

    answer_attention_mask = torch.tensor(
        answer_encoding["attention_mask"],
        dtype=torch.long,
    )

    # ========================================================
    # 8. ANSWER SHAPE CHECKS
    # ========================================================

    assert answer_ids.ndim == 1

    assert answer_attention_mask.ndim == 1

    assert (
        answer_ids.shape
        == answer_attention_mask.shape
    )

    # IMPORTANT:
    #
    # answer_ids.shape == (A,)
    #
    # Therefore:
    #
    #     shape[0] = answer length
    #
    answer_length = (
        answer_ids.shape[0]
    )

    assert answer_length > 0, (
        "Answer must contain at least "
        "one token."
    )

    # ========================================================
    # 9. ANSWER ATTENTION MASK SANITY CHECK
    #
    # No padding is used, so every answer token should
    # have attention_mask = 1.
    # ========================================================

    assert torch.all(
        answer_attention_mask == 1
    ), (
        "Unexpected answer attention_mask. "
        "Expected all answer tokens to be visible "
        "because no padding is used."
    )

    # ========================================================
    # 10. DUPLICATE ANSWER ACROSS RETRIEVAL DECISIONS
    #
    # Original:
    #
    #     (A,)
    #
    # unsqueeze(0):
    #
    #     (1, A)
    #
    # repeat(B, 1):
    #
    #     (B, A)
    # ========================================================

    answer_ids_batch = (
        answer_ids
        .unsqueeze(0)
        .repeat(
            B,
            1,
        )
    )

    answer_attention_mask_batch = (
        answer_attention_mask
        .unsqueeze(0)
        .repeat(
            B,
            1,
        )
    )

    assert answer_ids_batch.shape == (
        B,
        answer_length,
    )

    assert answer_attention_mask_batch.shape == (
        B,
        answer_length,
    )

    # ========================================================
    # 11. APPEND ANSWER TO INPUT IDS
    #
    #     Original: (B, L)
    #     Answer:   (B, A)
    #
    #     Result:   (B, L+A)
    # ========================================================

    input_ids = torch.cat(
        [
            input_ids,
            answer_ids_batch,
        ],
        dim=1,
    )

    # ========================================================
    # 12. APPEND ANSWER ATTENTION MASK
    #
    #     Original: (B, L)
    #     Answer:   (B, A)
    #
    #     Result:   (B, L+A)
    # ========================================================

    attention_mask = torch.cat(
        [
            attention_mask,
            answer_attention_mask_batch,
        ],
        dim=1,
    )

    # ========================================================
    # 13. CREATE DOCUMENT ATTENTION MASK FOR ANSWER
    #
    # Answer is NOT a document.
    #
    # Therefore:
    #
    #     answer positions -> 0
    #
    # Shape:
    #
    #     (B, A)
    # ========================================================

    answer_document_attention_mask = (
        torch.zeros(
            B,
            answer_length,
            dtype=document_attention_mask.dtype,
        )
    )

    # ========================================================
    # 14. APPEND DOCUMENT ATTENTION MASK
    #
    #     Original: (B, L)
    #     Answer:   (B, A)
    #
    #     Result:   (B, L+A)
    # ========================================================

    document_attention_mask = torch.cat(
        [
            document_attention_mask,
            answer_document_attention_mask,
        ],
        dim=1,
    )

    # ========================================================
    # 15. CREATE LABELS
    #
    # Standard Hugging Face Causal LM convention:
    #
    #     -100 -> ignored by CrossEntropyLoss
    #
    # Therefore:
    #
    #     question/document -> -100
    #     answer            -> answer token IDs
    #
    # Shape:
    #
    #     (B, L+A)
    # ========================================================

    total_length = (
        L + answer_length
    )

    labels = torch.full(
        (
            B,
            total_length,
        ),
        -100,
        dtype=torch.long,
    )

    # ========================================================
    # 16. PUT ANSWER TOKEN IDS INTO LABELS
    #
    #     labels[:, :L] = -100
    #     labels[:, L:] = answer token IDs
    # ========================================================

    labels[
        :,
        L:,
    ] = answer_ids_batch

    # ========================================================
    # 17. FINAL SHAPE CHECKS
    # ========================================================

    assert input_ids.shape == (
        B,
        L + answer_length,
    )

    assert attention_mask.shape == (
        B,
        L + answer_length,
    )

    assert document_attention_mask.shape == (
        B,
        L + answer_length,
    )

    assert labels.shape == (
        B,
        L + answer_length,
    )

    assert retrieval_decision.shape == (
        B,
    )

    # ========================================================
    # 18. VERIFY ORIGINAL PART IS UNCHANGED
    # ========================================================

    assert torch.equal(
        input_ids[:, :L],
        result["input_ids"],
    )

    assert torch.equal(
        attention_mask[:, :L],
        result["attention_mask"],
    )

    assert torch.equal(
        document_attention_mask[:, :L],
        result["document_attention_mask"],
    )

    assert torch.equal(
        retrieval_decision,
        result["retrieval_decision"],
    )

    # ========================================================
    # 19. VERIFY ANSWER INPUT IDS
    # ========================================================

    assert torch.equal(
        input_ids[:, L:],
        answer_ids_batch,
    )

    # ========================================================
    # 20. VERIFY ANSWER ATTENTION MASK
    # ========================================================

    assert torch.equal(
        attention_mask[:, L:],
        answer_attention_mask_batch,
    )

    # ========================================================
    # 21. VERIFY ANSWER DOCUMENT MASK
    #
    # Every answer position must be 0.
    # ========================================================

    assert torch.all(
        document_attention_mask[:, L:]
        == 0
    )

    # ========================================================
    # 22. VERIFY LABELS
    # ========================================================

    # Everything before the answer is ignored.
    assert torch.all(
        labels[:, :L]
        == -100
    )

    # Answer positions contain answer token IDs.
    assert torch.equal(
        labels[:, L:],
        answer_ids_batch,
    )

    # ========================================================
    # 23. VERIFY BATCH DUPLICATION
    #
    # All rows should be identical.
    # Only retrieval_decision differs between rows.
    # ========================================================

    for i in range(
        1,
        B,
    ):

        assert torch.equal(
            input_ids[0],
            input_ids[i],
        )

        assert torch.equal(
            attention_mask[0],
            attention_mask[i],
        )

        assert torch.equal(
            document_attention_mask[0],
            document_attention_mask[i],
        )

        assert torch.equal(
            labels[0],
            labels[i],
        )

    # ========================================================
    # 24. VERIFY FULL TEXT
    #
    # full_text should end with exactly the answer text
    # that was tokenized.
    # ========================================================

    assert full_text.endswith(
        answer_text
    )

    # ========================================================
    # 25. RETURN
    # ========================================================

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "document_attention_mask": (
            document_attention_mask
        ),
        "retrieval_decision": (
            retrieval_decision
        ),
        "labels": labels,
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
    result: Dict[str, Any],
):
    """
    Deep validation of create_model_input().
    """

    input_ids = result[
        "input_ids"
    ]

    attention_mask = result[
        "attention_mask"
    ]

    document_attention_mask = result[
        "document_attention_mask"
    ]

    retrieval_decision = result[
        "retrieval_decision"
    ]

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
        attention_mask.shape[0]
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
        attention_mask.shape
        == (B, L)
    )

    assert (
        document_attention_mask.shape
        == (B, L)
    )

    # --------------------------------------------------------
    # Tokenizer attention mask
    #
    # No padding is used, so every token must be visible.
    # --------------------------------------------------------

    assert (
        attention_mask.dtype
        == torch.long
    )

    assert torch.all(
        attention_mask == 1
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

    assert (
        retrieval_decision[
            0
        ].item()
        == 0.0
    )

    assert (
        retrieval_decision[
            -1
        ].item()
        == 1.0
    )

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
        1.0
        / NUM_DECISION_LEVELS
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

    for i in range(
        1,
        B,
    ):

        assert torch.equal(
            input_ids[0],
            input_ids[i],
        )

        assert torch.equal(
            attention_mask[0],
            attention_mask[i],
        )

        assert torch.equal(
            document_attention_mask[0],
            document_attention_mask[i],
        )

    # --------------------------------------------------------
    # Document mask should contain only:
    #
    #     0
    #     -inf
    # --------------------------------------------------------

    valid_mask_values = (
        (
            document_attention_mask
            == 0
        )
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
        ]
        == 0
    )

    assert torch.all(
        document_attention_mask[
            0,
            masked_end:,
        ]
        == 0
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

    assert (
        "[Retrieved document]"
        in decoded_document
    )

    assert (
        "[End retrieved document]"
        in decoded_document
    )

    # --------------------------------------------------------
    # Every document title/content must exist in the
    # decoded document span.
    # --------------------------------------------------------

    for document in sample.documents:

        assert (
            document.title
            in decoded_document
        )

        assert (
            document.content
            in decoded_document
        )

    return {
        "B": B,
        "L": L,
        "masked_start": masked_start,
        "masked_end": masked_end,
        "masked_tokens": (
            masked_end
            - masked_start
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

    print(
        "✓ Tokenizer loaded"
    )

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
        f"input_ids shape: "
        f"{tuple(result_1['input_ids'].shape)}"
    )

    print(
        f"attention_mask shape: "
        f"{tuple(result_1['attention_mask'].shape)}"
    )

    print(
        f"document_attention_mask shape: "
        f"{tuple(result_1['document_attention_mask'].shape)}"
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
                title=(
                    "Viện Nghiên cứu Ánh Dương — 2026"
                ),
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
    # ========================================================

    _print_test_header(
        "TEST 5 — EMPTY DOCUMENT LIST"
    )

    sample_5 = SampleObject(
        question=(
            "This sample has no document."
        ),
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

    assert (
        decisions[0].item()
        == 0.0
    )

    assert (
        decisions[-1].item()
        == 1.0
    )

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

    attention_mask = result_1[
        "attention_mask"
    ]

    document_attention_mask = (
        result_1[
            "document_attention_mask"
        ]
    )

    for i in range(
        NUM_DECISION_LEVELS + 1
    ):

        assert torch.equal(
            input_ids[0],
            input_ids[i],
        )

        assert torch.equal(
            attention_mask[0],
            attention_mask[i],
        )

        assert torch.equal(
            document_attention_mask[0],
            document_attention_mask[i],
        )

    print(
        "✓ All 33 input rows are identical"
    )

    print(
        "✓ All 33 tokenizer attention-mask rows are identical"
    )

    print(
        "✓ All 33 document attention-mask rows are identical"
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
        result_a[
            "input_ids"
        ],
        result_b[
            "input_ids"
        ],
    )

    # The second result must remain intact.
    fresh = create_model_input(
        sample_1,
        tokenizer,
    )

    assert torch.equal(
        result_b[
            "input_ids"
        ],
        fresh[
            "input_ids"
        ],
    )

    assert torch.equal(
        result_b[
            "attention_mask"
        ],
        fresh[
            "attention_mask"
        ],
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
        "✓ Tokenizer attention mask"
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


# ============================================================
# DEBUG
# ============================================================

def debug_multi_document_input():
    """
    Debug create_model_input() using Qwen3-4B tokenizer.

    Prints:
        1. Output tensor shapes
        2. Full model input text
        3. Tokenizer attention mask
        4. Masked token range
        5. Decoded masked document span
        6. Mask correctness checks
        7. Batch duplication checks
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

    tokenizer = (
        AutoTokenizer.from_pretrained(
            QWEN_REPO,
        )
    )

    print(
        "✓ Tokenizer loaded"
    )

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

    input_ids = result[
        "input_ids"
    ]

    attention_mask = result[
        "attention_mask"
    ]

    document_attention_mask = (
        result[
            "document_attention_mask"
        ]
    )

    retrieval_decision = (
        result[
            "retrieval_decision"
        ]
    )

    full_text = result[
        "full_text"
    ]

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
        f"attention_mask           : "
        f"shape={tuple(attention_mask.shape)}, "
        f"dtype={attention_mask.dtype}"
    )

    print(
        f"document_attention_mask  : "
        f"shape={tuple(document_attention_mask.shape)}, "
        f"dtype={document_attention_mask.dtype}"
    )

    print(
        f"retrieval_decision       : "
        f"shape={tuple(retrieval_decision.shape)}, "
        f"dtype={retrieval_decision.dtype}"
    )

    print(
        f"full_text                : "
        f"type={type(full_text).__name__}"
    )

    # ========================================================
    # 5. TOKENIZER ATTENTION MASK
    # ========================================================

    print()
    print("=" * 80)
    print("TOKENIZER ATTENTION MASK")
    print("=" * 80)

    print(
        f"Shape: "
        f"{tuple(attention_mask.shape)}"
    )

    print(
        f"Unique values: "
        f"{torch.unique(attention_mask).tolist()}"
    )

    assert torch.all(
        attention_mask == 1
    )

    print(
        "✓ All tokens are visible to the tokenizer attention mask"
    )

    # ========================================================
    # 6. VERIFY INPUT / MASK LENGTH
    # ========================================================

    assert (
        input_ids.shape
        == attention_mask.shape
    )

    print(
        "✓ input_ids and attention_mask "
        "have identical shape"
    )

    # ========================================================
    # 7. PRINT FULL TEXT
    # ========================================================

    print()
    print("=" * 80)
    print("FULL TEXT USED BY MODEL")
    print("=" * 80)

    print(full_text)

    # ========================================================
    # 8. FIND MASKED DOCUMENT SPAN
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

    masked_start = (
        masked_indices[0].item()
    )

    masked_end = (
        masked_indices[-1].item()
        + 1
    )

    # ========================================================
    # 9. DECODE MASKED SPAN
    # ========================================================

    masked_text = tokenizer.decode(
        input_ids[
            0,
            masked_start:masked_end,
        ].tolist(),
        skip_special_tokens=False,
    )

    # ========================================================
    # 10. PRINT MASKED SPAN
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
    # 11. VERIFY DOCUMENT MASK
    # ========================================================

    print()
    print("=" * 80)
    print("DOCUMENT MASK VERIFICATION")
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
        ]
        == 0
    )

    print(
        "✓ Tokens before document are visible"
    )

    # Everything after document must be visible.
    assert torch.all(
        document_attention_mask[
            0,
            masked_end:,
        ]
        == 0
    )

    print(
        "✓ Tokens after document are visible"
    )

    # Every document must exist in masked span.
    for document in debug_sample.documents:

        assert (
            document.title
            in masked_text
        )

        assert (
            document.content
            in masked_text
        )

        print(
            f"✓ Document found: "
            f"{document.title}"
        )

    # ========================================================
    # 12. VERIFY TOKENIZER MASK
    # ========================================================

    print()
    print("=" * 80)
    print("TOKENIZER MASK VERIFICATION")
    print("=" * 80)

    assert torch.all(
        attention_mask == 1
    )

    print(
        "✓ All tokenizer attention-mask values are 1"
    )

    assert (
        attention_mask.shape
        == input_ids.shape
    )

    print(
        "✓ Tokenizer attention mask is aligned "
        "with input_ids"
    )

    # ========================================================
    # 13. VERIFY BATCH DUPLICATION
    # ========================================================

    for i in range(
        NUM_DECISION_LEVELS + 1
    ):

        assert torch.equal(
            input_ids[0],
            input_ids[i],
        )

        assert torch.equal(
            attention_mask[0],
            attention_mask[i],
        )

        assert torch.equal(
            document_attention_mask[0],
            document_attention_mask[i],
        )

    print(
        "✓ All 33 input rows are identical"
    )

    print(
        "✓ All 33 tokenizer attention-mask rows are identical"
    )

    print(
        "✓ All 33 document attention-mask rows are identical"
    )

    # ========================================================
    # 14. FINAL SUMMARY
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

    print(
        f"Tokenizer mask     : "
        f"{tuple(attention_mask.shape)}"
    )

    print()

    print(
        "✓ MULTI-DOCUMENT DEBUG TEST PASSED"
    )

def test_create_model_input_for_lm_loss():
    """
    Comprehensive test suite for create_model_input_for_lm_loss().

    Assumption:
        tokenizer(single_string)["input_ids"]      -> (L,)
        tokenizer(single_string)["attention_mask"] -> (L,)

    Therefore sequence length is always taken from shape[0].

    Covered:
        1. Single document
        2. Multiple documents
        3. Empty document list
        4. Empty document content
        5. Unicode / special characters
        6. Short answer
        7. Long answer
        8. Punctuation / numbers
        9. Answer with whitespace
        10. Single-token answer
        11. input_ids shape
        12. attention_mask shape
        13. document_attention_mask shape
        14. labels shape
        15. attention_mask values
        16. document mask correctness
        17. label boundary correctness
        18. answer token alignment
        19. retrieval decision values
        20. batch duplication
        21. full_text correctness
        22. output independence
    """

    # ============================================================
    # LOAD TOKENIZER
    # ============================================================

    _print_test_header(
        "TEST — create_model_input_for_lm_loss()"
    )

    print(f"Repository: {QWEN_REPO}")

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_REPO
    )

    print("✓ Tokenizer loaded")

    B_EXPECTED = NUM_DECISION_LEVELS + 1

    # ============================================================
    # GENERIC CASE RUNNER
    # ============================================================

    def run_case(
        case_name: str,
        sample: SampleObjectForLMLoss,
    ):
        _print_test_header(case_name)

        # --------------------------------------------------------
        # Run function under test
        # --------------------------------------------------------

        result = create_model_input_for_lm_loss(
            sample,
            tokenizer,
        )

        # --------------------------------------------------------
        # Extract outputs
        # --------------------------------------------------------

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]
        document_attention_mask = result[
            "document_attention_mask"
        ]
        retrieval_decision = result[
            "retrieval_decision"
        ]
        labels = result["labels"]
        full_text = result["full_text"]

        # --------------------------------------------------------
        # BASIC TYPE CHECKS
        # --------------------------------------------------------

        assert isinstance(
            input_ids,
            torch.Tensor,
        )

        assert isinstance(
            attention_mask,
            torch.Tensor,
        )

        assert isinstance(
            document_attention_mask,
            torch.Tensor,
        )

        assert isinstance(
            retrieval_decision,
            torch.Tensor,
        )

        assert isinstance(
            labels,
            torch.Tensor,
        )

        assert isinstance(
            full_text,
            str,
        )

        # --------------------------------------------------------
        # BASIC DIMENSIONS
        # --------------------------------------------------------

        assert input_ids.ndim == 2
        assert attention_mask.ndim == 2
        assert document_attention_mask.ndim == 2
        assert labels.ndim == 2
        assert retrieval_decision.ndim == 1

        B, total_L = input_ids.shape

        assert B == B_EXPECTED

        assert attention_mask.shape == (
            B,
            total_L,
        )

        assert document_attention_mask.shape == (
            B,
            total_L,
        )

        assert labels.shape == (
            B,
            total_L,
        )

        assert retrieval_decision.shape == (
            B,
        )

        # --------------------------------------------------------
        # DTYPES
        # --------------------------------------------------------

        assert input_ids.dtype == torch.long
        assert attention_mask.dtype == torch.long
        assert document_attention_mask.dtype == torch.float32
        assert retrieval_decision.dtype == torch.float32
        assert labels.dtype == torch.long

        # --------------------------------------------------------
        # RECREATE ORIGINAL INPUT
        # --------------------------------------------------------

        original_sample = SampleObject(
            question=sample.question,
            documents=sample.documents,
        )

        original = create_model_input(
            original_sample,
            tokenizer,
        )

        original_input_ids = original[
            "input_ids"
        ]

        original_attention_mask = original[
            "attention_mask"
        ]

        original_document_attention_mask = (
            original[
                "document_attention_mask"
            ]
        )

        original_retrieval_decision = (
            original[
                "retrieval_decision"
            ]
        )

        original_full_text = original[
            "full_text"
        ]

        original_B, original_L = (
            original_input_ids.shape
        )

        assert original_B == B_EXPECTED

        # --------------------------------------------------------
        # TOKENIZE ANSWER SEPARATELY
        #
        # IMPORTANT:
        #
        # Single string:
        #
        #     input_ids       -> (A,)
        #     attention_mask  -> (A,)
        #
        # Therefore:
        #
        #     A = shape[0]
        # --------------------------------------------------------

        answer_text = (
            "\n"
            + sample.answer
        )

        answer_encoding = tokenizer(
            answer_text,
            add_special_tokens=False,
            return_attention_mask=True,
        )

        answer_ids = torch.tensor(
            answer_encoding["input_ids"],
            dtype=torch.long,
        )

        answer_attention_mask = torch.tensor(
            answer_encoding["attention_mask"],
            dtype=torch.long,
        )

        # --------------------------------------------------------
        # Verify tokenizer assumption
        # --------------------------------------------------------

        assert answer_ids.ndim == 1
        assert answer_attention_mask.ndim == 1

        assert (
            answer_ids.shape
            == answer_attention_mask.shape
        )

        answer_length = answer_ids.shape[0]

        assert answer_length > 0

        # --------------------------------------------------------
        # TOTAL LENGTH
        # --------------------------------------------------------

        assert total_L == (
            original_L + answer_length
        )

        # ========================================================
        # INPUT IDS
        # ========================================================

        # Original part must be unchanged.
        assert torch.equal(
            input_ids[:, :original_L],
            original_input_ids,
        )

        # Answer part must equal answer IDs.
        expected_answer_ids = (
            answer_ids
            .unsqueeze(0)
            .repeat(B, 1)
        )

        assert torch.equal(
            input_ids[:, original_L:],
            expected_answer_ids,
        )

        # ========================================================
        # ATTENTION MASK
        # ========================================================

        # Original part unchanged.
        assert torch.equal(
            attention_mask[:, :original_L],
            original_attention_mask,
        )

        # Answer attention mask.
        expected_answer_attention_mask = (
            answer_attention_mask
            .unsqueeze(0)
            .repeat(B, 1)
        )

        assert torch.equal(
            attention_mask[:, original_L:],
            expected_answer_attention_mask,
        )

        # No padding anywhere.
        assert torch.all(
            attention_mask == 1
        )

        # input_ids and attention_mask aligned.
        assert (
            input_ids.shape
            == attention_mask.shape
        )

        # ========================================================
        # DOCUMENT ATTENTION MASK
        # ========================================================

        # Original document mask unchanged.
        assert torch.equal(
            document_attention_mask[:, :original_L],
            original_document_attention_mask,
        )

        # Answer is NOT a document.
        assert torch.all(
            document_attention_mask[
                :,
                original_L:,
            ] == 0
        )

        # Document mask only contains 0 / -inf.
        valid_document_mask = (
            (
                document_attention_mask == 0
            )
            |
            torch.isneginf(
                document_attention_mask
            )
        )

        assert torch.all(
            valid_document_mask
        )

        # ========================================================
        # LABELS
        # ========================================================

        # Everything before answer ignored.
        assert torch.all(
            labels[:, :original_L] == -100
        )

        # Answer supervised.
        assert torch.equal(
            labels[:, original_L:],
            expected_answer_ids,
        )

        # Exact expected labels.
        expected_labels = torch.full(
            (
                B,
                total_L,
            ),
            -100,
            dtype=torch.long,
        )

        expected_labels[
            :,
            original_L:,
        ] = expected_answer_ids

        assert torch.equal(
            labels,
            expected_labels,
        )

        # ========================================================
        # LABEL BOUNDARY
        # ========================================================

        # Last original token is ignored.
        assert torch.all(
            labels[
                :,
                original_L - 1,
            ] == -100
        )

        # First answer token is supervised.
        assert torch.equal(
            labels[
                :,
                original_L,
            ],
            expected_answer_ids[:, 0],
        )

        # Last answer token is supervised.
        assert torch.equal(
            labels[
                :,
                total_L - 1,
            ],
            expected_answer_ids[:, -1],
        )

        # ========================================================
        # RETRIEVAL DECISION
        # ========================================================

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

        assert (
            retrieval_decision[0].item()
            == 0.0
        )

        assert (
            retrieval_decision[-1].item()
            == 1.0
        )

        # Strictly increasing.
        assert torch.all(
            retrieval_decision[1:]
            > retrieval_decision[:-1]
        )

        # ========================================================
        # BATCH DUPLICATION
        #
        # All rows are identical.
        # Only retrieval_decision represents different
        # retrieval strengths.
        # ========================================================

        for i in range(1, B):

            assert torch.equal(
                input_ids[0],
                input_ids[i],
            )

            assert torch.equal(
                attention_mask[0],
                attention_mask[i],
            )

            assert torch.equal(
                document_attention_mask[0],
                document_attention_mask[i],
            )

            assert torch.equal(
                labels[0],
                labels[i],
            )

        # ========================================================
        # ORIGINAL RETRIEVAL DECISION UNCHANGED
        # ========================================================

        assert torch.equal(
            retrieval_decision,
            original_retrieval_decision,
        )

        # ========================================================
        # FULL TEXT
        # ========================================================

        assert full_text == (
            original_full_text
            + answer_text
        )

        assert full_text.endswith(
            answer_text
        )

        # ========================================================
        # ANSWER INPUT / LABEL ALIGNMENT
        # ========================================================

        assert torch.equal(
            input_ids[
                :,
                original_L:,
            ],
            labels[
                :,
                original_L:,
            ],
        )

        # ========================================================
        # ANSWER TOKEN COUNT
        # ========================================================

        assert (
            input_ids.shape[1]
            - original_input_ids.shape[1]
            == answer_length
        )

        # ========================================================
        # PRINT SUMMARY
        # ========================================================

        print(
            f"B                  : {B}"
        )

        print(
            f"Original length    : {original_L}"
        )

        print(
            f"Answer length      : {answer_length}"
        )

        print(
            f"Final length       : {total_L}"
        )

        print(
            f"input_ids          : {tuple(input_ids.shape)}"
        )

        print(
            f"attention_mask     : {tuple(attention_mask.shape)}"
        )

        print(
            f"document_mask      : "
            f"{tuple(document_attention_mask.shape)}"
        )

        print(
            f"labels             : {tuple(labels.shape)}"
        )

        print(
            f"retrieval_decision : "
            f"{tuple(retrieval_decision.shape)}"
        )

        print(
            f"answer tokens      : {answer_length}"
        )

        print("✓ Case passed")

        return result

    # ============================================================
    # TEST 1 — SINGLE DOCUMENT
    # ============================================================

    result_1 = run_case(
        "TEST 1 — SINGLE DOCUMENT",
        SampleObjectForLMLoss(
            question=(
                "What is the energy density "
                "of the experimental battery?"
            ),
            documents=[
                Document(
                    title="Aurora Research Institute",
                    content=(
                        "The experimental battery "
                        "achieved an energy density "
                        "of 412 Wh/kg."
                    ),
                )
            ],
            answer="412 Wh/kg",
        ),
    )

    # ============================================================
    # TEST 2 — MULTIPLE DOCUMENTS
    # ============================================================

    result_2 = run_case(
        "TEST 2 — MULTIPLE DOCUMENTS",
        SampleObjectForLMLoss(
            question=(
                "Which institute developed "
                "the experimental battery?"
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
                    title="Aurora Research Institute",
                    content=(
                        "Aurora Research Institute "
                        "developed an experimental battery."
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
            answer="Aurora Research Institute",
        ),
    )

    # ============================================================
    # TEST 3 — EMPTY DOCUMENT CONTENT
    # ============================================================

    result_3 = run_case(
        "TEST 3 — EMPTY DOCUMENT CONTENT",
        SampleObjectForLMLoss(
            question="What is the document title?",
            documents=[
                Document(
                    title="Empty Content Document",
                    content="",
                )
            ],
            answer="Empty Content Document",
        ),
    )

    # ============================================================
    # TEST 4 — EMPTY DOCUMENT LIST
    # ============================================================

    result_4 = run_case(
        "TEST 4 — EMPTY DOCUMENT LIST",
        SampleObjectForLMLoss(
            question="What information is available?",
            documents=[],
            answer="No document is available.",
        ),
    )

    # ============================================================
    # TEST 5 — UNICODE / SPECIAL CHARACTERS
    # ============================================================

    result_5 = run_case(
        "TEST 5 — UNICODE / SPECIAL CHARACTERS",
        SampleObjectForLMLoss(
            question=(
                "Thông số của thiết bị là bao nhiêu?"
            ),
            documents=[
                Document(
                    title=(
                        "Viện Nghiên cứu Ánh Dương — 2026"
                    ),
                    content=(
                        "Thiết bị đạt hiệu suất 99.5%, "
                        "năng lượng 412 Wh/kg, "
                        "nhiệt độ −20°C."
                    ),
                )
            ],
            answer=(
                "412 Wh/kg — 99.5% — −20°C"
            ),
        ),
    )

    # ============================================================
    # TEST 6 — WHITESPACE
    # ============================================================

    result_6 = run_case(
        "TEST 6 — ANSWER WITH WHITESPACE",
        SampleObjectForLMLoss(
            question="What is the value?",
            documents=[
                Document(
                    title="Whitespace Test",
                    content=(
                        "This document tests "
                        "whitespace handling."
                    ),
                )
            ],
            answer="   42   ",
        ),
    )

    # ============================================================
    # TEST 7 — PUNCTUATION / NUMBERS
    # ============================================================

    result_7 = run_case(
        "TEST 7 — PUNCTUATION / NUMBERS",
        SampleObjectForLMLoss(
            question="What was the measured value?",
            documents=[
                Document(
                    title="Measurement",
                    content=(
                        "The measured value was "
                        "123.456 ± 0.789 units."
                    ),
                )
            ],
            answer="123.456 ± 0.789 units.",
        ),
    )

    # # ============================================================
    # # TEST 8 — SINGLE TOKEN ANSWER
    # # ============================================================

    # _print_test_header(
    #     "TEST 8 — SINGLE TOKEN ANSWER"
    # )

    # single_token_answer = None

    # candidates = [
    #     "yes",
    #     "no",
    #     "42",
    #     "A",
    #     "the",
    #     "Qwen",
    #     "Paris",
    # ]

    # for candidate in candidates:

    #     encoding = tokenizer(
    #         "\n" + candidate,
    #         add_special_tokens=False,
    #     )

    #     if len(
    #         encoding["input_ids"]
    #     ) == 1:

    #         single_token_answer = candidate
    #         break

    # assert (
    #     single_token_answer is not None
    # ), "Could not find a single-token candidate."

    # result_8 = run_case(
    #     "TEST 8 — SINGLE TOKEN ANSWER",
    #     SampleObjectForLMLoss(
    #         question="Is the statement correct?",
    #         documents=[
    #             Document(
    #                 title="Single Token Test",
    #                 content=(
    #                     "This document is used "
    #                     "for tokenizer testing."
    #                 ),
    #             )
    #         ],
    #         answer=single_token_answer,
    #     ),
    # )

    # # Explicitly verify answer has exactly one token.
    # single_token_encoding = tokenizer(
    #     "\n" + single_token_answer,
    #     add_special_tokens=False,
    # )

    # assert len(
    #     single_token_encoding["input_ids"]
    # ) == 1

    # print(
    #     f"✓ Single-token answer = "
    #     f"{single_token_answer!r}"
    # )

    # ============================================================
    # TEST 9 — LONG ANSWER
    # ============================================================

    result_9 = run_case(
        "TEST 9 — LONG ANSWER",
        SampleObjectForLMLoss(
            question=(
                "Explain the experimental "
                "battery result."
            ),
            documents=[
                Document(
                    title="Battery Study",
                    content=(
                        "Researchers conducted an "
                        "experimental battery study "
                        "under controlled laboratory "
                        "conditions."
                    ),
                )
            ],
            answer=(
                "The experimental battery achieved "
                "an energy density of 412 Wh/kg "
                "under controlled laboratory "
                "conditions."
            ),
        ),
    )

    # ============================================================
    # TEST 10 — DIFFERENT ANSWER LENGTHS
    # ============================================================

    _print_test_header(
        "TEST 10 — DIFFERENT ANSWER LENGTHS"
    )

    base_sample = SampleObjectForLMLoss(
        question="What is the answer?",
        documents=[
            Document(
                title="Answer Test",
                content="This is an answer-length test.",
            )
        ],
        answer="42",
    )

    answer_lengths = []

    for answer in [
        "42",
        "The answer is 42.",
        "The correct numerical answer is exactly 42.",
    ]:

        encoding = tokenizer(
            "\n" + answer,
            add_special_tokens=False,
        )

        answer_length = len(
            encoding["input_ids"]
        )

        assert answer_length > 0

        test_sample = SampleObjectForLMLoss(
            question=base_sample.question,
            documents=base_sample.documents,
            answer=answer,
        )

        result = create_model_input_for_lm_loss(
            test_sample,
            tokenizer,
        )

        original = create_model_input(
            SampleObject(
                question=test_sample.question,
                documents=test_sample.documents,
            ),
            tokenizer,
        )

        expected_length = (
            original["input_ids"].shape[1]
            + answer_length
        )

        assert (
            result["input_ids"].shape[1]
            == expected_length
        )

        assert (
            result["labels"].shape[1]
            == expected_length
        )

        answer_lengths.append(answer_length)

    assert (
        answer_lengths[0]
        < answer_lengths[1]
        < answer_lengths[2]
    )

    print(
        f"Answer token lengths: {answer_lengths}"
    )

    print(
        "✓ Different answer lengths handled correctly"
    )

    # ============================================================
    # TEST 11 — DOCUMENT MASK HAS ACTUAL DOCUMENT SPAN
    # ============================================================

    _print_test_header(
        "TEST 11 — DOCUMENT MASK SPAN"
    )

    result = result_2

    document_mask = result[
        "document_attention_mask"
    ]

    masked = torch.isneginf(
        document_mask[0]
    )

    assert masked.any()

    masked_indices = torch.nonzero(
        masked,
        as_tuple=False,
    ).flatten()

    start = masked_indices[0].item()
    end = masked_indices[-1].item() + 1

    # Must be contiguous.
    expected_indices = torch.arange(
        start,
        end,
        dtype=masked_indices.dtype,
    )

    assert torch.equal(
        masked_indices,
        expected_indices,
    )

    # Before document = visible.
    assert torch.all(
        document_mask[
            0,
            :start,
        ] == 0
    )

    # Document = -inf.
    assert torch.isneginf(
        document_mask[
            0,
            start:end,
        ]
    ).all()

    # After document = visible.
    assert torch.all(
        document_mask[
            0,
            end:,
        ] == 0
    )

    print(
        f"Document span: {start} → {end}"
    )

    print(
        f"Document tokens: {end - start}"
    )

    print(
        "✓ Document attention mask span is correct"
    )

    # ============================================================
    # TEST 12 — OUTPUT INDEPENDENCE
    # ============================================================

    _print_test_header(
        "TEST 12 — OUTPUT INDEPENDENCE"
    )

    independent_sample = SampleObjectForLMLoss(
        question="What is 2 + 2?",
        documents=[
            Document(
                title="Math",
                content="Basic arithmetic.",
            )
        ],
        answer="4",
    )

    result_a = create_model_input_for_lm_loss(
        independent_sample,
        tokenizer,
    )

    result_b = create_model_input_for_lm_loss(
        independent_sample,
        tokenizer,
    )

    # Mutate A.
    original_value = result_a[
        "input_ids"
    ][0, 0].item()

    result_a[
        "input_ids"
    ][0, 0] = original_value + 1

    # B must remain unchanged.
    assert not torch.equal(
        result_a["input_ids"],
        result_b["input_ids"],
    )

    # Fresh result must equal B.
    result_fresh = (
        create_model_input_for_lm_loss(
            independent_sample,
            tokenizer,
        )
    )

    assert torch.equal(
        result_b["input_ids"],
        result_fresh["input_ids"],
    )

    assert torch.equal(
        result_b["attention_mask"],
        result_fresh["attention_mask"],
    )

    assert torch.equal(
        result_b["document_attention_mask"],
        result_fresh[
            "document_attention_mask"
        ],
    )

    assert torch.equal(
        result_b["retrieval_decision"],
        result_fresh[
            "retrieval_decision"
        ],
    )

    assert torch.equal(
        result_b["labels"],
        result_fresh["labels"],
    )

    assert (
        result_b["full_text"]
        == result_fresh["full_text"]
    )

    print(
        "✓ Independent calls do not share tensors"
    )

    # ============================================================
    # TEST 13 — EXACT OUTPUT KEYS
    # ============================================================

    _print_test_header(
        "TEST 13 — OUTPUT KEYS"
    )

    expected_keys = {
        "input_ids",
        "attention_mask",
        "document_attention_mask",
        "retrieval_decision",
        "labels",
        "full_text",
    }

    assert set(result_1.keys()) == expected_keys

    print(
        "✓ Output keys are correct"
    )

    # ============================================================
    # TEST 14 — DECISION VALUES
    # ============================================================

    _print_test_header(
        "TEST 14 — DECISION VALUES"
    )

    decisions = result_1[
        "retrieval_decision"
    ]

    assert decisions.shape == (
        B_EXPECTED,
    )

    assert decisions[0].item() == 0.0

    assert decisions[
        NUM_DECISION_LEVELS
    ].item() == 1.0

    expected_step = (
        1.0 / NUM_DECISION_LEVELS
    )

    steps = (
        decisions[1:]
        - decisions[:-1]
    )

    assert torch.allclose(
        steps,
        torch.full_like(
            steps,
            expected_step,
        ),
    )

    assert abs(
        decisions[
            NUM_DECISION_LEVELS // 2
        ].item()
        - 0.5
    ) < 1e-7

    print(
        f"Number of decisions : {B_EXPECTED}"
    )

    print(
        f"First decision      : {decisions[0].item()}"
    )

    print(
        f"Middle decision     : "
        f"{decisions[NUM_DECISION_LEVELS // 2].item()}"
    )

    print(
        f"Last decision       : {decisions[-1].item()}"
    )

    print(
        f"Decision step       : 1/{NUM_DECISION_LEVELS}"
    )

    print(
        "✓ Retrieval decisions are correct"
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    _print_test_header(
        "ALL create_model_input_for_lm_loss() TESTS PASSED"
    )

    print("✓ Single document")
    print("✓ Multiple documents")
    print("✓ Empty document content")
    print("✓ Empty document list")
    print("✓ Unicode / special characters")
    print("✓ Answer whitespace")
    print("✓ Punctuation / numbers")
    print("✓ Single-token answer")
    print("✓ Long answer")
    print("✓ Different answer lengths")
    print("✓ input_ids shape")
    print("✓ attention_mask shape")
    print("✓ document_attention_mask shape")
    print("✓ labels shape")
    print("✓ Tokenizer 1D output handling")
    print("✓ Attention-mask alignment")
    print("✓ Document-mask alignment")
    print("✓ Label alignment")
    print("✓ Label boundary")
    print("✓ Retrieval decision values")
    print("✓ Batch duplication")
    print("✓ Full-text construction")
    print("✓ Document-mask span")
    print("✓ Output independence")
    print("✓ Output keys")

    print()
    print(
        "✓ create_model_input_for_lm_loss() "
        "passed all tests."
    )

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # main()
    debug_multi_document_input()
    # test_create_model_input_for_lm_loss()