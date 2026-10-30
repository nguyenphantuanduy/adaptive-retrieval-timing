from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step.
For each step, start by explaining your thought process.
If additional information is needed, the system will perform retrieval and return the top search results within <observation> and </observation>. Continue your reasoning using the retrieved information.
When you know the final answer, provide the final answer within <answer> and </answer> tags."""

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
def append_retrieval_observation(observation):
    """
    Create the text suffix used to inject retrieved evidence
    into the ongoing generation.

    Input:
        observation: raw retrieved context

    Output:
        A text string:

        <|im_end|>
        <|im_start|>user

        <observation>
        {observation}

        </observation><|im_end|>
        <|im_start|>assistant
    """

    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "\n"
        "<observation>\n\n"
        + observation
        + "\n\n"
        "</observation>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
def create_model_input(
    sample: SampleObject,
    tokenizer,
) -> Dict[str, Any]:
    """
    Create batched model inputs for one SampleObject.

    Input structure:

        SYSTEM PROMPT
            ↓
        USER QUESTION
            ↓
        chat template
            ↓
        <observation>
        Title: ...
        Content: ...
        ...
        </observation>

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

        document_attention_mask:
            shape = (33, L)

            Custom additive attention mask:

            0      -> normal / visible
            -inf   -> document span

        retrieval_decision:
            shape = (33,)

        full_text:
            str
    """

    # ========================================================
    # 1. BUILD SYSTEM + USER CHAT CONTEXT
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": sample.question,
        },
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

    document_parts = []

    for document in sample.documents:
        document_parts.append(
            f"Title: {document.title}\n"
            f"{document.content}"
        )

    observation = "\n\n".join(
        document_parts
    )

    document_text = append_retrieval_observation(
        observation
    )

    # System + user chat + documents.
    full_text = (
        base_text
        + document_text
    )

    # ========================================================
    # 3. DOCUMENT CHARACTER SPAN
    # ========================================================

    document_start = len(base_text)

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
    #     system + user chat
    #
    # and
    #
    #     observation / retrieved document section
    # ========================================================

    text_before = full_text[
        :document_start
    ]

    text_document = full_text[
        document_start:document_end
    ]

    # ========================================================
    # 5. TOKENIZE BEFORE / DOCUMENT SEPARATELY
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
    #     [system + user tokens] + [observation tokens]
    #
    # attention_mask:
    #
    #     [system + user mask] + [observation mask]
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
    # 8. DOCUMENT TOKEN SPAN
    # ========================================================

    document_start_token = len(
        before_ids
    )

    document_end_token = (
        document_start_token
        + len(document_ids)
    )

    # ========================================================
    # 9. DOCUMENT ATTENTION MASK
    #
    # This is NOT the tokenizer attention_mask.
    #
    # It is the custom additive mask used to control the
    # visibility of the observation/document span.
    #
    #     0      -> normal / visible
    #     -inf   -> document span
    # ========================================================

    document_attention_mask_single = torch.zeros(
        L,
        dtype=torch.float32,
    )

    document_attention_mask_single[
        document_start_token:document_end_token
    ] = float("-inf")

    # ========================================================
    # 10. DECISION VALUES
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
    # 11. DUPLICATE INPUT
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
    # 12. RETURN
    # ========================================================

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "document_attention_mask": document_attention_mask,
        "retrieval_decision": retrieval_decision,
        "full_text": full_text,
    }


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
    3. Append "\\n" + answer + EOS to full_text.
    4. Tokenize answer separately.
    5. Duplicate answer tokens across all retrieval decisions.
    6. Concatenate answer to input_ids.
    7. Concatenate answer attention_mask.
    8. Append zero document_attention_mask for answer.
    9. Create labels:
           question/document -> -100
           answer + EOS      -> answer token IDs.

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
        A = answer + EOS token length

    Notes
    -----
    - For a single string, tokenizer returns 1D sequences.
    - Therefore answer length is answer_ids.shape[0].
    - EOS token is explicitly appended to the supervised answer span.
    - "\\n" is intentionally included in the supervised answer span.
    - Labels are not shifted manually.
      Hugging Face causal LM models perform the shift internally.
    """

    # ========================================================
    # 1. CREATE ORIGINAL MODEL INPUT
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
    # 2. ORIGINAL SHAPES
    # ========================================================

    B, L = input_ids.shape

    # ========================================================
    # 3. BUILD ANSWER TEXT
    #
    # The newline and EOS are intentionally part of
    # the supervised answer span.
    # ========================================================

    answer_text = (
        "\n"
        + sample.answer
        + tokenizer.eos_token
    )

    # ========================================================
    # 4. APPEND ANSWER TO FULL TEXT
    # ========================================================

    full_text = (
        full_text
        + answer_text
    )

    # ========================================================
    # 5. TOKENIZE ANSWER
    #
    # For a single string:
    #
    #     input_ids      -> (A,)
    #     attention_mask -> (A,)
    #
    # EOS is explicitly included in answer_text.
    # There is NO batch dimension here.
    # ========================================================

    answer_encoding = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    # ========================================================
    # 6. CONVERT ANSWER TO TENSORS
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
    # 7. ANSWER LENGTH
    # ========================================================

    answer_length = (
        answer_ids.shape[0]
    )

    # ========================================================
    # 8. DUPLICATE ANSWER ACROSS RETRIEVAL DECISIONS
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

    # ========================================================
    # 9. APPEND ANSWER TO INPUT IDS
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
    # 10. APPEND ANSWER ATTENTION MASK
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
    # 11. CREATE DOCUMENT ATTENTION MASK FOR ANSWER
    #
    # Answer and EOS are NOT documents.
    #
    # Therefore:
    #
    #     answer positions -> 0
    #     EOS position     -> 0
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
    # 12. APPEND DOCUMENT ATTENTION MASK
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
    # 13. CREATE LABELS
    #
    # Standard Hugging Face Causal LM convention:
    #
    #     -100 -> ignored by CrossEntropyLoss
    #
    # Therefore:
    #
    #     question/document -> -100
    #     answer + EOS      -> answer token IDs
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
    # 14. PUT ANSWER TOKEN IDS INTO LABELS
    #
    #     labels[:, :L] = -100
    #     labels[:, L:] = answer token IDs
    #
    # EOS is included here and therefore is NOT masked.
    # ========================================================

    labels[
        :,
        L:,
    ] = answer_ids_batch

    # ========================================================
    # 15. RETURN
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


def test_create_model_input():
    # ========================================================
    # 1. LOAD TOKENIZER
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_REPO
    )

    # ========================================================
    # 2. CREATE DUMMY SAMPLE
    # ========================================================

    sample = SampleObject(
        question=(
            "Who wrote Common Sense and why was it important?"
        ),
        documents=[
            Document(
                title="Common Sense",
                content=(
                    "Common Sense was a pamphlet written "
                    "by Thomas Paine."
                ),
            ),
            Document(
                title="Thomas Paine",
                content=(
                    "Thomas Paine was an author and key figure "
                    "in the American Revolution. His work "
                    "Common Sense crystallized the rebellious "
                    "demand for independence from Great Britain."
                ),
            ),
        ],
    )

    # ========================================================
    # 3. RUN INPUT BUILDER
    # ========================================================

    result = create_model_input(
        sample=sample,
        tokenizer=tokenizer,
    )

    # ========================================================
    # 4. PRINT FULL TEXT
    # ========================================================

    print("=" * 80)
    print("FULL TEXT")
    print("=" * 80)

    print(result["full_text"])

    # ========================================================
    # 5. PRINT SHAPES
    # ========================================================

    print()
    print("=" * 80)
    print("TENSOR SHAPES")
    print("=" * 80)

    print(
        f"input_ids                  : "
        f"{tuple(result['input_ids'].shape)}"
    )

    print(
        f"attention_mask             : "
        f"{tuple(result['attention_mask'].shape)}"
    )

    print(
        f"document_attention_mask    : "
        f"{tuple(result['document_attention_mask'].shape)}"
    )

    print(
        f"retrieval_decision         : "
        f"{tuple(result['retrieval_decision'].shape)}"
    )

    # ========================================================
    # 6. PRINT RETRIEVAL DECISIONS
    # ========================================================

    print()
    print("=" * 80)
    print("RETRIEVAL DECISIONS")
    print("=" * 80)

    print(
        result["retrieval_decision"].tolist()
    )

    # ========================================================
    # 7. FIND DOCUMENT TOKENS FROM MASK
    #
    # document_attention_mask:
    #
    #     0      -> normal
    #     -inf   -> document
    # ========================================================

    document_mask = result[
        "document_attention_mask"
    ][0]

    document_token_indices = torch.where(
        torch.isneginf(document_mask)
    )[0]

    if len(document_token_indices) == 0:
        print()
        print(
            "No document tokens were masked."
        )
        return

    document_start_token = (
        document_token_indices[0].item()
    )

    document_end_token = (
        document_token_indices[-1].item() + 1
    )

    # ========================================================
    # 8. DECODE MASKED DOCUMENT TOKENS
    # ========================================================

    document_token_ids = result[
        "input_ids"
    ][0, document_start_token:document_end_token]

    masked_document_text = tokenizer.decode(
        document_token_ids,
        skip_special_tokens=False,
    )

    # ========================================================
    # 9. PRINT MASKED TEXT
    # ========================================================

    print()
    print("=" * 80)
    print("TEXT MASKED BY DOCUMENT ATTENTION MASK")
    print("=" * 80)

    print(masked_document_text)

    # ========================================================
    # 10. PRINT TOKEN SPAN
    # ========================================================

    print()
    print("=" * 80)
    print("MASKED TOKEN SPAN")
    print("=" * 80)

    print(
        f"Start token : {document_start_token}"
    )

    print(
        f"End token   : {document_end_token}"
    )

    print(
        f"Num tokens  : "
        f"{document_end_token - document_start_token}"
    )


def test_create_model_input_for_lm_loss():
    # ========================================================
    # 1. LOAD TOKENIZER
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_REPO
    )

    # ========================================================
    # 2. CREATE DUMMY SAMPLE
    # ========================================================

    sample = SampleObjectForLMLoss(
        question=(
            "Who wrote Common Sense and why was it important?"
        ),
        documents=[
            Document(
                title="Common Sense",
                content=(
                    "Common Sense was a pamphlet written "
                    "by Thomas Paine."
                ),
            ),
            Document(
                title="Thomas Paine",
                content=(
                    "Thomas Paine was an author and key figure "
                    "in the American Revolution. His work "
                    "Common Sense crystallized the rebellious "
                    "demand for independence from Great Britain."
                ),
            ),
        ],
        answer=(
            "Thomas Paine wrote Common Sense, and it "
            "helped build support for American independence."
        ),
    )

    # ========================================================
    # 3. RUN INPUT BUILDER
    # ========================================================

    result = create_model_input_for_lm_loss(
        sample=sample,
        tokenizer=tokenizer,
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

    labels = result[
        "labels"
    ]

    full_text = result[
        "full_text"
    ]

    # ========================================================
    # 4. PRINT FULL TEXT
    # ========================================================

    print()
    print("=" * 80)
    print("FULL TEXT")
    print("=" * 80)

    print(full_text)

    # ========================================================
    # 5. PRINT TENSOR SHAPES
    # ========================================================

    print()
    print("=" * 80)
    print("TENSOR SHAPES")
    print("=" * 80)

    print(
        f"input_ids                  : "
        f"{tuple(input_ids.shape)}"
    )

    print(
        f"attention_mask             : "
        f"{tuple(attention_mask.shape)}"
    )

    print(
        f"document_attention_mask    : "
        f"{tuple(document_attention_mask.shape)}"
    )

    print(
        f"retrieval_decision         : "
        f"{tuple(retrieval_decision.shape)}"
    )

    print(
        f"labels                     : "
        f"{tuple(labels.shape)}"
    )

    # ========================================================
    # 6. PRINT RETRIEVAL DECISIONS
    # ========================================================

    print()
    print("=" * 80)
    print("RETRIEVAL DECISIONS")
    print("=" * 80)

    print(
        retrieval_decision.tolist()
    )

    # ========================================================
    # 7. DOCUMENT MASK
    #
    # Use the first row because all rows have the same
    # document span.
    # ========================================================

    document_mask = (
        document_attention_mask[0]
    )

    document_token_indices = torch.where(
        torch.isneginf(document_mask)
    )[0]

    print()
    print("=" * 80)
    print("DOCUMENT ATTENTION MASK")
    print("=" * 80)

    if len(document_token_indices) == 0:

        print(
            "No document tokens are masked."
        )

    else:

        document_start_token = (
            document_token_indices[0].item()
        )

        document_end_token = (
            document_token_indices[-1].item()
            + 1
        )

        document_token_ids = input_ids[
            0,
            document_start_token:document_end_token,
        ]

        masked_document_text = tokenizer.decode(
            document_token_ids,
            skip_special_tokens=False,
        )

        print(
            f"Start token : {document_start_token}"
        )

        print(
            f"End token   : {document_end_token}"
        )

        print(
            f"Num tokens  : "
            f"{document_end_token - document_start_token}"
        )

        print()
        print("TEXT WITH document_attention_mask = -inf:")
        print("-" * 80)
        print(masked_document_text)
        print("-" * 80)

    # ========================================================
    # 8. LABEL MASK
    #
    # labels == -100:
    #     ignored by LM loss
    #
    # labels != -100:
    #     supervised by LM loss
    # ========================================================

    label_row = labels[0]

    supervised_token_indices = torch.where(
        label_row != -100
    )[0]

    ignored_token_indices = torch.where(
        label_row == -100
    )[0]

    print()
    print("=" * 80)
    print("LABEL MASK")
    print("=" * 80)

    print(
        f"Supervised tokens (labels != -100): "
        f"{len(supervised_token_indices)}"
    )

    print(
        f"Ignored tokens    (labels == -100): "
        f"{len(ignored_token_indices)}"
    )

    # ========================================================
    # 9. PRINT SUPERVISED TEXT
    # ========================================================

    if len(supervised_token_indices) > 0:

        supervised_start = (
            supervised_token_indices[0].item()
        )

        supervised_end = (
            supervised_token_indices[-1].item()
            + 1
        )

        supervised_token_ids = input_ids[
            0,
            supervised_start:supervised_end,
        ]

        supervised_text = tokenizer.decode(
            supervised_token_ids,
            skip_special_tokens=False,
        )

        print()
        print("TEXT WITH labels != -100:")
        print("-" * 80)
        print(supervised_text)
        print("-" * 80)

        print(
            f"Start token : {supervised_start}"
        )

        print(
            f"End token   : {supervised_end}"
        )

    # ========================================================
    # 10. PRINT IGNORED TEXT
    #
    # In this setup, this should be the entire
    # system + user question + observation/document section.
    # ========================================================

    if len(ignored_token_indices) > 0:

        ignored_start = (
            ignored_token_indices[0].item()
        )

        ignored_end = (
            ignored_token_indices[-1].item()
            + 1
        )

        ignored_token_ids = input_ids[
            0,
            ignored_start:ignored_end,
        ]

        ignored_text = tokenizer.decode(
            ignored_token_ids,
            skip_special_tokens=False,
        )

        print()
        print("TEXT WITH labels == -100:")
        print("-" * 80)
        print(ignored_text)
        print("-" * 80)

        print(
            f"Start token : {ignored_start}"
        )

        print(
            f"End token   : {ignored_end}"
        )

    # ========================================================
    # 11. PRINT EOS INFORMATION
    # ========================================================

    eos_token_id = tokenizer.eos_token_id

    eos_positions = torch.where(
        label_row == eos_token_id
    )[0]

    print()
    print("=" * 80)
    print("EOS TOKEN")
    print("=" * 80)

    print(
        f"EOS token    : {tokenizer.eos_token}"
    )

    print(
        f"EOS token ID : {eos_token_id}"
    )

    print(
        f"EOS positions in labels: "
        f"{eos_positions.tolist()}"
    )

    # ========================================================
    # 12. VERIFY EOS IS SUPERVISED
    # ========================================================

    if len(eos_positions) > 0:

        eos_position = (
            eos_positions[-1].item()
        )

        print(
            f"EOS label value : "
            f"{label_row[eos_position].item()}"
        )

        print(
            f"EOS is masked (-100): "
            f"{label_row[eos_position].item() == -100}"
        )

    # ========================================================
    # 13. PRINT FINAL TOKEN-BY-TOKEN SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("TOKEN MASK SUMMARY")
    print("=" * 80)

    print(
        "Document mask:"
    )

    print(
        "  -inf = document"
    )

    print(
        "  0    = non-document"
    )

    print()

    print(
        "Label mask:"
    )

    print(
        "  -100 = ignored by LM loss"
    )

    print(
        "  ID   = supervised token"
    )

# if __name__ == "__main__":
#     # test_create_model_input()
#     test_create_model_input_for_lm_loss()




if __name__ == "__main__":
    from datasets import load_dataset

    # ========================================================
    # 1. LOAD TOKENIZER AND DATASET
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO)

    DATASET_NAME = (
        "adaptive-retrieval-timing/"
        "teacher-generated-hypothesis-validation"
    )

    hop = 3
    config_name = f"{hop}-hop"

    dataset = load_dataset(DATASET_NAME, config_name)
    raw_sample = dataset["validation"][0]

    # ========================================================
    # 2. EXTRACT QUESTION, ANSWER, AND PASSAGES
    # ========================================================

    question = raw_sample["question"]
    answer = raw_sample["answer"]

    golden_passages = [
        p for p in raw_sample["paragraphs"]
        if p["is_supporting"]
    ]

    distract_passages = [
        p for p in raw_sample["paragraphs"]
        if not p["is_supporting"]
    ]

    # Print only a short excerpt of each passage.
    CONTENT_PREVIEW_LENGTH = 300

    def print_passages(title, passages):
        print(f"\n{'=' * 80}")
        print(f"{title} ({len(passages)})")
        print("=" * 80)

        for i, passage in enumerate(passages, start=1):
            content = " ".join(
                passage["paragraph_text"].split()
            )
            if len(content) > CONTENT_PREVIEW_LENGTH:
                content = content[:CONTENT_PREVIEW_LENGTH].rstrip() + "..."

            print(f"\n[{i}] {passage['title']}")
            print(content)

    # ========================================================
    # 3. PRINT SLIDE-FRIENDLY SAMPLE
    # ========================================================

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print_passages("GOLDEN PASSAGES", golden_passages)
    print_passages("DISTRACTOR PASSAGES", distract_passages)

    # ========================================================
    # 4. SELECT 3 DOCUMENTS FOR LM INPUT
    #    1 positive + 2 negative
    # ========================================================

    if not golden_passages:
        raise ValueError("This sample has no golden passage.")

    if len(distract_passages) < 2:
        raise ValueError(
            "This sample has fewer than 2 distractor passages."
        )

    selected_golden = golden_passages[0]
    selected_distractors = distract_passages[:2]

    selected_passages = [
        selected_golden,
        *selected_distractors,
    ]

    sample = SampleObjectForLMLoss(
        question=question,
        documents=[
            Document(
                title=p["title"],
                content=p["paragraph_text"],
            )
            for p in selected_passages
        ],
        answer=answer,
    )

    # ========================================================
    # 5. PRINT THE EXACT 3 DOCUMENTS FED TO THE MODEL
    # ========================================================

    print("\n" + "=" * 80)
    print("SELECTED DOCUMENTS FOR LM INPUT (1 POSITIVE + 2 NEGATIVE)")
    print("=" * 80)

    for i, passage in enumerate(selected_passages, start=1):
        label = "POSITIVE" if i == 1 else "NEGATIVE"
        content = " ".join(passage["paragraph_text"].split())

        if len(content) > CONTENT_PREVIEW_LENGTH:
            content = content[:CONTENT_PREVIEW_LENGTH].rstrip() + "..."

        print(f"\n[{i}] [{label}] {passage['title']}")
        print(content)

    # ========================================================
    # 6. BUILD MODEL INPUT
    # ========================================================

    result = create_model_input_for_lm_loss(
        sample=sample,
        tokenizer=tokenizer,
    )

    print("\n" + "=" * 80)
    print("FULL MODEL INPUT")
    print("=" * 80)
    print(result["full_text"])

    print("\n" + "=" * 80)
    print("TENSOR SHAPES")
    print("=" * 80)

    for key in (
        "input_ids",
        "attention_mask",
        "document_attention_mask",
        "retrieval_decision",
        "labels",
    ):
        print(f"{key:26}: {tuple(result[key].shape)}")
