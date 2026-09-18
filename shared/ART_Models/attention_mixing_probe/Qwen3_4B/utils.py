import torch
def ensure_document_attention_mask(
    document_attention_mask: torch.Tensor,
) -> torch.Tensor:

    if document_attention_mask.ndim == 2:

        return document_attention_mask.unsqueeze(
            1
        ).unsqueeze(
            1
        )

    return document_attention_mask

import torch


def create_position_ids_without_doc(
    position_ids_with_doc: torch.LongTensor,
    document_attention_mask: torch.Tensor,
) -> torch.LongTensor:
    """
    Create position IDs for the no-document attention path.

    The document is treated as if it never existed in the
    positional timeline.

    IMPORTANT
    ---------
    The document mask and position_ids_with_doc do NOT
    necessarily have the same sequence length.

    This is important during generation with KV cache.

    Example:

        document mask represents:
            positions 56 ... 259

        but during generation the model may receive only:

            position_ids_with_doc = [[260]]

        Therefore we must NOT use document positions as
        indices into position_ids_with_doc.

        Instead, document positions are interpreted as
        absolute position values.
    """

    # ========================================================
    # 1. CONVERT DOCUMENT MASK
    # ========================================================

    document_mask = (
        document_attention_mask[:, 0, 0, :]
        == float("-inf")
    )

    batch_size = (
        position_ids_with_doc.shape[0]
    )

    # ========================================================
    # 2. CLONE
    # ========================================================

    position_ids_without_doc = (
        position_ids_with_doc.clone()
    )

    # ========================================================
    # 3. PROCESS EACH SAMPLE
    # ========================================================

    for b in range(batch_size):

        doc_positions = torch.nonzero(
            document_mask[b],
            as_tuple=False,
        ).squeeze(-1)

        # ----------------------------------------------------
        # No document.
        # ----------------------------------------------------

        if doc_positions.numel() == 0:
            continue

        # ----------------------------------------------------
        # Document span in ABSOLUTE POSITION SPACE.
        #
        # Example:
        #
        #     56 ... 259
        #
        # document_start_position = 56
        # document_length = 204
        # document_last_position = 259
        # ----------------------------------------------------

        document_start_position = (
            doc_positions[0]
        )

        document_length = (
            doc_positions.numel()
        )

        document_last_position = (
            document_start_position
            + document_length
            - 1
        )

        # ----------------------------------------------------
        # Find tokens AFTER the document.
        #
        # IMPORTANT:
        #
        # Compare POSITION VALUES.
        #
        # Do NOT compare tensor indices.
        # ----------------------------------------------------

        after_document = (
            position_ids_with_doc[b]
            > document_last_position
        )

        # ----------------------------------------------------
        # Shift them backwards by document length.
        # ----------------------------------------------------

        position_ids_without_doc[
            b,
            after_document,
        ] = (
            position_ids_with_doc[
                b,
                after_document,
            ]
            - document_length
        )

    return position_ids_without_doc

from torch import nn

def align_document_attention_mask(
    document_attention_mask: torch.Tensor,
    causal_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Align document attention mask with the causal mask
    along the sequence dimension.

    Parameters
    ----------
    document_attention_mask:
        Shape:
            [B, 1, 1, L_doc]

    causal_mask:
        Causal mask whose last dimension defines the
        required sequence length.

    Returns
    -------
    document_attention_mask:
        Shape:
            [B, 1, 1, L_causal]
    """

    if causal_mask is None:
        return document_attention_mask

    causal_length = causal_mask.shape[-1]
    document_length = document_attention_mask.shape[-1]

    if document_length < causal_length:

        pad_length = causal_length - document_length

        document_attention_mask = nn.functional.pad(
            document_attention_mask,
            (0, pad_length),
            value=0.0,
        )

    elif document_length > causal_length:

        document_attention_mask = (
            document_attention_mask[..., :causal_length]
        )

    return document_attention_mask

# ============================================================
# TEST UTIL
# ============================================================

def make_document_attention_mask(
    document_spans,
    seq_len,
):
    """
    document_spans:
        List of (start, end) for each sample.

        end is exclusive.

    Returns:
        [B, 1, 1, L]
    """

    batch_size = len(
        document_spans
    )

    mask = torch.zeros(
        batch_size,
        1,
        1,
        seq_len,
        dtype=torch.float32,
    )

    for b, (start, end) in enumerate(
        document_spans
    ):

        if (
            start is not None
            and end is not None
        ):

            mask[
                b,
                0,
                0,
                start:end,
            ] = float("-inf")

    return mask


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print(
        "TEST — create_position_ids_without_doc"
    )
    print("=" * 80)


    # ========================================================
    # TEST 1 — NO DOCUMENT
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 1 — NO DOCUMENT")
    print("-" * 80)

    position_ids = torch.arange(
        10,
        dtype=torch.long,
    ).unsqueeze(0)

    document_mask = make_document_attention_mask(
        [(None, None)],
        seq_len=10,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = position_ids.clone()

    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # TEST 2 — DOCUMENT IN THE MIDDLE
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 2 — DOCUMENT IN THE MIDDLE")
    print("-" * 80)

    position_ids = torch.arange(
        10,
        dtype=torch.long,
    ).unsqueeze(0)

    # Document:
    #
    #     3 ... 6
    #
    # Length = 4

    document_mask = make_document_attention_mask(
        [(3, 7)],
        seq_len=10,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = torch.tensor(
        [[
            0, 1, 2,
            3, 4, 5, 6,
            3, 4, 5,
        ]],
        dtype=torch.long,
    )

    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # TEST 3 — DOCUMENT AT END
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 3 — DOCUMENT AT END")
    print("-" * 80)

    position_ids = torch.arange(
        10,
        dtype=torch.long,
    ).unsqueeze(0)

    # Document:
    #
    #     6 ... 9

    document_mask = make_document_attention_mask(
        [(6, 10)],
        seq_len=10,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = position_ids.clone()

    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # TEST 4 — DOCUMENT AT BEGINNING
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 4 — DOCUMENT AT BEGINNING")
    print("-" * 80)

    position_ids = torch.arange(
        10,
        dtype=torch.long,
    ).unsqueeze(0)

    # Document:
    #
    #     0 ... 3

    document_mask = make_document_attention_mask(
        [(0, 4)],
        seq_len=10,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = torch.tensor(
        [[
            0, 1, 2, 3,
            0, 1, 2, 3, 4, 5,
        ]],
        dtype=torch.long,
    )

    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # TEST 5 — REALISTIC FULL SEQUENCE
    # ========================================================

    print()
    print("-" * 80)
    print("TEST 5 — REALISTIC FULL SEQUENCE")
    print("-" * 80)

    seq_len = 263

    position_ids = torch.arange(
        seq_len,
        dtype=torch.long,
    ).unsqueeze(0)

    # Document:
    #
    #     56 ... 259
    #
    # Length:
    #
    #     204
    #
    # Answer:
    #
    #     260 261 262

    document_mask = make_document_attention_mask(
        [(56, 260)],
        seq_len=seq_len,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    print()
    print("Document:")
    print("  start  = 56")
    print("  end    = 260")
    print("  length = 204")

    print()
    print("Relevant positions:")

    for position in [
        55,
        56,
        57,
        258,
        259,
        260,
        261,
        262,
    ]:

        print(
            f"  target {position:3d}"
            f"  with_doc = "
            f"{position_ids[0, position].item():3d}"
            f"  without_doc = "
            f"{result[0, position].item():3d}"
        )

    # Before document unchanged.
    assert (
        result[0, 55].item()
        == 55
    )

    # Document positions retain original positions.
    assert (
        result[0, 56].item()
        == 56
    )

    assert (
        result[0, 57].item()
        == 57
    )

    assert (
        result[0, 259].item()
        == 259
    )

    # Answer positions shift backward.
    assert (
        result[0, 260].item()
        == 56
    )

    assert (
        result[0, 261].item()
        == 57
    )

    assert (
        result[0, 262].item()
        == 58
    )

    print()
    print("✓ PASS")


    # ========================================================
    # TEST 6 — BATCH WITH DIFFERENT DOCUMENT SPANS
    # ========================================================

    print()
    print("-" * 80)
    print(
        "TEST 6 — BATCH WITH DIFFERENT DOCUMENT SPANS"
    )
    print("-" * 80)

    seq_len = 12

    position_ids = (
        torch.arange(
            seq_len,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .repeat(3, 1)
    )

    document_spans = [
        (2, 5),   # length 3
        (4, 8),   # length 4
        (7, 9),   # length 2
    ]

    document_mask = make_document_attention_mask(
        document_spans,
        seq_len=seq_len,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = torch.tensor(
        [
            [
                0, 1, 2, 3, 4,
                2, 3, 4, 5, 6, 7, 8,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7,
                4, 5, 6, 7,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                7, 8, 9,
            ],
        ],
        dtype=torch.long,
    )

    print()
    print("with_doc:")
    print(position_ids)

    print()
    print("without_doc:")
    print(result)

    print()
    print("expected:")
    print(expected)

    assert torch.equal(
        result,
        expected,
    )

    print()
    print("✓ PASS")


    # ========================================================
    # TEST 7 — PAST CACHE, MULTIPLE NEW TOKENS
    # ========================================================

    print()
    print("-" * 80)
    print(
        "TEST 7 — PAST CACHE, MULTIPLE NEW TOKENS"
    )
    print("-" * 80)

    # Imagine:
    #
    # Document ends at position 259.
    #
    # The model is now processing:
    #
    #     260 261 262
    #
    # The current tensor only contains these 3 tokens.

    position_ids = torch.tensor(
        [[260, 261, 262]],
        dtype=torch.long,
    )

    # The document mask here represents the document's
    # ORIGINAL positions.
    #
    # We provide a mask large enough to include the document.

    document_mask = make_document_attention_mask(
        [(56, 260)],
        seq_len=263,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = torch.tensor(
        [[56, 57, 58]],
        dtype=torch.long,
    )

    print()
    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # TEST 8 — PAST CACHE, SINGLE TOKEN
    #
    # THIS IS THE IMPORTANT CASE
    # ========================================================

    print()
    print("-" * 80)
    print(
        "TEST 8 — PAST CACHE, SINGLE TOKEN"
    )
    print("-" * 80)

    # This simulates HF generation where only the newest
    # token is processed.
    #
    # The tensor has:
    #
    #     shape = [1, 1]
    #
    # but its position is:
    #
    #     260
    #
    # The tensor index is 0.
    #
    # Therefore using:
    #
    #     torch.arange(document_end, seq_len)
    #
    # would FAIL.
    #
    # We instead compare the POSITION VALUE 260 with
    # the document's last position 259.

    position_ids = torch.tensor(
        [[260]],
        dtype=torch.long,
    )

    document_mask = make_document_attention_mask(
        [(56, 260)],
        seq_len=263,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = torch.tensor(
        [[56]],
        dtype=torch.long,
    )

    print()
    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # TEST 9 — PAST CACHE, SINGLE TOKEN INSIDE DOCUMENT
    # ========================================================

    print()
    print("-" * 80)
    print(
        "TEST 9 — PAST CACHE, SINGLE TOKEN INSIDE DOCUMENT"
    )
    print("-" * 80)

    # Position 100 is inside the document.
    #
    # It must NOT be shifted.

    position_ids = torch.tensor(
        [[100]],
        dtype=torch.long,
    )

    document_mask = make_document_attention_mask(
        [(56, 260)],
        seq_len=263,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = torch.tensor(
        [[100]],
        dtype=torch.long,
    )

    print()
    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # TEST 10 — PAST CACHE, SINGLE TOKEN BEFORE DOCUMENT
    # ========================================================

    print()
    print("-" * 80)
    print(
        "TEST 10 — PAST CACHE, SINGLE TOKEN BEFORE DOCUMENT"
    )
    print("-" * 80)

    # Position 20 is before the document.
    #
    # It must NOT be shifted.

    position_ids = torch.tensor(
        [[20]],
        dtype=torch.long,
    )

    document_mask = make_document_attention_mask(
        [(56, 260)],
        seq_len=263,
    )

    result = create_position_ids_without_doc(
        position_ids,
        document_mask,
    )

    expected = torch.tensor(
        [[20]],
        dtype=torch.long,
    )

    print()
    print(
        "with_doc    :",
        position_ids[0].tolist(),
    )

    print(
        "without_doc :",
        result[0].tolist(),
    )

    print(
        "expected    :",
        expected[0].tolist(),
    )

    assert torch.equal(
        result,
        expected,
    )

    print("✓ PASS")


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)