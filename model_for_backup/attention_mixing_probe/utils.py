import torch
def ensure_document_attention_mask(
    input_ids: torch.Tensor,
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


def remove_document_from_rotary_embeddings(
    cos: torch.Tensor,
    sin: torch.Tensor,
    document_attention_mask: torch.Tensor,
):
    """
    Remove the positional effect of document tokens from RoPE.

    Parameters
    ----------
    cos:
        Shape: [B, L, D]

    sin:
        Shape: [B, L, D]

    document_attention_mask:
        Shape: [B, 1, 1, L]

        Document positions are marked with -inf.

    Returns
    -------
    new_cos:
        Shape: [B, L, D]

    new_sin:
        Shape: [B, L, D]
    """

    batch_size, seq_len, head_dim = cos.shape

    # ========================================================
    # 1. Convert document mask
    # ========================================================

    document_mask = (
        document_attention_mask[:, 0, 0, :]
        == float("-inf")
    )

    # Shape:
    # [B, L]

    # ========================================================
    # 2. Create output
    # ========================================================

    new_cos = cos.clone()
    new_sin = sin.clone()

    # ========================================================
    # 3. Process each sample independently
    # ========================================================

    for b in range(batch_size):

        doc_positions = torch.nonzero(
            document_mask[b],
            as_tuple=False,
        ).squeeze(-1)

        # No document
        if doc_positions.numel() == 0:
            continue

        # ----------------------------------------------------
        # Assume document is one contiguous span
        # ----------------------------------------------------

        document_start = doc_positions[0]
        document_end = doc_positions[-1] + 1

        document_length = (
            document_end - document_start
        )

        # ----------------------------------------------------
        # Tokens before document:
        # Keep original cos/sin
        # ----------------------------------------------------

        # Already preserved by clone()

        # ----------------------------------------------------
        # Tokens after document:
        # Shift their positional index backwards
        # by document_length.
        # ----------------------------------------------------

        after_document = torch.arange(
            document_end,
            seq_len,
            device=cos.device,
        )

        shifted_positions = (
            after_document - document_length
        )

        new_cos[
            b,
            after_document,
        ] = cos[
            b,
            shifted_positions,
        ]

        new_sin[
            b,
            after_document,
        ] = sin[
            b,
            shifted_positions,
        ]

        # ----------------------------------------------------
        # Document positions:
        # Keep original cos/sin
        # ----------------------------------------------------

    return new_cos, new_sin


import torch


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("TEST — REMOVE DOCUMENT FROM ROTARY EMBEDDINGS")
    print("=" * 80)

    # --------------------------------------------------------
    # Fake RoPE
    # --------------------------------------------------------
    #
    # Shape:
    #   B = 1
    #   L = 10
    #   D = 2
    #
    # Mỗi position có giá trị riêng để nhìn rõ việc shift.
    #

    cos = torch.arange(
        10 * 2,
        dtype=torch.float32,
    ).reshape(1, 10, 2)

    sin = cos + 100

    # --------------------------------------------------------
    # Document:
    #
    # positions 3,4,5,6
    #
    # document_length = 4
    #
    # Therefore:
    #
    # position 7 -> position 3
    # position 8 -> position 4
    # position 9 -> position 5
    # --------------------------------------------------------

    document_attention_mask = torch.zeros(
        1,
        1,
        1,
        10,
        dtype=torch.float32,
    )

    document_attention_mask[
        0,
        0,
        0,
        3:7,
    ] = float("-inf")

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    new_cos, new_sin = (
        remove_document_from_rotary_embeddings(
            cos,
            sin,
            document_attention_mask,
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("Original COS:")
    print(cos)

    print()
    print("New COS:")
    print(new_cos)

    print()
    print("Document mask:")
    print(document_attention_mask[0, 0, 0])

    # ========================================================
    # EXPECTED
    # ========================================================

    expected_cos = cos.clone()

    # Position 7 <- position 3
    expected_cos[0, 7] = cos[0, 3]

    # Position 8 <- position 4
    expected_cos[0, 8] = cos[0, 4]

    # Position 9 <- position 5
    expected_cos[0, 9] = cos[0, 5]

    # ========================================================
    # ASSERT
    # ========================================================

    assert torch.equal(
        new_cos,
        expected_cos,
    ), (
        "COS shift is incorrect!"
    )

    # --------------------------------------------------------
    # SIN should follow exactly the same mapping
    # --------------------------------------------------------

    expected_sin = sin.clone()

    expected_sin[0, 7] = sin[0, 3]
    expected_sin[0, 8] = sin[0, 4]
    expected_sin[0, 9] = sin[0, 5]

    assert torch.equal(
        new_sin,
        expected_sin,
    ), (
        "SIN shift is incorrect!"
    )

    # ========================================================
    # EXPLICIT POSITION CHECK
    # ========================================================

    print()
    print("-" * 80)
    print("POSITION CHECK")
    print("-" * 80)

    for position in range(10):

        if position < 3:

            expected_position = position
            region = "BEFORE DOC"

        elif position < 7:

            expected_position = position
            region = "DOCUMENT"

        else:

            expected_position = position - 4
            region = "AFTER DOC"

        actual = new_cos[
            0,
            position,
        ]

        expected = cos[
            0,
            expected_position,
        ]

        print(
            f"Position {position:2d} | "
            f"{region:11s} | "
            f"expected source = {expected_position:2d} | "
            f"actual = {actual.tolist()}"
        )

        assert torch.equal(
            actual,
            expected,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("✓ ROPE SHIFT IS CORRECT")
    print("=" * 80)