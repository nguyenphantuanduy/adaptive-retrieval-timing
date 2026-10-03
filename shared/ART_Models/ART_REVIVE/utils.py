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

import torch


def create_document_attention_mask(
    current_sequence_len: int,
    document_len: int,
) -> torch.Tensor:
    """
    Create an additive attention mask for the current sequence
    followed by the retrieved document.

    Layout:

        current sequence        document
        [ 0, 0, 0, ..., 0,   -inf, -inf, ..., -inf ]

    Args:
        current_sequence_len:
            Number of tokens in the current sequence.

        document_len:
            Number of tokens in the retrieved document.

    Returns:
        Tensor of shape:
            (1, current_sequence_len + document_len)

        with:
            0.0   for current-sequence tokens
            -inf  for document tokens
    """

    if current_sequence_len < 0:
        raise ValueError(
            "current_sequence_len must be non-negative"
        )

    if document_len < 0:
        raise ValueError(
            "document_len must be non-negative"
        )

    mask = torch.zeros(
        1,
        current_sequence_len + document_len,
        dtype=torch.float32,
    )

    if document_len > 0:
        mask[:, current_sequence_len:] = float("-inf")

    return mask

import torch


def update_attention_mask(
    attention_mask: torch.Tensor,
    num_tokens: int = 1,
) -> torch.Tensor:
    """
    Append valid tokens to the attention mask.

    Args:
        attention_mask:
            Tensor of shape (B, L).

        num_tokens:
            Number of valid tokens to append.
            Defaults to 1.

    Returns:
        Tensor of shape (B, L + num_tokens).
    """

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape (B, L)"
        )

    if num_tokens <= 0:
        raise ValueError(
            "num_tokens must be > 0"
        )

    new_token_mask = attention_mask.new_ones(
        (attention_mask.shape[0], num_tokens)
    )

    return torch.cat(
        [attention_mask, new_token_mask],
        dim=-1,
    )