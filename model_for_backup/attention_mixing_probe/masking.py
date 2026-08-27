import torch


def prepare_document_attention_mask(
    document_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a document mask from (B, L) to an additive
    attention mask of shape (B, 1, 1, L).

    Input:
        document_mask:
            0 -> normal token
            1 -> document token

    Output:
        0    -> normal token
        -inf -> document token
    """
    if document_mask.dim() != 2:
        raise ValueError(
            f"Expected document_mask with shape (B, L), "
            f"got {tuple(document_mask.shape)}"
        )

    mask = document_mask[:, None, None, :].float()

    return mask.masked_fill(
        mask.bool(),
        float("-inf"),
    )