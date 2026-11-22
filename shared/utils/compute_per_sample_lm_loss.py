import torch
import torch.nn.functional as F


def compute_per_sample_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute causal LM loss independently for each sample.

    Parameters
    ----------
    logits:
        Shape: [B, L, V]

    labels:
        Shape: [B, L]

        Tokens with label -100 are ignored.

    Returns
    -------
    torch.Tensor
        Shape: [B]

        Mean LM loss for each sample.
    """

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    batch_size = shift_labels.shape[0]

    token_loss = F.cross_entropy(
        shift_logits.view(
            -1,
            shift_logits.size(-1),
        ),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )

    token_loss = token_loss.view(
        batch_size,
        -1,
    )

    valid_tokens = (
        shift_labels != -100
    )

    per_sample_loss = (
        (token_loss * valid_tokens).sum(dim=1)
        / valid_tokens.sum(dim=1).clamp_min(1)
    )

    return per_sample_loss