import torch
import torch.nn.functional as F


def main():

    # ========================================================
    # TEST float("-inf") ATTENTION MASK
    # ========================================================

    print("=" * 80)
    print('TEST — float("-inf") MASK')
    print("=" * 80)

    # --------------------------------------------------------
    # Attention scores
    # --------------------------------------------------------

    scores = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Mask middle position
    # --------------------------------------------------------

    mask = torch.zeros(
        3,
        dtype=torch.float32,
    )

    mask[1] = float("-inf")

    print()
    print("Scores:")
    print(scores)

    print()
    print("Mask:")
    print(mask)

    print()
    print("Scores + Mask:")
    print(scores + mask)

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    attention_weights = F.softmax(
        scores + mask,
        dim=-1,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("Attention weights:")
    print(attention_weights)

    print()
    print("Masked position:")
    print(
        f"  Value     : {attention_weights[1].item()}"
    )

    print(
        f"  Is zero   : "
        f"{attention_weights[1].item() == 0.0}"
    )

    print(
        f"  Is finite : "
        f"{torch.isfinite(attention_weights[1]).item()}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert torch.isinf(mask[1])
    assert mask[1] < 0

    assert attention_weights[1].item() == 0.0

    assert torch.isclose(
        attention_weights.sum(),
        torch.tensor(1.0),
    )

    print()
    print("=" * 80)
    print("✓ float('-inf') MASK WORKS")
    print("=" * 80)


if __name__ == "__main__":
    main()