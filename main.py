import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "lm_loss_2_hop.csv"

df = pd.read_csv(CSV_FILE)

# ============================================================
# PARSE COLUMNS
# ============================================================

doc_groups = {
    "2:1": [],
    "1:2": [],
    "0:3": [],
}

for col in df.columns:
    for doc in doc_groups:
        if col.startswith(doc + "/"):
            d = float(col.split("/")[-1])
            doc_groups[doc].append((d, col))

# ============================================================
# MEAN CURVES
# ============================================================

mean_curves = {}

print("=" * 80)
print("MEAN LM LOSS — 2-HOP — 200 SAMPLES")
print("=" * 80)

for doc, columns in doc_groups.items():

    columns = sorted(columns)

    d_values = []
    mean_losses = []

    print(f"\nDocument position: {doc}")
    print("-" * 80)

    for d, col in columns:
        mean_loss = df[col].mean()

        d_values.append(d)
        mean_losses.append(mean_loss)

        print(
            f"d={d:.4f} | "
            f"mean_loss={mean_loss:.6f} | "
            f"valid={df[col].notna().sum()}"
        )

    mean_curves[doc] = (d_values, mean_losses)

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

for doc, (d_values, mean_losses) in mean_curves.items():

    # Ignore NaN just in case
    valid = [
        (d, loss)
        for d, loss in zip(d_values, mean_losses)
        if pd.notna(loss)
    ]

    best_d, best_loss = min(valid, key=lambda x: x[1])

    loss_d0 = mean_losses[0]
    loss_d1 = mean_losses[-1]

    print(f"\n{doc}")
    print(f"  Mean loss @ d=0 : {loss_d0:.6f}")
    print(f"  Mean loss @ d=1 : {loss_d1:.6f}")
    print(f"  Δ loss (0 -> 1) : {loss_d1 - loss_d0:.6f}")
    print(f"  Best d          : {best_d:.4f}")
    print(f"  Best mean loss   : {best_loss:.6f}")

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

for doc, (d_values, mean_losses) in mean_curves.items():
    plt.plot(
        d_values,
        mean_losses,
        marker="o",
        markersize=3,
        label=doc,
    )

plt.xlabel("Document Attention Modulation d")
plt.ylabel("Mean LM Loss")
plt.title("Mean LM Loss vs Document Attention Modulation — 2-Hop")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("lm_loss_2_hop_mean_curve.png", dpi=200)
plt.show()