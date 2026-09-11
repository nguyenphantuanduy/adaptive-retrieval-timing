import pandas as pd

CSV_FILE = "lm_loss_2_hop.csv"

df = pd.read_csv(CSV_FILE)

# ============================================================
# DOCUMENT GROUPS TO CHECK
# ============================================================

DOC_GROUPS = {
    "2:1": [],
    "1:2": [],
    "0:3": [],
}

for col in df.columns:
    for doc in DOC_GROUPS:
        if col.startswith(doc + "/"):
            d = float(col.split("/")[-1])
            DOC_GROUPS[doc].append((d, col))

# ============================================================
# NAN CHECK
# ============================================================

print("=" * 80)
print("NAN CHECK — 2-HOP")
print("=" * 80)

print(f"Total samples: {len(df)}")
print()

total_nan_samples = set()

for doc, columns in DOC_GROUPS.items():

    columns = sorted(columns)

    print("=" * 80)
    print(f"DOCUMENT: {doc}")
    print("=" * 80)

    doc_nan_count = 0

    for d, col in columns:

        nan_mask = df[col].isna()
        nan_indices = df.index[nan_mask].tolist()

        if len(nan_indices) > 0:

            print(
                f"\nd={d:.4f} | "
                f"NaN={len(nan_indices)}"
            )

            for idx in nan_indices:
                sample_id = df.loc[idx, "sample_id"]
                sample_index = df.loc[idx, "sample_index"]

                print(
                    f"  row={idx} | "
                    f"sample_index={sample_index} | "
                    f"sample_id={sample_id}"
                )

                total_nan_samples.add(idx)

            doc_nan_count += len(nan_indices)

    if doc_nan_count == 0:
        print("No NaN found.")

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total samples              : {len(df)}")
print(f"Samples with NaN            : {len(total_nan_samples)}")

if len(total_nan_samples) == 0:
    print("\n✓ No NaN found in 2:1, 1:2, or 0:3.")
else:

    print("\nSamples containing NaN:")
    print("-" * 80)

    for idx in sorted(total_nan_samples):

        sample_id = df.loc[idx, "sample_id"]
        sample_index = df.loc[idx, "sample_index"]

        nan_fields = []

        for doc, columns in DOC_GROUPS.items():
            for d, col in sorted(columns):

                if pd.isna(df.loc[idx, col]):
                    nan_fields.append(
                        f"{doc}/{d:.4f}"
                    )

        print(
            f"\nrow={idx}"
            f" | sample_index={sample_index}"
            f" | sample_id={sample_id}"
        )

        print("NaN fields:")
        for field in nan_fields:
            print(f"  - {field}")