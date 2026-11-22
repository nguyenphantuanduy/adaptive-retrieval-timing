from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = (
    "adaptive-retrieval-timing/"
    "teacher-generated-hypothesis-validation"
)

MODEL_NAME = "Qwen/Qwen3-4B"


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_teacher_generated_hypothesis_validation():

    datasets = {}

    for hop in (2, 3, 4):

        config_name = f"{hop}-hop"

        dataset = load_dataset(
            DATASET_NAME,
            config_name,
        )

        datasets[config_name] = (
            dataset["validation"]
        )

    return datasets


# ============================================================
# STATISTICS
# ============================================================

def analyze_document_token_distribution(datasets):

    for hop, samples in datasets.items():

        positive_token_counts = []
        negative_token_counts = []

        for sample in samples:

            paragraphs = sample["paragraphs"]

            for paragraph in paragraphs:

                text = paragraph["paragraph_text"]

                token_ids = tokenizer(
                    text,
                    add_special_tokens=False,
                )["input_ids"]

                token_count = len(token_ids)

                if paragraph["is_supporting"]:

                    positive_token_counts.append(
                        token_count
                    )

                else:

                    negative_token_counts.append(
                        token_count
                    )

        print()
        print("=" * 80)
        print(
            f"{hop.upper()} — DOCUMENT TOKEN DISTRIBUTION"
        )
        print("=" * 80)

        print(
            f"Samples                  : "
            f"{len(samples):,}"
        )

        # ----------------------------------------------------
        # POSITIVE
        # ----------------------------------------------------

        print()
        print("Positive documents")

        print(
            f"  Number of documents     : "
            f"{len(positive_token_counts):,}"
        )

        print(
            f"  Min tokens              : "
            f"{min(positive_token_counts):,}"
        )

        print(
            f"  Max tokens              : "
            f"{max(positive_token_counts):,}"
        )

        print(
            f"  Mean tokens             : "
            f"{sum(positive_token_counts) / len(positive_token_counts):.2f}"
        )

        # ----------------------------------------------------
        # NEGATIVE
        # ----------------------------------------------------

        print()
        print("Negative documents")

        print(
            f"  Number of documents     : "
            f"{len(negative_token_counts):,}"
        )

        print(
            f"  Min tokens              : "
            f"{min(negative_token_counts):,}"
        )

        print(
            f"  Max tokens              : "
            f"{max(negative_token_counts):,}"
        )

        print(
            f"  Mean tokens             : "
            f"{sum(negative_token_counts) / len(negative_token_counts):.2f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    datasets = (
        load_teacher_generated_hypothesis_validation()
    )

    analyze_document_token_distribution(
        datasets
    )