# ============================================================
# MUSIQUE TEACHER DATASET UPLOAD
#
# Convert generated MuSiQue teacher-generation JSONL files
# into Hugging Face Dataset configurations and upload them.
#
# Target repository:
#
#   adaptive-retrieval-timing/
#   teacher-generated-hypothesis-validation
#
#
# Repository structure:
#
#   2-hop
#       └── validation
#
#   3-hop
#       └── validation
#
#   4-hop
#       └── validation
#
#
# Later:
#
#   from datasets import load_dataset
#
#   dataset = load_dataset(
#       "adaptive-retrieval-timing/"
#       "teacher-generated-hypothesis-validation",
#       "2-hop",
#   )
#
#   validation = dataset["validation"]
#
# ============================================================


from pathlib import Path

from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

HF_REPO_ID = (
    "adaptive-retrieval-timing/"
    "teacher-generated-hypothesis-validation"
)


# ------------------------------------------------------------
# Each hop becomes a separate Hugging Face configuration.
# ------------------------------------------------------------

CONFIGS = {
    2: "2-hop",
    3: "3-hop",
    4: "4-hop",
}


# ------------------------------------------------------------
# The generated data comes from the MuSiQue dev/validation
# split, so the uploaded Hugging Face split is also named
# "validation".
# ------------------------------------------------------------

HF_SPLIT = "validation"


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE_DIR = (
    Path(__file__).resolve().parent
)


DATA_DIR = (
    CURRENT_FILE_DIR
    / "data"
    / "teacher_generation_for_hypothesis_validation"
)


# ============================================================
# LOAD ONE HOP
# ============================================================

def load_teacher_dataset(
    hop: int,
):
    """
    Load one generated MuSiQue teacher JSONL file
    as a Hugging Face Dataset.

    Expected file:

        data/
        teacher_generation_for_hypothesis_validation/
            musique_2_hop.jsonl
            musique_3_hop.jsonl
            musique_4_hop.jsonl
    """

    # --------------------------------------------------------
    # Validate hop
    # --------------------------------------------------------

    if hop not in CONFIGS:

        raise ValueError(
            f"Unsupported hop: {hop}. "
            f"Expected one of: "
            f"{list(CONFIGS.keys())}"
        )

    # --------------------------------------------------------
    # Input path
    # --------------------------------------------------------

    input_path = (
        DATA_DIR
        / f"musique_{hop}_hop.jsonl"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Teacher dataset not found:\n"
            f"{input_path}"
        )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"LOADING {hop}-HOP TEACHER DATASET"
    )
    print("=" * 80)

    print()
    print(
        f"Input file:"
    )

    print(
        f"  {input_path}"
    )

    # --------------------------------------------------------
    # Load JSONL
    # --------------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            HF_SPLIT: str(input_path),
        },
    )

    validation_dataset = dataset[
        HF_SPLIT
    ]

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print()
    print(
        "✓ Dataset loaded"
    )

    print(
        f"  Samples: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"  Features:"
    )

    for feature in (
        validation_dataset.column_names
    ):

        print(
            f"    - {feature}"
        )

    return validation_dataset


# ============================================================
# VALIDATE ONE HOP
# ============================================================

def validate_teacher_dataset(
    dataset,
    hop: int,
):
    """
    Validate the generated teacher dataset before upload.

    Required fields:

        id
        teacher_generation_output

    All original MuSiQue fields are preserved.
    """

    print()
    print("=" * 80)
    print(
        f"VALIDATING {hop}-HOP DATASET"
    )
    print("=" * 80)

    # ========================================================
    # 1. CHECK EMPTY DATASET
    # ========================================================

    if len(dataset) == 0:

        raise ValueError(
            f"{hop}-hop dataset is empty."
        )

    # ========================================================
    # 2. REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "id",
        "teacher_generation_output",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in dataset.column_names
    ]

    if missing_fields:

        raise ValueError(
            f"{hop}-hop dataset is missing "
            f"required fields: "
            f"{missing_fields}"
        )

    # ========================================================
    # 3. VALIDATE IDS
    # ========================================================

    ids = dataset["id"]

    if len(ids) != len(set(ids)):

        raise ValueError(
            f"{hop}-hop dataset contains "
            f"duplicate IDs."
        )

    # ========================================================
    # 4. VALIDATE TEACHER OUTPUTS
    # ========================================================

    teacher_outputs = dataset[
        "teacher_generation_output"
    ]

    invalid_teacher_outputs = 0

    for teacher_output in teacher_outputs:

        if not isinstance(
            teacher_output,
            str,
        ):

            invalid_teacher_outputs += 1

            continue

        if not teacher_output.strip():

            invalid_teacher_outputs += 1

    if invalid_teacher_outputs > 0:

        raise ValueError(
            f"{hop}-hop dataset contains "
            f"{invalid_teacher_outputs:,} "
            f"empty or invalid teacher outputs."
        )

    # ========================================================
    # 5. PRINT VALIDATION RESULT
    # ========================================================

    print()
    print(
        "✓ Validation passed"
    )

    print(
        f"  Samples       : "
        f"{len(dataset):,}"
    )

    print(
        f"  Unique IDs    : "
        f"{len(set(ids)):,}"
    )

    print(
        f"  Features      : "
        f"{len(dataset.column_names)}"
    )

    print(
        f"  Teacher output: "
        f"valid"
    )


# ============================================================
# UPLOAD ONE HOP
# ============================================================

def upload_hop_dataset(
    dataset,
    hop: int,
):
    """
    Upload one hop as a separate Hugging Face
    Dataset configuration.

    Example:

        config_name = "2-hop"
        split       = "validation"
    """

    config_name = CONFIGS[
        hop
    ]

    print()
    print("=" * 80)
    print(
        f"UPLOADING {hop}-HOP DATASET"
    )
    print("=" * 80)

    print()
    print(
        f"Repository:"
    )

    print(
        f"  {HF_REPO_ID}"
    )

    print(
        f"Configuration:"
    )

    print(
        f"  {config_name}"
    )

    print(
        f"Split:"
    )

    print(
        f"  {HF_SPLIT}"
    )

    print(
        f"Samples:"
    )

    print(
        f"  {len(dataset):,}"
    )

    # ========================================================
    # PUSH TO HUB
    # ========================================================

    dataset.push_to_hub(
        HF_REPO_ID,
        config_name=config_name,
        split=HF_SPLIT,
        commit_message=(
            f"Add MuSiQue {hop}-hop "
            f"teacher-generated validation data"
        ),
    )

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print(
        f"✓ {hop}-hop dataset uploaded"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MUSIQUE TEACHER DATASET UPLOAD"
    )
    print("=" * 80)

    print()
    print(
        "Purpose:"
    )

    print(
        "Upload teacher-generated MuSiQue "
        "validation data for hypothesis validation."
    )

    print()
    print(
        "Target repository:"
    )

    print(
        f"  {HF_REPO_ID}"
    )

    print()
    print(
        "Source directory:"
    )

    print(
        f"  {DATA_DIR}"
    )

    print()

    # ========================================================
    # 1. CHECK SOURCE DIRECTORY
    # ========================================================

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            "Teacher generation directory "
            "does not exist:\n"
            f"{DATA_DIR}"
        )

    # ========================================================
    # 2. CHECK ALL EXPECTED FILES
    # ========================================================

    print("=" * 80)
    print(
        "CHECKING INPUT FILES"
    )
    print("=" * 80)

    print()

    for hop in (2, 3, 4):

        input_path = (
            DATA_DIR
            / f"musique_{hop}_hop.jsonl"
        )

        if not input_path.exists():

            raise FileNotFoundError(
                f"Missing {hop}-hop file:\n"
                f"{input_path}"
            )

        print(
            f"✓ {input_path.name}"
        )

    # ========================================================
    # 3. LOAD + VALIDATE + UPLOAD
    # ========================================================

    uploaded_counts = {}

    for hop in (2, 3, 4):

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        dataset = (
            load_teacher_dataset(
                hop=hop
            )
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validate_teacher_dataset(
            dataset=dataset,
            hop=hop,
        )

        # ----------------------------------------------------
        # Upload
        # ----------------------------------------------------

        upload_hop_dataset(
            dataset=dataset,
            hop=hop,
        )

        uploaded_counts[
            hop
        ] = len(dataset)

    # ========================================================
    # 4. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "MUSIQUE TEACHER DATASET UPLOAD FINISHED"
    )
    print("=" * 80)

    print()

    print(
        f"Repository:"
    )

    print(
        f"  {HF_REPO_ID}"
    )

    print()

    print(
        "Configurations:"
    )

    for hop in (2, 3, 4):

        print(
            f"  {CONFIGS[hop]}"
            f" / {HF_SPLIT}"
            f" → "
            f"{uploaded_counts[hop]:,} samples"
        )

    print()

    print(
        "Expected structure:"
    )

    print()

    print(
        "  2-hop"
    )

    print(
        "    └── validation"
    )

    print(
        "  3-hop"
    )

    print(
        "    └── validation"
    )

    print(
        "  4-hop"
    )

    print(
        "    └── validation"
    )

    print()
    print("=" * 80)

    print(
        "✓ ALL DATASETS UPLOADED SUCCESSFULLY"
    )

    print("=" * 80)


# ============================================================
# OFFICIAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()