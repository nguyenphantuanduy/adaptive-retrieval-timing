
"""
Streamlit reviewer for removed MuSiQue teacher-training samples (v2).

Features
--------
1. Load teacher_training_removed_samples_v2.json.
2. Display one sample per page.
3. Navigate between samples using Previous / Next.
4. Filter by:
       - Config (2-hop / 3-hop / 4-hop)
       - Split
       - Removal reasons
       - Sample ID / question search
5. Display sample metadata and answers.
6. Display ONLY golden/supporting passages.
7. Display the complete sample as copyable text.
8. Show dataset-level and filtered statistics.

Run from repository root:

    .\\.venv\\Scripts\\python.exe -m streamlit run ^
        shared\\MuSiQue\\data\\musique\\review_removed_samples_v2.py

JSON:
    shared/MuSiQue/data/teacher_training_removed_samples_v2.json
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================
# CONFIG
# ============================================================

# JSON_PATH = (
#     Path(__file__).resolve().parent.parent
#     / "data"/"musique"/"teacher_training_removed_samples_v2.json"
# )
JSON_PATH = Path("shared/MuSiQue/data/teacher_training_removed_samples_v2.json")

PAGE_TITLE = "MuSiQue Removed Samples Reviewer"

REASON_LABELS = {
    "missing_teacher_output": "Missing teacher output",
    "empty_teacher_output": "Empty teacher output",
    "missing_final_answer": "Missing final answer",
    "answer_mismatch": "Answer mismatch",
}

REASON_DESCRIPTIONS = {
    "missing_teacher_output": (
        "teacher_generation_output is None."
    ),
    "empty_teacher_output": (
        "teacher_generation_output exists but is empty."
    ),
    "missing_final_answer": (
        "No valid final <answer>...</answer> block was found."
    ),
    "answer_mismatch": (
        "Extracted final answer does not match ground truth."
    ),
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD JSON
# ============================================================

@st.cache_data(show_spinner=False)
def load_removed_samples(
    file_path: str,
    modified_time: float,
) -> dict[str, Any]:
    """
    Load the removed-samples JSON.

    modified_time is included in the cache key so that
    changes to the JSON file trigger a reload.
    """
    del modified_time

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "JSON root must be an object."
        )

    if "metadata" not in data:
        raise ValueError(
            "JSON is missing the 'metadata' field."
        )

    if "removed_samples" not in data:
        raise ValueError(
            "JSON is missing the 'removed_samples' field."
        )

    if not isinstance(data["removed_samples"], list):
        raise ValueError(
            "'removed_samples' must be a list."
        )

    return data


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )

    return str(value)


def get_sample_data(
    item: dict[str, Any],
) -> dict[str, Any]:
    sample = item.get("sample", {})

    if not isinstance(sample, dict):
        return {}

    return sample


def get_question(
    item: dict[str, Any],
) -> str:
    sample = get_sample_data(item)

    return safe_string(
        sample.get("question", "No question available.")
    )


def get_sample_id(
    item: dict[str, Any],
) -> str:
    return safe_string(
        item.get("id", "Unknown ID")
    )


def get_removal_reasons(
    item: dict[str, Any],
) -> list[str]:
    reasons = item.get("removal_reasons", [])

    if not isinstance(reasons, list):
        return []

    return [
        str(reason)
        for reason in reasons
    ]


def is_golden_passage(
    passage: dict[str, Any],
) -> bool:
    """
    Check whether a passage is a golden/supporting passage.

    Supports boolean True and string "true".
    """
    value = passage.get("is_supporting", False)

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() == "true"


def get_golden_passages(
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return only passages with is_supporting=True.
    """
    sample = get_sample_data(item)

    paragraphs = sample.get("paragraphs", [])

    if not isinstance(paragraphs, list):
        return []

    return [
        passage
        for passage in paragraphs
        if isinstance(passage, dict)
        and is_golden_passage(passage)
    ]


# ============================================================
# FORMAT COMPLETE SAMPLE FOR COPYING
# ============================================================

def build_copy_text(
    item: dict[str, Any],
) -> str:
    """
    Build a complete text representation of one sample.

    Includes:
        - Review metadata
        - Removal reasons
        - Expected answer
        - Predicted answer
        - Complete original sample JSON

    The original sample is preserved in full, including
    non-golden passages, so the reviewer has complete context.
    """

    sample = get_sample_data(item)

    lines = [
        "=" * 90,
        "MUSIQUE TEACHER TRAINING SAMPLE REVIEW",
        "=" * 90,
        "",
        "[REVIEW METADATA]",
        f"Config           : {safe_string(item.get('config'))}",
        f"Split            : {safe_string(item.get('split'))}",
        f"Index            : {safe_string(item.get('index'))}",
        f"Sample ID        : {get_sample_id(item)}",
        "",
        "[REMOVAL REASONS]",
    ]

    reasons = get_removal_reasons(item)

    if reasons:
        for reason in reasons:
            lines.append(
                f"- {reason}: "
                f"{REASON_DESCRIPTIONS.get(reason, 'No description.')}"
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "[ANSWER COMPARISON]",
            f"Expected answer  : {safe_string(item.get('expected_answer'))}",
            f"Predicted answer : {safe_string(item.get('predicted_answer'))}",
            "",
            "[QUESTION]",
            get_question(item),
            "",
            "[GOLDEN PASSAGES]",
        ]
    )

    golden_passages = get_golden_passages(item)

    if golden_passages:

        for index, passage in enumerate(golden_passages, start=1):

            lines.extend(
                [
                    "",
                    f"--- Golden Passage {index} ---",
                    f"Title : {safe_string(passage.get('title'))}",
                    f"Index : {safe_string(passage.get('idx'))}",
                    "",
                    safe_string(
                        passage.get("paragraph_text", "")
                    ),
                ]
            )

    else:

        lines.append("No golden passages found.")

    lines.extend(
        [
            "",
            "=" * 90,
            "[COMPLETE ORIGINAL SAMPLE JSON]",
            "=" * 90,
            json.dumps(
                sample,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "",
            "=" * 90,
            "END OF SAMPLE",
            "=" * 90,
        ]
    )

    return "\n".join(lines)


# ============================================================
# FILTER SAMPLES
# ============================================================

def filter_samples(
    samples: list[dict[str, Any]],
    selected_configs: list[str],
    selected_splits: list[str],
    selected_reasons: list[str],
    search_query: str,
    match_all_reasons: bool,
) -> list[dict[str, Any]]:
    """
    Filter samples.

    When multiple removal reasons are selected:
        - OR mode: sample matches at least one selected reason.
        - AND mode: sample matches every selected reason.
    """

    filtered = []

    query = search_query.strip().lower()

    for item in samples:

        # ----------------------------------------------------
        # Config
        # ----------------------------------------------------

        config = str(item.get("config", ""))

        if (
            selected_configs
            and config not in selected_configs
        ):
            continue

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        split = str(item.get("split", ""))

        if (
            selected_splits
            and split not in selected_splits
        ):
            continue

        # ----------------------------------------------------
        # Removal reasons
        # ----------------------------------------------------

        reasons = set(
            get_removal_reasons(item)
        )

        if selected_reasons:

            selected = set(selected_reasons)

            if match_all_reasons:

                if not selected.issubset(reasons):
                    continue

            else:

                if not selected.intersection(reasons):
                    continue

        # ----------------------------------------------------
        # Search ID / Question / Answers
        # ----------------------------------------------------

        if query:

            searchable_text = " ".join(
                [
                    get_sample_id(item),
                    get_question(item),
                    safe_string(item.get("expected_answer")),
                    safe_string(item.get("predicted_answer")),
                ]
            ).lower()

            if query not in searchable_text:
                continue

        filtered.append(item)

    return filtered


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
    }

    .sample-id {
        font-family: monospace;
        font-size: 0.9rem;
        color: #888888;
    }

    .passage-title {
        font-weight: 700;
        font-size: 1rem;
    }

    .passage-text {
        line-height: 1.7;
        text-align: justify;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.15);
        padding: 12px 16px;
        border-radius: 10px;
    }

    div.stButton > button {
        border-radius: 9px;
        min-height: 2.7rem;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">MuSiQue Sample Reviewer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Review teacher-generated samples removed during dataset cleaning.
        Inspect golden evidence and copy complete samples for manual evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

if not JSON_PATH.exists():

    st.error(
        f"JSON file not found:\n\n{JSON_PATH}"
    )

    st.info(
        "Run the dataset cleaning script first to generate "
        "teacher_training_removed_samples_v2.json."
    )

    st.stop()


try:

    data = load_removed_samples(
        str(JSON_PATH),
        JSON_PATH.stat().st_mtime,
    )

except Exception as error:

    st.error(
        f"Failed to load JSON: {error}"
    )

    st.stop()


metadata = data["metadata"]

all_samples = data["removed_samples"]

# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown("### Dataset overview")

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

overview_col1.metric(
    "Total removed",
    f"{len(all_samples):,}",
)

overview_col2.metric(
    "2-hop",
    sum(
        item.get("config") == "2-hop"
        for item in all_samples
    ),
)

overview_col3.metric(
    "3-hop",
    sum(
        item.get("config") == "3-hop"
        for item in all_samples
    ),
)

overview_col4.metric(
    "4-hop",
    sum(
        item.get("config") == "4-hop"
        for item in all_samples
    ),
)

st.caption(
    f"Dataset: {metadata.get('dataset', 'Unknown')} "
    f"| Mode: {metadata.get('mode', 'Unknown')} "
    f"| JSON: {JSON_PATH}"
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Filters")

st.sidebar.caption(
    "Narrow down the samples you want to review."
)

# ------------------------------------------------------------
# Config filter
# ------------------------------------------------------------

available_configs = sorted(
    {
        str(item.get("config", "Unknown"))
        for item in all_samples
    }
)

selected_configs = st.sidebar.multiselect(
    "Hop configuration",
    options=available_configs,
    default=available_configs,
)

# ------------------------------------------------------------
# Split filter
# ------------------------------------------------------------

available_splits = sorted(
    {
        str(item.get("split", "Unknown"))
        for item in all_samples
    }
)

selected_splits = st.sidebar.multiselect(
    "Dataset split",
    options=available_splits,
    default=available_splits,
)

# ------------------------------------------------------------
# Removal reason filter
# ------------------------------------------------------------

available_reasons = sorted(
    {
        reason
        for item in all_samples
        for reason in get_removal_reasons(item)
    }
)

selected_reasons = st.sidebar.multiselect(
    "Removal reasons",
    options=available_reasons,
    default=[],
    format_func=lambda reason: REASON_LABELS.get(
        reason,
        reason,
    ),
)

match_all_reasons = st.sidebar.checkbox(
    "Require ALL selected reasons",
    value=False,
    help=(
        "Disabled: OR logic, matching any selected reason. "
        "Enabled: AND logic, requiring every selected reason."
    ),
)

# ------------------------------------------------------------
# Search
# ------------------------------------------------------------

search_query = st.sidebar.text_input(
    "Search samples",
    placeholder="Sample ID, question, answer...",
)

# ------------------------------------------------------------
# Reset filters
# ------------------------------------------------------------

if st.sidebar.button(
    "Reset filters",
    use_container_width=True,
):

    st.session_state.pop(
        "review_cursor",
        None,
    )

    st.session_state.pop(
        "review_filter_signature",
        None,
    )

    st.rerun()


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_samples = filter_samples(
    samples=all_samples,
    selected_configs=selected_configs,
    selected_splits=selected_splits,
    selected_reasons=selected_reasons,
    search_query=search_query,
    match_all_reasons=match_all_reasons,
)

# ------------------------------------------------------------
# Filter statistics
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.metric(
    "Matching samples",
    f"{len(filtered_samples):,}",
)

st.sidebar.metric(
    "Total samples",
    f"{len(all_samples):,}",
)


# ============================================================
# EMPTY RESULT
# ============================================================

if not filtered_samples:

    st.warning(
        "No samples match the current filters."
    )

    st.info(
        "Try changing the hop configuration, removal reasons, "
        "or search query."
    )

    st.stop()


# ============================================================
# NAVIGATION STATE
# ============================================================

# Reset cursor when filters change.
filter_signature = (
    tuple(selected_configs),
    tuple(selected_splits),
    tuple(selected_reasons),
    search_query,
    match_all_reasons,
)

if (
    st.session_state.get("review_filter_signature")
    != filter_signature
):

    st.session_state.review_cursor = 0

    st.session_state.review_filter_signature = (
        filter_signature
    )


if "review_cursor" not in st.session_state:

    st.session_state.review_cursor = 0


# Keep cursor within valid bounds.
st.session_state.review_cursor = max(
    0,
    min(
        st.session_state.review_cursor,
        len(filtered_samples) - 1,
    ),
)


# ============================================================
# CURRENT SAMPLE
# ============================================================

cursor = st.session_state.review_cursor

current_sample = filtered_samples[cursor]

sample = get_sample_data(current_sample)

sample_id = get_sample_id(current_sample)

reasons = get_removal_reasons(current_sample)

golden_passages = get_golden_passages(
    current_sample
)


# ============================================================
# SAMPLE NAVIGATION HEADER
# ============================================================

st.divider()

nav_left, nav_center, nav_right = st.columns(
    [1, 3, 1]
)

with nav_left:

    if st.button(
        "← Previous",
        disabled=(cursor == 0),
        use_container_width=True,
    ):

        st.session_state.review_cursor -= 1

        st.rerun()


with nav_center:

    st.markdown(
        f"<h3 style='text-align:center; margin:0;'>"
        f"Sample {cursor + 1:,} / {len(filtered_samples):,}"
        f"</h3>",
        unsafe_allow_html=True,
    )

    st.progress(
        (cursor + 1) / len(filtered_samples)
    )


with nav_right:

    if st.button(
        "Next →",
        disabled=(cursor == len(filtered_samples) - 1),
        use_container_width=True,
    ):

        st.session_state.review_cursor += 1

        st.rerun()


# ============================================================
# SAMPLE IDENTIFICATION
# ============================================================

st.markdown("## Sample details")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.metric(
        "Configuration",
        safe_string(current_sample.get("config")),
    )

with info_col2:

    st.metric(
        "Split",
        safe_string(current_sample.get("split")),
    )

with info_col3:

    st.metric(
        "Original index",
        safe_string(current_sample.get("index")),
    )


st.markdown(
    f'<div class="sample-id">ID: {sample_id}</div>',
    unsafe_allow_html=True,
)


# ============================================================
# REMOVAL REASONS
# ============================================================

st.markdown("### Removal reasons")

if reasons:

    reason_cols = st.columns(
        min(len(reasons), 4)
    )

    for index, reason in enumerate(reasons):

        with reason_cols[index % len(reason_cols)]:

            with st.container(border=True):

                st.markdown(
                    f"**{REASON_LABELS.get(reason, reason)}**"
                )

                st.caption(
                    REASON_DESCRIPTIONS.get(
                        reason,
                        "No description available.",
                    )
                )

else:

    st.info("No removal reasons recorded.")


# ============================================================
# QUESTION
# ============================================================

st.divider()

st.markdown("### Question")

st.markdown(
    f"**{get_question(current_sample)}**"
)


# ============================================================
# ANSWER COMPARISON
# ============================================================

st.markdown("### Answer comparison")

answer_col1, answer_col2 = st.columns(2)

with answer_col1:

    with st.container(border=True):

        st.caption("GROUND TRUTH")

        st.markdown(
            f"### {safe_string(current_sample.get('expected_answer'))}"
        )


with answer_col2:

    with st.container(border=True):

        st.caption("TEACHER PREDICTION")

        st.markdown(
            f"### {safe_string(current_sample.get('predicted_answer'))}"
        )


# ============================================================
# TEACHER GENERATION OUTPUT
# ============================================================

st.divider()

st.markdown("### Teacher generation output")

teacher_output = sample.get(
    "teacher_generation_output"
)

if teacher_output is None:

    st.error(
        "teacher_generation_output is None."
    )

elif not str(teacher_output).strip():

    st.warning(
        "teacher_generation_output is empty."
    )

else:

    st.text_area(
        "Complete teacher output",
        value=str(teacher_output),
        height=300,
        disabled=True,
        label_visibility="collapsed",
        key=f"teacher_output_{cursor}",
    )


# ============================================================
# GOLDEN PASSAGES ONLY
# ============================================================

st.divider()

st.markdown("### Golden passages")

st.caption(
    "Only passages with is_supporting = True are displayed."
)

if not golden_passages:

    st.warning(
        "No golden/supporting passages found in this sample."
    )

else:

    st.caption(
        f"{len(golden_passages)} golden passage(s)"
    )

    for index, passage in enumerate(
        golden_passages,
        start=1,
    ):

        title = safe_string(
            passage.get("title", "Untitled")
        )

        paragraph_idx = safe_string(
            passage.get("idx", "N/A")
        )

        paragraph_text = safe_string(
            passage.get("paragraph_text", "")
        )

        with st.container(border=True):

            passage_col1, passage_col2 = st.columns(
                [4, 1]
            )

            with passage_col1:

                st.markdown(
                    f"**{index}. {title}**"
                )

            with passage_col2:

                st.caption(
                    f"Paragraph idx: {paragraph_idx}"
                )

            st.markdown(
                paragraph_text
            )


# ============================================================
# COMPLETE SAMPLE COPY AREA
# ============================================================

st.divider()

st.markdown("## Copy complete sample")

st.markdown(
    """
    Use the copy button in the code block below to copy
    the complete review context.

    It includes review metadata, removal reasons, expected
    and predicted answers, golden passages, and the entire
    original sample JSON.
    """
)

copy_text = build_copy_text(
    current_sample
)

# st.code provides a built-in copy button.
st.code(
    copy_text,
    language="text",
    wrap_lines=True,
)

# ============================================================
# FOOTER NAVIGATION
# ============================================================

st.divider()

footer_left, footer_center, footer_right = st.columns(
    [1, 2, 1]
)

with footer_left:

    if st.button(
        "← Previous sample",
        disabled=(cursor == 0),
        use_container_width=True,
        key="footer_previous",
    ):

        st.session_state.review_cursor -= 1

        st.rerun()


with footer_center:

    st.caption(
        f"Currently reviewing: {sample_id}"
    )


with footer_right:

    if st.button(
        "Next sample →",
        disabled=(cursor == len(filtered_samples) - 1),
        use_container_width=True,
        key="footer_next",
    ):

        st.session_state.review_cursor += 1

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:gray;
        font-size:0.8rem;
        margin-top:2rem;
    ">
        MuSiQue Teacher Training Dataset Reviewer · ART Project
    </div>
    """,
    unsafe_allow_html=True,
)
