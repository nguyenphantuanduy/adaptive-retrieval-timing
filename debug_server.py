import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from shared.MuSiQue.load_dataset import load_musique_ans


# ============================================================
# CONFIGURATION
# ============================================================

# DEFAULT_INPUT_FILE = (
#     r"REVIVE_training\result\rsearch_musique_art_results.jsonl"
# )

DEFAULT_INPUT_FILE = r"art-backup\demo-3\result\threshold60.jsonl"

TOP_K = 5
# DEFAULT_INPUT_FILE = (
#     r"REVIVE_training\result\rsearch_musique_art_results.jsonl"
# )

# TOP_K = 3

# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(text: str) -> tuple[str, bool]:
    """Extract the content inside the last <answer>...</answer> tag."""
    matches = list(
        re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL)
    )

    if matches:
        return matches[-1].group(1).strip(), True

    return text.strip(), False


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_question(question: Any) -> str:
    if not isinstance(question, str):
        return ""
    return " ".join(question.split()).casefold()


def safe_json(value: Any) -> str:
    """Serialize arbitrary data for the copyable debug report."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


def as_list(value: Any) -> list:
    """Normalize a value into a list without splitting strings into chars."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def get_passage_text(passage: Any) -> str:
    """Return a passage as readable text, including plain-string passages."""
    if passage is None:
        return ""

    if isinstance(passage, str):
        return passage

    if isinstance(passage, dict):
        for key in (
            "paragraph_text",
            "text",
            "contents",
            "content",
            "passage",
            "document",
        ):
            value = passage.get(key)
            if isinstance(value, str) and value:
                return value
        return safe_json(passage)

    if isinstance(passage, list):
        return "\n".join(get_passage_text(item) for item in passage)

    return str(passage)


def get_query_text(query: Any) -> str:
    if isinstance(query, str):
        return query

    if isinstance(query, dict):
        for key in ("query", "text", "search_query", "content"):
            value = query.get(key)
            if value is not None:
                return str(value)

    return safe_json(query)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def render_value(value: Any):
    """Render strings, lists, and other JSON values readably."""
    if value is None:
        st.caption("(empty)")
    elif isinstance(value, str):
        if value:
            st.text(value)
        else:
            st.caption("(empty string)")
    else:
        st.json(value)


# ============================================================
# LOAD ART JSONL
# ============================================================

@st.cache_data(show_spinner="Loading ART results...")
def load_samples(file_path: str) -> list[dict]:
    path = Path(file_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"ART result file not found: {path.resolve()}"
        )

    samples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if isinstance(item, dict):
                samples.append(item)

    return samples


# ============================================================
# LOAD ORIGINAL MuSiQue-Ans
# ============================================================

@st.cache_data(show_spinner="Loading original MuSiQue-Ans dataset...")
def load_original_dataset() -> dict:
    return load_musique_ans()


@st.cache_data(show_spinner="Building question index...")
def build_question_index() -> dict:
    dataset = load_original_dataset()
    question_index = {}

    for split_name in ("train", "dev", "test"):
        for sample in dataset.get(split_name, []):
            key = normalize_question(sample.get("question"))

            if key:
                question_index.setdefault(key, []).append(
                    {
                        "split": split_name,
                        "sample": sample,
                    }
                )

    return question_index


def find_original_matches(question: str) -> list[dict]:
    key = normalize_question(question)
    if not key:
        return []

    return build_question_index().get(key, [])


# ============================================================
# ORIGINAL DATASET HELPERS
# ============================================================

def build_paragraph_lookup(sample: dict) -> dict:
    """Index valid paragraph dictionaries by their original paragraph ID."""
    paragraphs = as_list(sample.get("paragraphs", []))
    return {
        paragraph.get("idx"): paragraph
        for paragraph in paragraphs
        if isinstance(paragraph, dict) and paragraph.get("idx") is not None
    }


def render_reasoning_path(original_sample: dict):
    decomposition = original_sample.get("question_decomposition", [])
    paragraph_lookup = build_paragraph_lookup(original_sample)

    if not decomposition:
        st.info("No question decomposition available.")
        return

    items = []

    for step_number, step in enumerate(as_list(decomposition), start=1):
        if not isinstance(step, dict):
            items.append(f"**Step {step_number}:** Invalid decomposition entry")
            continue

        support = paragraph_lookup.get(
            step.get("paragraph_support_idx")
        )
        title = (
            support.get("title", "Unknown document")
            if support
            else "Supporting document unavailable"
        )

        items.append(
            f"**Step {step_number}:** "
            f"{step.get('question', 'N/A')}\n\n"
            f"- Answer: `{step.get('answer', 'N/A')}`\n"
            f"- Supporting document: **{title}**"
        )

    st.markdown("\n\n".join(items))


def render_reasoning_decomposition(original_sample: dict):
    decomposition = original_sample.get("question_decomposition", [])
    paragraph_lookup = build_paragraph_lookup(original_sample)

    if not decomposition:
        st.info("No question decomposition available.")
        return

    for number, step in enumerate(as_list(decomposition), start=1):
        if not isinstance(step, dict):
            with st.expander(f"Step {number}", expanded=False):
                st.warning("Invalid decomposition entry; expected an object.")
            continue

        support_idx = step.get("paragraph_support_idx")
        support = paragraph_lookup.get(support_idx)

        with st.expander(f"Step {number}", expanded=False):
            st.markdown("**Sub-question**")
            st.write(step.get("question", "N/A"))

            st.markdown("**Intermediate answer**")
            st.write(step.get("answer", "N/A"))

            st.markdown("**Supporting document**")
            if support:
                st.markdown(f"**{support.get('title', 'Untitled')}**")
                st.write(support.get("paragraph_text", ""))
            else:
                st.warning(
                    f"Could not resolve supporting paragraph index "
                    f"{support_idx!r}."
                )


def render_golden_documents(original_sample: dict):
    paragraphs = as_list(original_sample.get("paragraphs", []))
    golden = [
        paragraph for paragraph in paragraphs
        if isinstance(paragraph, dict)
        and paragraph.get("is_supporting") is True
    ]

    st.markdown("### Golden documents")

    if not golden:
        st.warning("No paragraphs marked is_supporting=True.")
        return

    st.caption(f"Supporting documents: {len(golden)}")

    for number, paragraph in enumerate(golden, start=1):
        title = paragraph.get("title", "Untitled document")
        idx = paragraph.get("idx", "N/A")

        with st.expander(f"{number}. {title}", expanded=False):
            st.caption(f"Original paragraph index: {idx}")
            st.write(paragraph.get("paragraph_text", ""))


def render_original_sample(original_sample: dict, split_name: str):
    st.subheader("Original MuSiQue-Ans sample")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Dataset split", split_name)

    with col2:
        st.write("**Original ID**")
        st.code(str(original_sample.get("id", "N/A")))

    st.markdown("**Original question**")
    st.write(original_sample.get("question", "N/A"))

    st.markdown("**Original gold answer**")
    st.success(str(original_sample.get("answer", "N/A")))

    aliases = original_sample.get("answer_aliases", [])
    if aliases:
        st.markdown("**Answer aliases**")
        st.write(aliases)

    st.write("**Answerable**")
    st.write(original_sample.get("answerable", "N/A"))

    st.divider()
    st.markdown("### Reasoning path")
    render_reasoning_path(original_sample)

    st.divider()
    st.markdown("### Reasoning decomposition")
    render_reasoning_decomposition(original_sample)

    st.divider()
    render_golden_documents(original_sample)


# ============================================================
# GROUP RETRIEVED PASSAGES BY QUERY
# ============================================================

def group_passages_by_query(
    queries: list,
    passages: list,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Assumption:
        Passages are stored in query order, with top-k passages
        for each query stored consecutively.

    Example for top_k=5:
        Query 1 -> passages [0:5]
        Query 2 -> passages [5:10]
        Query 3 -> passages [10:15]

    If there are more passages than queries can account for,
    create additional groups so no passage is discarded.
    """
    queries = as_list(queries)
    passages = as_list(passages)

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    number_of_groups = max(
        len(queries),
        (len(passages) + top_k - 1) // top_k,
    )

    groups = []

    for group_index in range(number_of_groups):
        start = group_index * top_k
        end = min(start + top_k, len(passages))

        group_passages = passages[start:end]
        query = (
            queries[group_index]
            if group_index < len(queries)
            else None
        )

        groups.append(
            {
                "query_index": group_index + 1,
                "query": query,
                "query_text": (
                    get_query_text(query)
                    if query is not None
                    else "(No corresponding query found)"
                ),
                "passage_start": start + 1 if group_passages else None,
                "passage_end": end if group_passages else None,
                "passages": group_passages,
                "expected_top_k": top_k,
                "actual_count": len(group_passages),
            }
        )

    return groups


# ============================================================
# COPYABLE DEBUG REPORT
# ============================================================

def build_debug_report(
    sample: dict,
    sample_index: int,
    total_samples: int,
    queries: list,
    passages: list,
    groups: list[dict],
    generated_text: str,
    predicted_answer: str,
    has_answer_tag: bool,
    original_match: dict | None,
    matching_count: int,
) -> str:
    """Build one self-contained report for copying into ChatGPT."""
    report = []

    def section(title: str):
        report.append("")
        report.append("=" * 80)
        report.append(title)
        report.append("=" * 80)

    section("ART SAMPLE METADATA")
    report.append(f"Sample position: {sample_index + 1}/{total_samples}")
    report.append(f"Sample ID: {sample.get('id', 'N/A')}")
    report.append(f"Top-k grouping: {TOP_K}")
    report.append(f"Number of queries: {len(queries)}")
    report.append(f"Number of retrieved passages: {len(passages)}")
    report.append(f"Number of query groups: {len(groups)}")

    section("QUESTION")
    report.append(str(sample.get("question", "N/A")))

    section("ART GOLDEN ANSWERS")
    report.append(safe_json(sample.get("golden_answers", "N/A")))

    section("PREDICTED ANSWER")
    report.append(f"Answer tag found: {has_answer_tag}")
    report.append(predicted_answer)

    section("RETRIEVAL QUERIES AND PASSAGES")

    for group in groups:
        report.append("")
        report.append("-" * 80)
        report.append(f"QUERY {group['query_index']}")
        report.append("-" * 80)
        report.append(group["query_text"])
        report.append(
            f"Passage range: {group['passage_start']}–"
            f"{group['passage_end']} "
            f"(actual={group['actual_count']}, expected={TOP_K})"
        )

        for local_index, passage in enumerate(
            group["passages"], start=1
        ):
            global_index = group["passage_start"] + local_index - 1
            report.append("")
            report.append(
                f"[Passage {global_index} | "
                f"Rank {local_index} for Query {group['query_index']}]"
            )
            report.append(get_passage_text(passage))

    section("RAW MODEL GENERATION")
    report.append(generated_text)

    section("ORIGINAL MuSiQue-Ans MATCH")
    report.append(
        f"Number of exact normalized-question matches: {matching_count}"
    )

    if original_match:
        original = original_match["sample"]
        report.append(f"Split: {original_match['split']}")
        report.append(f"Original ID: {original.get('id', 'N/A')}")
        report.append(f"Original question: {original.get('question', 'N/A')}")
        report.append(f"Gold answer: {original.get('answer', 'N/A')}")
        report.append(
            "Answer aliases: "
            + safe_json(original.get("answer_aliases", []))
        )
        report.append(
            "Question decomposition:\n"
            + safe_json(original.get("question_decomposition", []))
        )

        report.append("")
        report.append("GOLDEN SUPPORTING DOCUMENTS:")

        golden = [
            paragraph
            for paragraph in as_list(original.get("paragraphs", []))
            if isinstance(paragraph, dict)
            and paragraph.get("is_supporting") is True
        ]

        if not golden:
            report.append("(No paragraphs marked is_supporting=True.)")

        for number, paragraph in enumerate(golden, start=1):
            report.append("")
            report.append(
                f"[Golden document {number}] "
                f"Title: {paragraph.get('title', 'Untitled')}"
            )
            report.append(
                f"Paragraph index: {paragraph.get('idx', 'N/A')}"
            )
            report.append(paragraph.get("paragraph_text", ""))

    else:
        report.append(
            "No matching original sample found. "
            "Ground-truth decomposition and supporting documents unavailable."
        )

    section("FULL ORIGINAL ART SAMPLE JSON")
    report.append(safe_json(sample))

    return "\n".join(report).strip()


# ============================================================
# PASSAGE DISPLAY
# ============================================================

def render_passage(passage: Any, global_rank: int, local_rank: int):
    """Render a retrieved passage, whether it is a string or structured data."""
    with st.expander(
        f"Rank {local_rank} · Passage {global_rank}",
        expanded=False,
    ):
        st.markdown("**Passage text**")
        st.text(get_passage_text(passage))


def render_retrieval_groups(groups: list[dict]):
    st.subheader("Retrieval Debug · Query → Top-5 Documents")

    if not groups:
        st.info("No retrieval queries or passages available.")
        return

    for group in groups:
        query_number = group["query_index"]
        count = group["actual_count"]
        expected = group["expected_top_k"]

        with st.container(border=True):
            st.markdown(f"### Query {query_number}")
            st.write(group["query_text"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Retrieved passages", count)
            with col2:
                st.metric("Expected top-k", expected)

            if count == 0:
                st.warning("No passages assigned to this query group.")
            elif count < expected:
                st.warning(
                    f"This group contains {count} passages, "
                    f"fewer than top-k={expected}."
                )

            start = group["passage_start"]

            for local_rank, passage in enumerate(
                group["passages"], start=1
            ):
                global_rank = start + local_rank - 1
                render_passage(passage, global_rank, local_rank)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="ART Sample Reviewer",
    page_icon="🔎",
    layout="wide",
)

st.title("ART Sample Reviewer")
st.caption(
    "Inspect ART predictions, retrieval hops, supporting evidence, "
    "and the original MuSiQue-Ans ground truth."
)


# ============================================================
# INPUT FILE
# ============================================================

input_file = st.text_input(
    "ART JSONL file path",
    value=DEFAULT_INPUT_FILE,
    help="Use a path relative to the project root or an absolute path.",
)

col_load, col_hint = st.columns([1, 3])

with col_load:
    load_clicked = st.button(
        "Load / Reload file",
        type="primary",
        use_container_width=True,
    )

with col_hint:
    st.caption(
        "Reload after changing the JSONL file contents. "
        "Grouping assumes passages are ordered by query, top-k at a time."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "loaded_path" not in st.session_state:
    st.session_state.loaded_path = None

if "samples" not in st.session_state:
    st.session_state.samples = []

if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0


# ============================================================
# LOAD ART RESULTS
# ============================================================

if load_clicked or (
    st.session_state.loaded_path != input_file
    and input_file.strip()
):
    try:
        load_samples.clear()
        st.session_state.samples = load_samples(input_file.strip())
        st.session_state.loaded_path = input_file.strip()
        st.session_state.sample_index = 0

    except (OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()


samples = st.session_state.samples

if not samples:
    st.info(
        "Enter a valid JSONL path and click **Load / Reload file**."
    )
    st.stop()


# ============================================================
# SAMPLE NAVIGATION
# ============================================================

st.session_state.sample_index = max(
    0,
    min(st.session_state.sample_index, len(samples) - 1),
)

index = st.session_state.sample_index
sample = samples[index]

col_prev, col_status, col_next = st.columns([1, 3, 1])

with col_prev:
    if st.button(
        "← Previous",
        disabled=index == 0,
        use_container_width=True,
    ):
        st.session_state.sample_index -= 1
        st.rerun()

with col_status:
    st.markdown(
        f"<div style='text-align:center; padding-top:0.4rem;'>"
        f"<b>Sample {index + 1} / {len(samples)}</b>"
        f" &nbsp; | &nbsp; ID: "
        f"<b>{sample.get('id', 'N/A')}</b></div>",
        unsafe_allow_html=True,
    )

with col_next:
    if st.button(
        "Next →",
        disabled=index >= len(samples) - 1,
        use_container_width=True,
    ):
        st.session_state.sample_index += 1
        st.rerun()


# ============================================================
# PREPARE SAMPLE DATA
# ============================================================

generated_text = sample.get("pure_art_answer", "")
if not isinstance(generated_text, str):
    generated_text = str(generated_text)

predicted_answer, has_answer_tag = extract_answer(generated_text)

queries = as_list(sample.get("pure_art_queries", []))
passages = as_list(sample.get("pure_art_passages", []))

groups = group_passages_by_query(
    queries=queries,
    passages=passages,
    top_k=TOP_K,
)

question = sample.get("question", "")
matches = find_original_matches(question)

selected_match = matches[0] if matches else None


# ============================================================
# SAMPLE OVERVIEW
# ============================================================

st.divider()
st.header("1. Sample Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Queries", len(queries))

with col2:
    st.metric("Retrieved passages", len(passages))

with col3:
    st.metric("Query groups", len(groups))

with col4:
    st.metric("Original matches", len(matches))

st.markdown("**Question**")
st.write(question)

col_gold, col_pred = st.columns(2)

with col_gold:
    st.markdown("**ART golden answers**")
    render_value(sample.get("golden_answers", "N/A"))

with col_pred:
    st.markdown("**Extracted prediction**")
    if has_answer_tag:
        st.success("Found <answer>...</answer> tag.")
    else:
        st.warning("No answer tag; displaying full generation.")
    st.write(predicted_answer)


# ============================================================
# RETRIEVAL DEBUG BY QUERY
# ============================================================

st.divider()
st.header("2. Retrieval Evidence")
render_retrieval_groups(groups)


# ============================================================
# ORIGINAL GROUND TRUTH
# ============================================================

st.divider()
st.header("3. Original MuSiQue-Ans Ground Truth")

if not matches:
    st.warning(
        "No exact normalized-question match found in the original "
        "MuSiQue-Ans train/dev/test splits."
    )
else:
    if len(matches) > 1:
        st.warning(
            f"Found {len(matches)} matching records. "
            "Select the correct original record."
        )

        options = [
            (
                f"{match['split']} | "
                f"ID: {match['sample'].get('id', 'N/A')} | "
                f"Answer: {match['sample'].get('answer', 'N/A')}"
            )
            for match in matches
        ]

        selected_match_index = st.selectbox(
            "Matching original sample",
            options=range(len(matches)),
            format_func=lambda i: options[i],
            key=f"original_match_{index}",
        )
        selected_match = matches[selected_match_index]

    if selected_match:
        render_original_sample(
            selected_match["sample"],
            selected_match["split"],
        )


# ============================================================
# RAW MODEL OUTPUT
# ============================================================

st.divider()
st.header("4. Raw Model Output")

with st.expander("Raw Generation (pure_art_answer)", expanded=False):
    st.code(generated_text, language="text")

with st.expander("Raw Retrieval Queries (JSON)", expanded=False):
    st.code(safe_json(sample.get("pure_art_queries", [])), language="json")

with st.expander("Raw Retrieved Passages (JSON)", expanded=False):
    st.code(safe_json(sample.get("pure_art_passages", [])), language="json")


# ============================================================
# ONE-CLICK COPYABLE DEBUG REPORT
# ============================================================

st.divider()
st.header("5. Copy All Debug Information")

debug_report = build_debug_report(
    sample=sample,
    sample_index=index,
    total_samples=len(samples),
    queries=queries,
    passages=passages,
    groups=groups,
    generated_text=generated_text,
    predicted_answer=predicted_answer,
    has_answer_tag=has_answer_tag,
    original_match=selected_match,
    matching_count=len(matches),
)

st.caption(
    "The report includes the question, gold and predicted answers, "
    "queries, passages grouped by top-k, raw generation, original "
    "reasoning decomposition, golden supporting documents, and the "
    "full ART sample JSON."
)

st.code(debug_report, language="text")

st.caption(
    "Use the copy icon in the top-right corner of the text block above "
    "to copy the entire report, then paste it into ChatGPT."
)
