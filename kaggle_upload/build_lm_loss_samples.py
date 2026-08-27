import random
from typing import Any

from model_input import (
    SampleObjectForLMLoss,
    Document,
)


# ============================================================
# CONFIG
# ============================================================

MAX_DOCS = 3

RANDOM_SEED = 42


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================
#
# Use a local RNG so that sampling in this module does not
# modify the global random state of the rest of the project.
#
# ============================================================

_rng = random.Random(RANDOM_SEED)


# ============================================================
# GENERATE DOCUMENT RATIOS
# ============================================================

def _generate_document_ratios(
    max_docs: int,
) -> list[tuple[int, int]]:
    """
    Generate all possible (positive, negative) document
    compositions whose total number of documents equals
    max_docs.

    Example
    -------
    max_docs = 3

    Returns
    -------
    [
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
    ]
    """

    ratios = []

    for num_positive in range(
        max_docs,
        -1,
        -1,
    ):

        num_negative = (
            max_docs
            - num_positive
        )

        ratios.append(
            (
                num_positive,
                num_negative,
            )
        )

    return ratios


# ============================================================
# PRECOMPUTED DOCUMENT RATIOS
# ============================================================
#
# Generated once when this module is imported.
#
# With MAX_DOCS = 3:
#
#     (3, 0)
#     (2, 1)
#     (1, 2)
#     (0, 3)
#
# ============================================================

DOCUMENT_RATIOS = _generate_document_ratios(
    MAX_DOCS
)


# ============================================================
# CREATE LM-LOSS SAMPLES
# ============================================================

def create_lm_loss_samples(
    sample: dict[str, Any],
) -> dict[
    tuple[int, int],
    SampleObjectForLMLoss,
]:
    """
    Create SampleObjectForLMLoss instances for every valid
    positive/negative document composition.

    A random pool of at most MAX_DOCS positive documents and
    at most MAX_DOCS negative documents is sampled first.

    The sampled pools are then reused across all document
    ratio conditions.

    Therefore, the same underlying documents are used for
    all valid ratios of one sample.

    Conditions
    ----------
    (3, 0):
        3 positive + 0 negative

    (2, 1):
        2 positive + 1 negative

    (1, 2):
        1 positive + 2 negative

    (0, 3):
        0 positive + 3 negative

    A condition is skipped if the sample does not contain
    enough documents of the required type.

    Parameters
    ----------
    sample:
        Teacher-generated hypothesis validation sample.

    Returns
    -------
    dict[tuple[int, int], SampleObjectForLMLoss]

        Mapping:

            (num_positive, num_negative)
                ->
            SampleObjectForLMLoss
    """

    # ========================================================
    # 1. GET QUESTION / ANSWER
    # ========================================================

    question = sample["question"]
    answer = sample["teacher_generation_output"]

    # ========================================================
    # 2. SPLIT POSITIVE / NEGATIVE DOCUMENTS
    # ========================================================

    positive_paragraphs = [
        paragraph
        for paragraph in sample["paragraphs"]
        if paragraph["is_supporting"]
    ]

    negative_paragraphs = [
        paragraph
        for paragraph in sample["paragraphs"]
        if not paragraph["is_supporting"]
    ]

    # ========================================================
    # 3. RANDOMLY SAMPLE DOCUMENT POOLS
    # ========================================================
    #
    # We only need MAX_DOCS documents from each category.
    #
    # random.sample() guarantees that documents within each
    # category are unique.
    #
    # The pools are sampled ONCE per input sample.
    #
    # Therefore every ratio condition is constructed from
    # the same sampled positive/negative pools.
    #
    # ========================================================

    num_positive_to_sample = min(
        len(positive_paragraphs),
        MAX_DOCS,
    )

    num_negative_to_sample = min(
        len(negative_paragraphs),
        MAX_DOCS,
    )

    positive_paragraphs = _rng.sample(
        positive_paragraphs,
        num_positive_to_sample,
    )

    negative_paragraphs = _rng.sample(
        negative_paragraphs,
        num_negative_to_sample,
    )

    # ========================================================
    # 4. CONVERT PARAGRAPHS -> Document
    # ========================================================

    positive_documents = [
        Document(
            title=paragraph["title"],
            content=paragraph["paragraph_text"],
        )
        for paragraph in positive_paragraphs
    ]

    negative_documents = [
        Document(
            title=paragraph["title"],
            content=paragraph["paragraph_text"],
        )
        for paragraph in negative_paragraphs
    ]

    # ========================================================
    # 5. CREATE SAMPLES FOR EACH RATIO
    # ========================================================

    results = {}

    for (
        num_positive,
        num_negative,
    ) in DOCUMENT_RATIOS:

        # ----------------------------------------------------
        # Check whether enough positive documents exist.
        # ----------------------------------------------------

        if len(positive_documents) < num_positive:
            continue

        # ----------------------------------------------------
        # Check whether enough negative documents exist.
        # ----------------------------------------------------

        if len(negative_documents) < num_negative:
            continue

        # ----------------------------------------------------
        # Select positive documents.
        #
        # Selection is deterministic from the already sampled
        # positive pool.
        # ----------------------------------------------------

        selected_positive = positive_documents[
            :num_positive
        ]

        # ----------------------------------------------------
        # Select negative documents.
        #
        # Selection is deterministic from the already sampled
        # negative pool.
        # ----------------------------------------------------

        selected_negative = negative_documents[
            :num_negative
        ]

        # ----------------------------------------------------
        # Combine documents.
        # ----------------------------------------------------

        selected_documents = (
            selected_positive
            + selected_negative
        )

        # ----------------------------------------------------
        # Sanity check.
        # ----------------------------------------------------

        assert (
            len(selected_documents)
            == MAX_DOCS
        )

        # ----------------------------------------------------
        # Create SampleObjectForLMLoss.
        # ----------------------------------------------------

        lm_loss_sample = SampleObjectForLMLoss(
            question=question,
            documents=selected_documents,
            answer=answer,
        )

        results[
            (
                num_positive,
                num_negative,
            )
        ] = lm_loss_sample

    return results


# # ============================================================
# # TEST
# # ============================================================

# def test_create_lm_loss_samples():

#     from shared.MuSiQue.load_hypothesis_validation import (
#         load_teacher_generated_hypothesis_validation,
#     )

#     print()
#     print("=" * 80)
#     print("TEST — CREATE LM LOSS SAMPLES")
#     print("=" * 80)

#     print()
#     print(f"MAX_DOCS     : {MAX_DOCS}")
#     print(f"RANDOM_SEED  : {RANDOM_SEED}")

#     datasets = (
#         load_teacher_generated_hypothesis_validation()
#     )

#     expected_ratios = {
#         "2-hop": [
#             (2, 1),
#             (1, 2),
#             (0, 3),
#         ],
#         "3-hop": [
#             (3, 0),
#             (2, 1),
#             (1, 2),
#             (0, 3),
#         ],
#         "4-hop": [
#             (3, 0),
#             (2, 1),
#             (1, 2),
#             (0, 3),
#         ],
#     }

#     # ========================================================
#     # TEST EVERY HOP
#     # ========================================================

#     for hop, samples in datasets.items():

#         print()
#         print("=" * 80)
#         print(f"{hop.upper()} — TESTING")
#         print("=" * 80)

#         # ----------------------------------------------------
#         # Use the first real validation sample.
#         # ----------------------------------------------------

#         sample = samples[0]

#         results = create_lm_loss_samples(
#             sample
#         )

#         expected = expected_ratios[hop]

#         # ----------------------------------------------------
#         # Check generated ratios.
#         # ----------------------------------------------------

#         assert list(results.keys()) == expected, (
#             f"{hop}: unexpected ratios.\n"
#             f"Expected: {expected}\n"
#             f"Got:      {list(results.keys())}"
#         )

#         # ----------------------------------------------------
#         # Check every generated condition.
#         # ----------------------------------------------------

#         for ratio, lm_sample in results.items():

#             num_positive, num_negative = ratio

#             # ------------------------------------------------
#             # Total documents must equal MAX_DOCS.
#             # ------------------------------------------------

#             assert (
#                 len(lm_sample.documents)
#                 == MAX_DOCS
#             )

#             # ------------------------------------------------
#             # Count positive / negative documents.
#             # ------------------------------------------------

#             positive_count = 0
#             negative_count = 0

#             for document in lm_sample.documents:

#                 matching_paragraph = next(
#                     paragraph
#                     for paragraph in sample["paragraphs"]
#                     if (
#                         paragraph["title"]
#                         == document.title
#                         and
#                         paragraph["paragraph_text"]
#                         == document.content
#                     )
#                 )

#                 if matching_paragraph["is_supporting"]:
#                     positive_count += 1
#                 else:
#                     negative_count += 1

#             # ------------------------------------------------
#             # Check composition.
#             # ------------------------------------------------

#             assert (
#                 positive_count
#                 == num_positive
#             ), (
#                 f"{hop} {ratio}: "
#                 f"expected {num_positive} positive docs, "
#                 f"got {positive_count}"
#             )

#             assert (
#                 negative_count
#                 == num_negative
#             ), (
#                 f"{hop} {ratio}: "
#                 f"expected {num_negative} negative docs, "
#                 f"got {negative_count}"
#             )

#             # ------------------------------------------------
#             # Check question / answer.
#             # ------------------------------------------------

#             assert (
#                 lm_sample.question
#                 == sample["question"]
#             )

#             assert (
#                 lm_sample.answer
#                 == sample["answer"]
#             )

#             print(
#                 f"  ✓ {num_positive} positive "
#                 f"+ {num_negative} negative"
#             )

#             print(
#                 "    Documents:"
#             )

#             for document in lm_sample.documents:

#                 print(
#                     f"      - {document.title}"
#                 )

#         # ----------------------------------------------------
#         # Check impossible conditions.
#         # ----------------------------------------------------

#         if hop == "2-hop":

#             assert (3, 0) not in results

#             print(
#                 "  ✓ (3, 0) correctly skipped "
#                 "(only 2 positive documents)"
#             )

#         else:

#             assert (3, 0) in results

#         print()
#         print(
#             f"✓ {hop.upper()} passed"
#         )

#     # ========================================================
#     # FINAL
#     # ========================================================

#     print()
#     print("=" * 80)
#     print("✓ ALL HOP TESTS PASSED")
#     print("=" * 80)


# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":

#     test_create_lm_loss_samples()