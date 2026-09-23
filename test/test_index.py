
import json

import faiss
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

REPO_ID = "adaptive-retrieval-timing/MuSiQue-ART-E5-Retrieval-Index"
REPO_TYPE = "dataset"

MODEL_NAME = "intfloat/e5-base-v2"

TOP_K = 3
MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TEST CASES
#
# IMPORTANT:
# Each retrieval query is EXACTLY ONE CoT SENTENCE.
#
# No question.
# No previous sentences.
# No cumulative context.
#
# target_title = document expected to be retrieved.
# Use None when there is no specific retrieval target.
# ============================================================

TEST_CASES = [

    {
        "name": "Hop 1",
        "sentence": (
            "To answer this, we first need to identify the city that used "
            "to be the U.S. capitol."
        ),
        "target_title": "United States Capitol",
    },

    {
        "name": "Hop 2",
        "sentence": (
            "Thus, Philadelphia is the city in question."
        ),
        "target_title": "Philadelphia",
    },

    {
        "name": "Hop 3",
        "sentence": (
            "This establishes the religion in question as the AME."
        ),
        "target_title": "Philadelphia",
    },

    {
        "name": "Hop 4",
        "sentence": (
            "The question then asks who started the Bethel branch of this "
            "religion."
        ),
        "target_title": "Mother Bethel A.M.E. Church",
    },
]


# ============================================================
# E5 MEAN POOLING
# ============================================================

def mean_pooling(
    last_hidden_state,
    attention_mask,
):

    mask = attention_mask[..., None].bool()

    masked_hidden = last_hidden_state.masked_fill(
        ~mask,
        0.0,
    )

    return masked_hidden.sum(dim=1) / attention_mask.sum(
        dim=1
    )[..., None]


# ============================================================
# LOAD FAISS + CORPUS
# ============================================================

def load_hf_artifact():

    print("=" * 80)
    print("LOADING RETRIEVAL ARTIFACT")
    print("=" * 80)

    print(f"\nRepo: {REPO_ID}")


    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print("\nLoading FAISS index...")

    index_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="e5_Flat.index",
        repo_type=REPO_TYPE,
    )

    index = faiss.read_index(index_path)

    print(f"Index vectors  : {index.ntotal}")
    print(f"Index dimension: {index.d}")


    # --------------------------------------------------------
    # CORPUS
    # --------------------------------------------------------

    print("\nLoading corpus...")

    corpus_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="musique_corpus.jsonl",
        repo_type=REPO_TYPE,
    )

    corpus = []

    with open(
        corpus_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            corpus.append(json.loads(line))


    print(f"Corpus passages: {len(corpus)}")


    # --------------------------------------------------------
    # ALIGNMENT
    # --------------------------------------------------------

    if len(corpus) != index.ntotal:

        raise RuntimeError(
            "Corpus / FAISS mismatch!\n"
            f"Corpus: {len(corpus)}\n"
            f"Index : {index.ntotal}"
        )

    print("\nCorpus / FAISS alignment: OK")

    return index, corpus


# ============================================================
# LOAD E5
# ============================================================

def load_e5():

    print("\n" + "=" * 80)
    print("LOADING E5")
    print("=" * 80)

    print(f"Model : {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
    )

    model.eval()
    model.to(DEVICE)

    return tokenizer, model


# ============================================================
# ENCODE ONE SENTENCE
# ============================================================

@torch.no_grad()
def encode_sentence(
    sentence,
    tokenizer,
    model,
):

    # E5 retrieval convention
    text = f"query: {sentence}"

    inputs = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model(
        **inputs,
        return_dict=True,
    )

    embedding = mean_pooling(
        outputs.last_hidden_state,
        inputs["attention_mask"],
    )

    embedding = F.normalize(
        embedding,
        p=2,
        dim=1,
    )

    return embedding.cpu().numpy().astype("float32")


# ============================================================
# SEARCH
# ============================================================

def search(
    sentence,
    index,
    corpus,
    tokenizer,
    model,
):

    query_embedding = encode_sentence(
        sentence,
        tokenizer,
        model,
    )

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):

        idx = int(idx)

        doc = corpus[idx]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "index": idx,
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
            }
        )

    return results


# ============================================================
# FIND TARGET
# ============================================================

def find_target(
    results,
    target_title,
):

    if target_title is None:
        return None

    for result in results:

        if result["title"] == target_title:
            return result

    return None


# ============================================================
# PRINT ONE TEST
# ============================================================

def print_test(
    test_case,
    results,
    target,
):

    print("\n")
    print("=" * 80)
    print(test_case["name"])
    print("=" * 80)

    print("\nSENTENCE:")
    print(test_case["sentence"])

    print("\nEXPECTED TARGET:")
    print(test_case["target_title"])

    print("\nTOP-3:")

    for result in results:

        print(
            f"\nRank {result['rank']}"
        )

        print(
            f"Score : {result['score']:.6f}"
        )

        print(
            f"ID    : {result['id']}"
        )

        print(
            f"Title : {result['title']}"
        )

        print(
            f"Content:\n{result['content']}"
        )


    print("\nTARGET CHECK:")

    if test_case["target_title"] is None:

        print("No target specified.")

    elif target is None:

        print("TARGET FOUND: NO")

    else:

        print("TARGET FOUND: YES")
        print(f"Rank : {target['rank']}")
        print(f"Score: {target['score']:.6f}")
        print(f"ID   : {target['id']}")


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load retrieval artifact
    # --------------------------------------------------------

    index, corpus = load_hf_artifact()


    # --------------------------------------------------------
    # Load E5
    # --------------------------------------------------------

    tokenizer, model = load_e5()


    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    all_results = []

    print("\n")
    print("=" * 80)
    print("SINGLE-SENTENCE MULTI-HOP RETRIEVAL TEST")
    print("=" * 80)

    print(
        "\nEach retrieval query consists of EXACTLY ONE sentence."
    )

    print(
        "\nNo question."
        "\nNo previous sentence."
        "\nNo cumulative context."
    )


    for test_case in TEST_CASES:

        sentence = test_case["sentence"]

        results = search(
            sentence=sentence,
            index=index,
            corpus=corpus,
            tokenizer=tokenizer,
            model=model,
        )

        target = find_target(
            results,
            test_case["target_title"],
        )

        print_test(
            test_case=test_case,
            results=results,
            target=target,
        )

        all_results.append(
            {
                "name": test_case["name"],
                "target": test_case["target_title"],
                "found": target is not None,
                "rank": (
                    target["rank"]
                    if target is not None
                    else None
                ),
                "score": (
                    target["score"]
                    if target is not None
                    else None
                ),
            }
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "\nHop    Target                         Found   Rank   Score"
    )

    print(
        "-" * 80
    )

    for item in all_results:

        rank = (
            str(item["rank"])
            if item["rank"] is not None
            else "-"
        )

        score = (
            f"{item['score']:.6f}"
            if item["score"] is not None
            else "-"
        )

        print(
            f"{item['name']:<7}"
            f"{item['target']:<31}"
            f"{str(item['found']):<8}"
            f"{rank:<7}"
            f"{score}"
        )


    # ========================================================
    # RECALL
    # ========================================================

    valid_cases = [
        x
        for x in all_results
        if x["target"] is not None
    ]

    if valid_cases:

        found = sum(
            x["found"]
            for x in valid_cases
        )

        total = len(valid_cases)

        print("\n" + "-" * 80)

        print(
            f"Target found @Top-{TOP_K}: "
            f"{found}/{total}"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{found / total:.4f}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
