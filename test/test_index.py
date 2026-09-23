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
# SAMPLE
# ============================================================

QUESTION = (
    "Who started the Bethel branch of the religion founded by the black "
    "community in the city that used to be the US capitol?"
)

# This is the sentence that leads to the next hop.
NEXT_HOP_SENTENCE = (
    "The United States Capitol document states that before Washington, D.C. "
    "became the capital, the U.S. government met in Philadelphia."
)


# ============================================================
# E5 MEAN POOLING
# Same as R-Search
# ============================================================

def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling with padding tokens excluded.

    Same logic as R-Search:
        hidden.sum(dim=1) / attention_mask.sum(dim=1)
    """

    mask = attention_mask[..., None].bool()

    masked_hidden = last_hidden_state.masked_fill(
        ~mask,
        0.0,
    )

    return masked_hidden.sum(dim=1) / attention_mask.sum(
        dim=1
    )[..., None]


# ============================================================
# LOAD FAISS INDEX + CORPUS FROM HUGGING FACE
# ============================================================

def load_hf_artifact():

    print("=" * 80)
    print("LOADING RETRIEVAL ARTIFACT FROM HUGGING FACE")
    print("=" * 80)

    print(f"\nRepo: {REPO_ID}")

    # --------------------------------------------------------
    # FAISS index
    # --------------------------------------------------------

    print("\nDownloading / loading FAISS index...")

    index_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="e5_Flat.index",
        repo_type=REPO_TYPE,
    )

    print(f"Index path: {index_path}")

    index = faiss.read_index(index_path)

    print(f"Index vectors : {index.ntotal}")
    print(f"Index dimension: {index.d}")


    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    print("\nDownloading / loading corpus...")

    corpus_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="musique_corpus.jsonl",
        repo_type=REPO_TYPE,
    )

    print(f"Corpus path: {corpus_path}")

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
    # Verify alignment
    # --------------------------------------------------------

    if len(corpus) != index.ntotal:

        raise RuntimeError(
            "\nCorpus / FAISS index mismatch!\n"
            f"Corpus passages : {len(corpus)}\n"
            f"FAISS vectors   : {index.ntotal}"
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

    print(f"\nModel : {MODEL_NAME}")
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
# ENCODE QUERY
# ============================================================

@torch.no_grad()
def encode_query(
    query,
    tokenizer,
    model,
):
    """
    Encode one query using the same E5 procedure
    used for retrieval-index construction.

    Query format:
        query: <query>
    """

    text = f"query: {query}"

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

    # Same as R-Search:
    # F.normalize(embeddings, p=2, dim=1)
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
    query,
    index,
    corpus,
    tokenizer,
    model,
    top_k=TOP_K,
):

    query_embedding = encode_query(
        query=query,
        tokenizer=tokenizer,
        model=model,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
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
# PRINT RESULTS
# ============================================================

def print_results(
    test_name,
    query,
    results,
):

    print("\n")
    print("=" * 80)
    print(test_name)
    print("=" * 80)

    print("\nQUERY:")
    print(query)

    print("\nTOP RESULTS:")

    for result in results:

        print("\n" + "-" * 80)

        print(f"Rank  : {result['rank']}")
        print(f"Score : {result['score']:.6f}")
        print(f"Index : {result['index']}")
        print(f"ID    : {result['id']}")
        print(f"Title : {result['title']}")

        print("\nContent:")
        print(result["content"])


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    results_question,
    results_question_plus_sentence,
):

    print("\n")
    print("=" * 80)
    print("RETRIEVAL COMPARISON")
    print("=" * 80)

    print("\nQuestion only:")

    for result in results_question:

        print(
            f"  Rank {result['rank']} | "
            f"ID={result['id']} | "
            f"Score={result['score']:.6f} | "
            f"Title={result['title']}"
        )


    print("\nQuestion + 1 sentence:")

    for result in results_question_plus_sentence:

        print(
            f"  Rank {result['rank']} | "
            f"ID={result['id']} | "
            f"Score={result['score']:.6f} | "
            f"Title={result['title']}"
        )


    # --------------------------------------------------------
    # Compare top-k IDs
    # --------------------------------------------------------

    ids_question = [
        result["id"]
        for result in results_question
    ]

    ids_question_plus_sentence = [
        result["id"]
        for result in results_question_plus_sentence
    ]

    shared_ids = (
        set(ids_question)
        & set(ids_question_plus_sentence)
    )

    print("\nTop-k ID comparison:")

    print(
        f"  Question only          : "
        f"{ids_question}"
    )

    print(
        f"  Question + 1 sentence  : "
        f"{ids_question_plus_sentence}"
    )

    print(
        f"  Shared documents       : "
        f"{sorted(shared_ids)}"
    )

    print(
        f"  Number shared          : "
        f"{len(shared_ids)}/{TOP_K}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load shared retrieval artifact
    # --------------------------------------------------------

    index, corpus = load_hf_artifact()


    # --------------------------------------------------------
    # Load E5
    # --------------------------------------------------------

    tokenizer, model = load_e5()


    # ========================================================
    # TEST 1
    # Question only
    # ========================================================

    results_question = search(
        query=QUESTION,
        index=index,
        corpus=corpus,
        tokenizer=tokenizer,
        model=model,
        top_k=TOP_K,
    )

    print_results(
        test_name="TEST 1 — QUESTION ONLY",
        query=QUESTION,
        results=results_question,
    )


    # ========================================================
    # TEST 2
    # Question + 1 sentence leading to next hop
    # ========================================================

    question_plus_sentence = (
        QUESTION
        + " "
        + NEXT_HOP_SENTENCE
    )

    results_question_plus_sentence = search(
        query=question_plus_sentence,
        index=index,
        corpus=corpus,
        tokenizer=tokenizer,
        model=model,
        top_k=TOP_K,
    )

    print_results(
        test_name="TEST 2 — QUESTION + 1 NEXT-HOP SENTENCE",
        query=question_plus_sentence,
        results=results_question_plus_sentence,
    )


    # ========================================================
    # COMPARISON
    # ========================================================

    print_comparison(
        results_question=results_question,
        results_question_plus_sentence=(
            results_question_plus_sentence
        ),
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
