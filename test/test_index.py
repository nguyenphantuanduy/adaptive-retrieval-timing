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
# COT SENTENCES
#
# Each sentence is tested independently.
#
#   S1
#   S2
#   S3
#   S4
#
# NO QUESTION
# NO CUMULATION
# ============================================================

SENTENCES = [
    (
        "S1",
        "The question asks about the founder of the Bethel branch of a "
        "religion founded by the black community in the city that used "
        "to be the U.S. capitol."
    ),

    (
        "S2",
        "To answer this, we first need to identify the city that used "
        "to be the U.S. capitol."
    ),

    (
        "S3",
        "The United States Capitol document states that before Washington, "
        "D.C. became the capital, the U.S. government met in Philadelphia."
    ),

    (
        "S4",
        "Thus, Philadelphia is the city in question."
    ),
]


# ============================================================
# TARGET DOCUMENT
# ============================================================

TARGET_TITLE = "United States Capitol"


# ============================================================
# E5 MEAN POOLING
# Same as R-Search / E5 model card
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
# LOAD FAISS + CORPUS FROM HF
# ============================================================

def load_hf_artifact():

    print("=" * 80)
    print("LOADING RETRIEVAL ARTIFACT FROM HUGGING FACE")
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

    print(f"Index vectors : {index.ntotal}")
    print(f"Index dimension: {index.d}")


    # --------------------------------------------------------
    # Corpus
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
    # Alignment check
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
# ENCODE QUERY
# ============================================================

@torch.no_grad()
def encode_query(
    query,
    tokenizer,
    model,
):

    # --------------------------------------------------------
    # E5 retrieval convention
    #
    # query: <text>
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # L2 normalization
    # --------------------------------------------------------

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
):

    query_embedding = encode_query(
        query,
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

    for result in results:

        if result["title"] == target_title:
            return result

    return None


# ============================================================
# PRINT TEST
# ============================================================

def print_test(
    name,
    sentence,
    results,
):

    print("\n")
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("\nQUERY:")
    print(sentence)

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
    # Store results
    # --------------------------------------------------------

    all_results = []


    # ========================================================
    # TEST S1..S4
    #
    # Each sentence is tested independently.
    #
    # IMPORTANT:
    # NO QUESTION
    # NO CUMULATION
    # ========================================================

    for sentence_name, sentence in SENTENCES:

        results = search(
            query=sentence,
            index=index,
            corpus=corpus,
            tokenizer=tokenizer,
            model=model,
        )

        print_test(
            name=f"TEST — {sentence_name} ONLY",
            sentence=sentence,
            results=results,
        )

        target = find_target(
            results,
            TARGET_TITLE,
        )

        all_results.append(
            {
                "name": sentence_name,
                "sentence": sentence,
                "results": results,
                "target": target,
            }
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("HOP-1 TARGET SUMMARY — SENTENCE ONLY")
    print("=" * 80)

    print(
        f"\nTarget document: {TARGET_TITLE}"
    )

    print(
        f"Top-K: {TOP_K}"
    )


    for item in all_results:

        print("\n" + "-" * 80)

        print(
            f"Sentence state: {item['name']}"
        )

        target = item["target"]

        if target is None:

            print(
                "TARGET FOUND: NO"
            )

        else:

            print(
                "TARGET FOUND: YES"
            )

            print(
                f"Rank : {target['rank']}"
            )

            print(
                f"Score: {target['score']:.6f}"
            )

            print(
                f"ID   : {target['id']}"
            )


    # ========================================================
    # RANK TRANSITION
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TARGET RANK TRANSITION — SENTENCE ONLY")
    print("=" * 80)

    print(
        "\nState                     Rank    Score"
    )

    print(
        "-" * 80
    )

    for item in all_results:

        target = item["target"]

        if target is None:

            print(
                f"{item['name']:<25} NOT FOUND"
            )

        else:

            print(
                f"{item['name']:<25} "
                f"{target['rank']:<7} "
                f"{target['score']:.6f}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
