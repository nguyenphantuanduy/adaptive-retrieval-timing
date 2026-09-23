```python
import json
import torch
import faiss
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

COT_SENTENCE = (
    "The question asks about the founder of the Bethel branch of a religion "
    "founded by the black community in the city that used to be the U.S. capitol."
)

QUESTION_ONLY = QUESTION

QUESTION_PLUS_ONE_SENTENCE = (
    QUESTION + " " + COT_SENTENCE
)


# ============================================================
# E5 POOLING
# Same as R-Search
# ============================================================

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask[..., None].bool()

    masked_hidden = last_hidden_state.masked_fill(~mask, 0.0)

    return masked_hidden.sum(dim=1) / attention_mask.sum(
        dim=1
    )[..., None]


# ============================================================
# LOAD HF ARTIFACT
# ============================================================

def load_hf_artifact():
    print("=" * 80)
    print("DOWNLOADING / LOADING RETRIEVAL ARTIFACT FROM HUGGING FACE")
    print("=" * 80)

    print(f"\nRepo: {REPO_ID}")

    # --------------------------------------------------------
    # FAISS index
    # --------------------------------------------------------

    index_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="e5_Flat.index",
        repo_type=REPO_TYPE,
    )

    print(f"\nFAISS index:")
    print(f"  {index_path}")

    index = faiss.read_index(index_path)

    print(f"  ntotal : {index.ntotal}")
    print(f"  dim    : {index.d}")

    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    corpus_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="musique_corpus.jsonl",
        repo_type=REPO_TYPE,
    )

    print(f"\nCorpus:")
    print(f"  {corpus_path}")

    corpus = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    print(f"  passages: {len(corpus)}")

    # --------------------------------------------------------
    # Verify alignment
    # --------------------------------------------------------

    if len(corpus) != index.ntotal:
        raise RuntimeError(
            f"Corpus/index mismatch!\n"
            f"  corpus = {len(corpus)}\n"
            f"  index  = {index.ntotal}"
        )

    print("\nCorpus/index alignment: OK")

    return index, corpus


# ============================================================
# LOAD E5
# ============================================================

def load_e5():
    print("\n" + "=" * 80)
    print("LOADING E5")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")
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
def encode_query(query, tokenizer, model):
    # E5 query prefix
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
    # F.normalize(..., p=2, dim=1)
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
        query,
        tokenizer,
        model,
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
        doc = corpus[int(idx)]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "index": int(idx),
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
            }
        )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(query_name, query, results):
    print("\n")
    print("=" * 80)
    print(query_name)
    print("=" * 80)

    print("\nQUERY:")
    print(query)

    print("\nRETRIEVED PASSAGES:")

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

    # --------------------------------------------------------
    # Test 1:
    # Question only
    # --------------------------------------------------------

    results_question_only = search(
        query=QUESTION_ONLY,
        index=index,
        corpus=corpus,
        tokenizer=tokenizer,
        model=model,
        top_k=TOP_K,
    )

    print_results(
        "TEST 1 — QUESTION ONLY",
        QUESTION_ONLY,
        results_question_only,
    )

    # --------------------------------------------------------
    # Test 2:
    # Question + 1 CoT sentence
    # --------------------------------------------------------

    results_question_plus_sentence = search(
        query=QUESTION_PLUS_ONE_SENTENCE,
        index=index,
        corpus=corpus,
        tokenizer=tokenizer,
        model=model,
        top_k=TOP_K,
    )

    print_results(
        "TEST 2 — QUESTION + 1 COT SENTENCE",
        QUESTION_PLUS_ONE_SENTENCE,
        results_question_plus_sentence,
    )

    # --------------------------------------------------------
    # Compare rankings
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RANKING COMPARISON")
    print("=" * 80)

    print("\nQuestion only:")

    for r in results_question_only:
        print(
            f"  Rank {r['rank']}: "
            f"ID={r['id']} | "
            f"score={r['score']:.6f} | "
            f"title={r['title']}"
        )

    print("\nQuestion + 1 sentence:")

    for r in results_question_plus_sentence:
        print(
            f"  Rank {r['rank']}: "
            f"ID={r['id']} | "
            f"score={r['score']:.6f} | "
            f"title={r['title']}"
        )

    # --------------------------------------------------------
    # Shared documents
    # --------------------------------------------------------

    ids_question_only = {
        r["id"] for r in results_question_only
    }

    ids_question_plus_sentence = {
        r["id"] for r in results_question_plus_sentence
    }

    shared = ids_question_only & ids_question_plus_sentence

    print("\nShared top-k document IDs:")
    print(f"  {sorted(shared)}")


if __name__ == "__main__":
    main()
```
