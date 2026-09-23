import os
import json
import shutil
import argparse
from typing import cast

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import HfApi


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "intfloat/e5-base-v2"

DEFAULT_REPO_ID = (
    "adaptive-retrieval-timing/MuSiQue-ART-E5-Retrieval-Index"
)

DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 512


# ============================================================
# MODEL
# ============================================================

def load_model(
    model_path: str,
    use_fp16: bool = False,
):
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model.eval()
    model.cuda()

    if use_fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )

    return model, tokenizer


# ============================================================
# POOLING
# ============================================================

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask=None,
    pooling_method="mean",
):
    if pooling_method == "mean":

        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )

        return (
            last_hidden.sum(dim=1)
            / attention_mask.sum(dim=1)[..., None]
        )

    elif pooling_method == "cls":

        return last_hidden_state[:, 0]

    elif pooling_method == "pooler":

        return pooler_output

    else:

        raise NotImplementedError(
            "Pooling method not implemented!"
        )


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus(corpus_path: str):
    """
    Load the already-built MuSiQue corpus.

    Expected JSONL:

    {
        "id": 0,
        "title": "...",
        "content": "...",
        "document": "\"<title>\"\\n<paragraph_text>"
    }

    The document field follows the R-Search representation.
    """

    print("=" * 80)
    print("LOADING CORPUS")
    print("=" * 80)

    corpus = []

    with open(
        corpus_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc="Loading corpus",
        ):

            item = json.loads(line)

            if "document" in item:

                document = item["document"]

            else:

                title = item["title"]

                content = item.get(
                    "content",
                    item.get("text", ""),
                )

                document = (
                    f'"{title}"\n'
                    f"{content}"
                )

            corpus.append(
                {
                    "id": item["id"],
                    "title": item["title"],
                    "content": item.get(
                        "content",
                        item.get("text", ""),
                    ),
                    "document": document,
                }
            )

    print()
    print(f"Corpus size: {len(corpus):,}")
    print()

    return corpus


# ============================================================
# ENCODE ALL
# ============================================================

@torch.no_grad()
def encode_all(
    corpus,
    encoder,
    tokenizer,
    retrieval_method="e5",
    batch_size=512,
    max_length=512,
):
    """
    Encode the entire corpus.

    Intentionally follows the R-Search implementation:

        all_embeddings = []

        for batch:
            tokenize
            encode
            mean pooling
            L2 normalize
            CPU -> NumPy
            append

        np.concatenate(all_embeddings)
        astype(np.float32)
    """

    print("=" * 80)
    print("ENCODING ALL PASSAGES")
    print("=" * 80)

    all_embeddings = []

    for start_idx in tqdm(
        range(
            0,
            len(corpus),
            batch_size,
        ),
        desc="Inference Embeddings",
    ):

        end_idx = min(
            start_idx + batch_size,
            len(corpus),
        )

        batch_data = [
            item["document"]
            for item in corpus[
                start_idx:end_idx
            ]
        ]

        # ----------------------------------------------------
        # EXACT E5 PREFIX USED BY R-SEARCH
        # ----------------------------------------------------

        if retrieval_method == "e5":

            batch_data = [
                f"passage: {doc}"
                for doc in batch_data
            ]

        # ----------------------------------------------------
        # TOKENIZATION
        # ----------------------------------------------------

        inputs = tokenizer(
            batch_data,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )

        inputs = {
            key: value.cuda()
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        output = encoder(
            **inputs,
            return_dict=True,
        )

        # ----------------------------------------------------
        # MEAN POOLING
        # ----------------------------------------------------

        embeddings = pooling(
            output.pooler_output,
            output.last_hidden_state,
            inputs["attention_mask"],
            "mean",
        )

        # ----------------------------------------------------
        # L2 NORMALIZATION
        # ----------------------------------------------------

        if "dpr" not in retrieval_method:

            embeddings = torch.nn.functional.normalize(
                embeddings,
                dim=-1,
            )

        # ----------------------------------------------------
        # GPU -> CPU -> NUMPY
        # ----------------------------------------------------

        embeddings = cast(
            torch.Tensor,
            embeddings,
        )

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .numpy()
        )

        all_embeddings.append(
            embeddings
        )

    # --------------------------------------------------------
    # SAME AS R-SEARCH
    # --------------------------------------------------------

    print()
    print("Concatenating embeddings...")

    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    all_embeddings = all_embeddings.astype(
        np.float32
    )

    print(
        f"Embedding shape: {all_embeddings.shape}"
    )

    print(
        f"Embedding dtype: {all_embeddings.dtype}"
    )

    print()

    return all_embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    all_embeddings,
    embedding_path,
):
    """
    Save the final embedding matrix as a float32 memmap.

    This happens AFTER np.concatenate(), matching R-Search.
    """

    print("=" * 80)
    print("SAVING EMBEDDINGS")
    print("=" * 80)

    memmap = np.memmap(
        embedding_path,
        shape=all_embeddings.shape,
        mode="w+",
        dtype=np.float32,
    )

    save_batch_size = 10000

    for start_idx in tqdm(
        range(
            0,
            len(all_embeddings),
            save_batch_size,
        ),
        desc="Writing embeddings",
    ):

        end_idx = min(
            start_idx + save_batch_size,
            len(all_embeddings),
        )

        memmap[start_idx:end_idx] = (
            all_embeddings[start_idx:end_idx]
        )

    memmap.flush()

    del memmap

    size_mb = (
        os.path.getsize(embedding_path)
        / (1024 ** 2)
    )

    print()
    print(f"Embedding file: {embedding_path}")
    print(f"Shape: {all_embeddings.shape}")
    print(f"Size: {size_mb:.2f} MB")
    print()


# ============================================================
# BUILD FAISS
# ============================================================

def build_faiss_index(
    all_embeddings,
    index_path,
):
    """
    Build exactly the default dense FAISS index used by
    R-Search:

        faiss.index_factory(
            dim,
            "Flat",
            faiss.METRIC_INNER_PRODUCT
        )

    This is IndexFlatIP.
    """

    print("=" * 80)
    print("BUILDING FAISS INDEX")
    print("=" * 80)

    dimension = all_embeddings.shape[-1]

    print(f"Number of vectors: {len(all_embeddings):,}")
    print(f"Dimension: {dimension}")
    print("FAISS type: Flat")
    print("Metric: Inner Product")
    print()

    faiss_index = faiss.index_factory(
        dimension,
        "Flat",
        faiss.METRIC_INNER_PRODUCT,
    )

    # Same logic as R-Search.
    if not faiss_index.is_trained:

        print("Training FAISS index...")

        faiss_index.train(
            all_embeddings
        )

    print("Adding embeddings to FAISS...")

    faiss_index.add(
        all_embeddings
    )

    print(
        f"FAISS ntotal: {faiss_index.ntotal:,}"
    )

    if faiss_index.ntotal != len(
        all_embeddings
    ):
        raise RuntimeError(
            "FAISS index size does not match "
            "embedding count."
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    faiss.write_index(
        faiss_index,
        index_path,
    )

    size_mb = (
        os.path.getsize(index_path)
        / (1024 ** 2)
    )

    print()
    print(f"Index file: {index_path}")
    print(f"Index size: {size_mb:.2f} MB")
    print()

    return faiss_index


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check(
    corpus,
    embeddings,
    index,
):
    print("=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    if len(corpus) != len(embeddings):

        raise RuntimeError(
            f"Corpus/embedding mismatch: "
            f"{len(corpus)} vs {len(embeddings)}"
        )

    if index.ntotal != len(corpus):

        raise RuntimeError(
            f"Corpus/FAISS mismatch: "
            f"{len(corpus)} vs {index.ntotal}"
        )

    print(
        f"Corpus:     {len(corpus):,}"
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        f"FAISS:      {index.ntotal:,}"
    )

    print("✓ Sizes match")

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    sample_size = min(
        1000,
        len(embeddings),
    )

    norms = np.linalg.norm(
        embeddings[:sample_size],
        axis=1,
    )

    print()
    print(
        f"Norm mean: {norms.mean():.8f}"
    )

    print(
        f"Norm min:  {norms.min():.8f}"
    )

    print(
        f"Norm max:  {norms.max():.8f}"
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):

        print(
            "WARNING: embeddings are not "
            "perfectly normalized."
        )

    else:

        print(
            "✓ L2 normalization verified"
        )

    # --------------------------------------------------------
    # SELF RETRIEVAL TEST
    # --------------------------------------------------------

    print()
    print("Testing FAISS retrieval using passage #0...")

    query_embedding = np.asarray(
        embeddings[0:1],
        dtype=np.float32,
    )

    scores, indices = index.search(
        query_embedding,
        3,
    )

    print()

    for rank, (
        idx,
        score,
    ) in enumerate(
        zip(
            indices[0],
            scores[0],
        ),
        start=1,
    ):

        print(
            f"{rank}. "
            f"FAISS id={idx} | "
            f"score={score:.6f} | "
            f"title={corpus[idx]['title']}"
        )

    print()
    print("✓ FAISS retrieval works")
    print()


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    path,
    corpus,
    model_name,
    max_length,
    batch_size,
):
    metadata = {
        "artifact": (
            "MuSiQue-ART-E5-Retrieval-Index"
        ),

        "dataset": "MuSiQue v1.0",

        "num_documents": len(corpus),

        "embedding_model": model_name,

        "embedding": {
            "dimension": 768,
            "pooling": "mean",
            "normalization": "L2",
            "dtype": "float32",
            "max_length": max_length,
            "document_prefix": "passage: ",
            "query_prefix": "query: ",
        },

        "faiss": {
            "index_type": "IndexFlatIP",
            "factory": "Flat",
            "metric": "inner_product",
            "normalized_embeddings": True,
        },

        "document_format": (
            '"<title>"\\n'
            '<paragraph_text>'
        ),

        "corpus_order": (
            "Same order as musique_corpus.jsonl"
        ),

        "batch_size": batch_size,

        "r_search_compatible": True,

        "files": {
            "corpus": "musique_corpus.jsonl",
            "embedding": "emb_e5.memmap",
            "index": "e5_Flat.index",
        },
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Metadata saved: {path}"
    )


# ============================================================
# README
# ============================================================

def save_readme(
    path,
    corpus_size,
    model_name,
):
    readme = f"""# MuSiQue-ART-E5-Retrieval-Index

Frozen retrieval artifacts for the Adaptive Retrieval Timing (ART)
project and retrieval baselines on MuSiQue v1.0.

## Corpus

Number of passages:

`{corpus_size:,}`

The corpus follows the MuSiQue corpus construction used by R-Search.

Duplicate passages are removed using:

```text
title + " " + paragraph_text