import os
import json
import shutil
import argparse
from typing import cast

import faiss
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
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

DEFAULT_FAISS_TYPE = "Flat"


# ============================================================
# MODEL
# ============================================================

def load_model(
    model_path: str,
    use_fp16: bool = False,
):
    """
    Load E5 model and tokenizer.

    This follows the R-Search implementation:

        model.eval()
        model.cuda()

    FP16 is optional, but disabled by default for
    closer numerical reproduction of R-Search.
    """

    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    print(f"Model: {model_path}")
    print(f"FP16:  {use_fp16}")

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

    print()
    print("Model loaded successfully.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

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
    """
    Same pooling implementation used by R-Search.
    """

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
            f"Pooling method not implemented: {pooling_method}"
        )


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus(corpus_path: str):
    """
    Load the already-built MuSiQue corpus.

    Expected format:

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

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            # ------------------------------------------------
            # Prefer pre-built R-Search document
            # ------------------------------------------------

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
    batch_size=DEFAULT_BATCH_SIZE,
    max_length=DEFAULT_MAX_LENGTH,
):
    """
    Encode the entire corpus.

    This intentionally follows the R-Search dense indexing
    implementation:

        all_embeddings = []

        for batch:
            add "passage: "
            tokenize
            encode
            mean pooling
            L2 normalization
            GPU -> CPU -> NumPy
            append

        np.concatenate(all_embeddings, axis=0)
        astype(np.float32)
    """

    print("=" * 80)
    print("ENCODING ALL PASSAGES")
    print("=" * 80)

    print(f"Batch size:  {batch_size}")
    print(f"Max length:  {max_length}")
    print(f"Prefix:      passage: ")
    print()

    all_embeddings = []

    num_batches = (
        len(corpus) + batch_size - 1
    ) // batch_size

    print(f"Total batches: {num_batches:,}")
    print()

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

        # ----------------------------------------------------
        # Get documents
        # ----------------------------------------------------

        batch_data = [
            item["document"]
            for item in corpus[
                start_idx:end_idx
            ]
        ]

        # ----------------------------------------------------
        # EXACT E5 PASSAGE PREFIX USED BY R-SEARCH
        # ----------------------------------------------------

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
    Save embeddings as float32 NumPy memmap.

    This happens AFTER np.concatenate(), matching
    the R-Search implementation.
    """

    print("=" * 80)
    print("SAVING EMBEDDINGS")
    print("=" * 80)

    print(
        f"Path: {embedding_path}"
    )

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

        memmap[
            start_idx:end_idx
        ] = all_embeddings[
            start_idx:end_idx
        ]

    memmap.flush()

    del memmap

    size_mb = (
        os.path.getsize(embedding_path)
        / (1024 ** 2)
    )

    print()
    print(
        f"Embedding shape: {all_embeddings.shape}"
    )

    print(
        f"Embedding dtype: {all_embeddings.dtype}"
    )

    print(
        f"Embedding size:  {size_mb:.2f} MB"
    )

    print()


# ============================================================
# BUILD FAISS
# ============================================================

def build_faiss_index(
    all_embeddings,
    index_path,
    faiss_type="Flat",
):
    """
    Build the default dense FAISS index used by R-Search:

        faiss.index_factory(
            dim,
            "Flat",
            faiss.METRIC_INNER_PRODUCT
        )

    This produces IndexFlatIP.
    """

    print("=" * 80)
    print("BUILDING FAISS INDEX")
    print("=" * 80)

    dimension = all_embeddings.shape[-1]

    print(
        f"Number of vectors: {len(all_embeddings):,}"
    )

    print(
        f"Dimension:         {dimension}"
    )

    print(
        f"FAISS type:        {faiss_type}"
    )

    print(
        "Metric:             Inner Product"
    )

    print()

    # --------------------------------------------------------
    # Same as R-Search
    # --------------------------------------------------------

    faiss_index = faiss.index_factory(
        dimension,
        faiss_type,
        faiss.METRIC_INNER_PRODUCT,
    )

    # --------------------------------------------------------
    # Flat does not require training.
    #
    # Keep the generic R-Search logic anyway so this remains
    # compatible with other FAISS factory types.
    # --------------------------------------------------------

    if not faiss_index.is_trained:

        print("Training FAISS index...")

        faiss_index.train(
            all_embeddings
        )

    print(
        "Adding embeddings to FAISS..."
    )

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
    # Save
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
    print(
        f"Index file: {index_path}"
    )

    print(
        f"Index size: {size_mb:.2f} MB"
    )

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
    """
    Verify corpus / embedding / FAISS consistency.
    """

    print("=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

    # --------------------------------------------------------
    # SIZE CHECK
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

    print(
        "✓ Sizes match"
    )

    # --------------------------------------------------------
    # DTYPE
    # --------------------------------------------------------

    print()
    print(
        f"Embedding dtype: {embeddings.dtype}"
    )

    if embeddings.dtype != np.float32:

        raise RuntimeError(
            "Embeddings are not float32."
        )

    print(
        "✓ Float32 verified"
    )

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
    # SELF RETRIEVAL
    # --------------------------------------------------------

    print()
    print(
        "Testing FAISS retrieval using passage #0..."
    )

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

    # --------------------------------------------------------
    # Passage #0 should retrieve itself at rank 1.
    # --------------------------------------------------------

    if indices[0][0] != 0:

        print()
        print(
            "WARNING: passage #0 was not the "
            "top-1 self retrieval result."
        )

    else:

        print()
        print(
            "✓ Self retrieval verified"
        )

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
    faiss_type,
):
    """
    Save reproducibility metadata.
    """

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
            "factory": faiss_type,
            "metric": "inner_product",
            "normalized_embeddings": True,
            "gpu_build": False,
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
    max_length,
    batch_size,
):
    """
    Save artifact documentation.
    """

    readme = (
        "# MuSiQue-ART-E5-Retrieval-Index\n\n"

        "Frozen retrieval artifacts for the Adaptive Retrieval "
        "Timing (ART) project and retrieval baselines on MuSiQue v1.0.\n\n"

        "## Corpus\n\n"

        f"Number of passages: `{corpus_size:,}`\n\n"

        "The corpus follows the MuSiQue corpus construction used by "
        "R-Search.\n\n"

        "Each paragraph becomes one retrieval document.\n\n"

        "Duplicate passages are removed using:\n\n"

        "```text\n"
        "title + \" \" + paragraph_text\n"
        "```\n\n"

        "The document representation is:\n\n"

        "```text\n"
        "\"<title>\"\n"
        "<paragraph_text>\n"
        "```\n\n"

        "The E5 passage prefix is NOT stored in the corpus. "
        "It is added during embedding.\n\n"

        "## Embedding\n\n"

        f"Model: `{model_name}`\n\n"

        f"Batch size: `{batch_size}`\n\n"

        f"Maximum sequence length: `{max_length}`\n\n"

        "E5 input format:\n\n"

        "```text\n"
        "passage: \"<title>\"\n"
        "<paragraph_text>\n"
        "```\n\n"

        "Configuration:\n\n"

        "- Pooling: mean pooling\n"
        "- Normalization: L2\n"
        "- Data type: float32\n"
        "- Passage prefix: `passage: `\n"
        "- Query prefix: `query: `\n\n"

        "The tokenizer uses truncation at 512 tokens, matching "
        "the R-Search implementation.\n\n"

        "## FAISS\n\n"

        "Index type: `IndexFlatIP`\n\n"

        "Factory: `Flat`\n\n"

        "Metric: Inner Product\n\n"

        "Embeddings are L2-normalized, so inner product corresponds "
        "to cosine similarity.\n\n"

        "FAISS is built on CPU, matching the default R-Search "
        "configuration when `--faiss_gpu` is not enabled.\n\n"

        "## Query Encoding\n\n"

        "Queries must use the E5 query prefix:\n\n"

        "```text\n"
        "query: <question>\n"
        "```\n\n"

        "Then apply the same mean pooling and L2 normalization "
        "before searching the FAISS index.\n\n"

        "For example:\n\n"

        "```python\n"
        "query_text = \"query: Who was the director of ...?\"\n\n"
        "inputs = tokenizer(\n"
        "    [query_text],\n"
        "    padding=True,\n"
        "    truncation=True,\n"
        "    return_tensors=\"pt\",\n"
        "    max_length=512,\n"
        ")\n\n"
        "with torch.no_grad():\n"
        "    output = model(**inputs, return_dict=True)\n\n"
        "# mean pooling + L2 normalization\n"
        "# then:\n"
        "scores, indices = index.search(query_embedding, k=3)\n"
        "```\n\n"

        "## Corpus / Index Alignment\n\n"

        "FAISS vector position corresponds directly to the same "
        "document position in `musique_corpus.jsonl`.\n\n"

        "Therefore, if FAISS returns index `12345`, the corresponding "
        "document is line/document position `12345` in the corpus.\n\n"

        "## Files\n\n"

        "- `musique_corpus.jsonl` — R-Search-compatible MuSiQue corpus\n"
        "- `emb_e5.memmap` — float32 E5 embeddings\n"
        "- `e5_Flat.index` — FAISS IndexFlatIP\n"
        "- `metadata.json` — reproducibility metadata\n"
        "- `README.md` — artifact documentation\n\n"

        "## Compatibility\n\n"

        "This artifact is intended to be shared across ART and "
        "retrieval baselines so that all systems use the same:\n\n"

        "- corpus\n"
        "- document representation\n"
        "- E5 encoder\n"
        "- embedding preprocessing\n"
        "- FAISS index\n"
        "- retrieval metric\n\n"

        "The retrieval backend is therefore frozen for downstream "
        "experiments.\n"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(readme)

    print(
        f"README saved: {path}"
    )


# ============================================================
# UPLOAD TO HUGGING FACE
# ============================================================

def upload_to_huggingface(
    output_dir,
    repo_id,
):
    """
    Upload the complete retrieval artifact directory
    to a Hugging Face Dataset repository.

    Authentication should be done beforehand with:

        hf auth login

    or through an existing HF_TOKEN environment variable.
    """

    print("=" * 80)
    print("UPLOADING TO HUGGING FACE")
    print("=" * 80)

    print(
        f"Repository: {repo_id}"
    )

    api = HfApi()

    # --------------------------------------------------------
    # Create repository if it does not exist.
    # --------------------------------------------------------

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    print(
        "\nUploading files..."
    )

    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=(
            "Add MuSiQue E5 retrieval index and embeddings"
        ),
    )

    print()
    print("=" * 80)
    print("UPLOAD COMPLETE")
    print("=" * 80)

    print(
        f"https://huggingface.co/datasets/{repo_id}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build R-Search-compatible MuSiQue "
            "E5 + FAISS retrieval artifacts."
        )
    )

    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help=(
            "Path to musique_corpus.jsonl"
        ),
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./MuSiQue-ART-E5-Retrieval-Index",
        help=(
            "Local output directory."
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL,
    )

    # --------------------------------------------------------
    # R-Search parameters
    # --------------------------------------------------------

    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Optional FP16
    #
    # Disabled by default because the goal is to stay close
    # to the default R-Search FP32 behavior.
    # --------------------------------------------------------

    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=False,
        help=(
            "Use FP16 E5 inference. "
            "Not recommended for exact R-Search-style "
            "numerical reproduction."
        ),
    )

    # --------------------------------------------------------
    # Hugging Face
    # --------------------------------------------------------

    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        default=False,
        help=(
            "Upload the complete output directory "
            "to the Hugging Face Dataset repository."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # START
    # ========================================================

    print()
    print("=" * 80)
    print("MuSiQue E5 + FAISS INDEX BUILDER")
    print("=" * 80)
    print()

    print(
        f"Model:       {args.model_path}"
    )

    print(
        f"Corpus:      {args.corpus_path}"
    )

    print(
        f"Output:      {args.output_dir}"
    )

    print(
        f"Batch size:  {args.batch_size}"
    )

    print(
        f"Max length:  {args.max_length}"
    )

    print(
        f"FP16:        {args.use_fp16}"
    )

    print(
        f"FAISS:       IndexFlatIP / CPU"
    )

    print(
        f"Upload:      {args.upload}"
    )

    print()

    # ========================================================
    # CHECK CUDA
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "E5 indexing is configured to run on GPU."
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print()

    # ========================================================
    # PREPARE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    embedding_path = os.path.join(
        args.output_dir,
        "emb_e5.memmap",
    )

    index_path = os.path.join(
        args.output_dir,
        "e5_Flat.index",
    )

    corpus_output_path = os.path.join(
        args.output_dir,
        "musique_corpus.jsonl",
    )

    metadata_path = os.path.join(
        args.output_dir,
        "metadata.json",
    )

    readme_path = os.path.join(
        args.output_dir,
        "README.md",
    )

    # ========================================================
    # LOAD CORPUS
    # ========================================================

    corpus = load_corpus(
        args.corpus_path
    )

    if len(corpus) == 0:

        raise RuntimeError(
            "Corpus is empty."
        )

    # ========================================================
    # VERIFY IDS / ORDER
    # ========================================================

    print("=" * 80)
    print("VERIFYING CORPUS")
    print("=" * 80)

    for expected_id, item in enumerate(
        corpus
    ):

        if item["id"] != expected_id:

            raise RuntimeError(
                "Corpus IDs are not sequential. "
                f"Expected {expected_id}, "
                f"got {item['id']}."
            )

    print(
        "✓ IDs are sequential."
    )

    print(
        "✓ Corpus order is stable."
    )

    print()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    encoder, tokenizer = load_model(
        model_path=args.model_path,
        use_fp16=args.use_fp16,
    )

    # ========================================================
    # ENCODE
    # ========================================================

    all_embeddings = encode_all(
        corpus=corpus,
        encoder=encoder,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # ========================================================
    # VERIFY EMBEDDING SHAPE
    # ========================================================

    if all_embeddings.shape[0] != len(corpus):

        raise RuntimeError(
            "Embedding count does not match corpus size."
        )

    if all_embeddings.shape[1] != 768:

        raise RuntimeError(
            "Unexpected E5 embedding dimension: "
            f"{all_embeddings.shape[1]}"
        )

    # ========================================================
    # SAVE EMBEDDINGS
    # ========================================================

    save_embeddings(
        all_embeddings=all_embeddings,
        embedding_path=embedding_path,
    )

    # ========================================================
    # BUILD FAISS
    # ========================================================

    faiss_index = build_faiss_index(
        all_embeddings=all_embeddings,
        index_path=index_path,
        faiss_type=DEFAULT_FAISS_TYPE,
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    sanity_check(
        corpus=corpus,
        embeddings=all_embeddings,
        index=faiss_index,
    )

    # ========================================================
    # COPY CORPUS
    # ========================================================

    print("=" * 80)
    print("COPYING CORPUS")
    print("=" * 80)

    shutil.copy2(
        args.corpus_path,
        corpus_output_path,
    )

    print(
        f"Corpus copied to: {corpus_output_path}"
    )

    print()

    # ========================================================
    # SAVE METADATA
    # ========================================================

    save_metadata(
        path=metadata_path,
        corpus=corpus,
        model_name=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        faiss_type=DEFAULT_FAISS_TYPE,
    )

    # ========================================================
    # SAVE README
    # ========================================================

    save_readme(
        path=readme_path,
        corpus_size=len(corpus),
        model_name=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ========================================================
    # FREE LARGE OBJECTS
    # ========================================================

    del all_embeddings
    del faiss_index
    del corpus
    del encoder
    del tokenizer

    torch.cuda.empty_cache()

    # ========================================================
    # LOCAL SUMMARY
    # ========================================================

    print("=" * 80)
    print("LOCAL ARTIFACT COMPLETE")
    print("=" * 80)

    for filename in sorted(
        os.listdir(args.output_dir)
    ):

        filepath = os.path.join(
            args.output_dir,
            filename,
        )

        if os.path.isfile(filepath):

            size_mb = (
                os.path.getsize(filepath)
                / (1024 ** 2)
            )

            print(
                f"{filename:<30} "
                f"{size_mb:>10.2f} MB"
            )

    print()

    # ========================================================
    # UPLOAD
    # ========================================================

    if args.upload:

        upload_to_huggingface(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
        )

    else:

        print("=" * 80)
        print("UPLOAD SKIPPED")
        print("=" * 80)

        print(
            "Use --upload to upload the artifact "
            "to Hugging Face."
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()