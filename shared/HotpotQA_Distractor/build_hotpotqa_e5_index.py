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
    "adaptive-retrieval-timing/HotpotQA-ART-E5-Retrieval-Index"
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

    Follows the R-Search loading sequence.
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
    Load the already-built HotpotQA corpus.

    Expected input JSONL format:

    {
        "id": "...",
        "title": "...",
        "content": "...",
        "document": "\\"<title>\\"\\n<text>"
    }

    The input record order is preserved.

    Important:
    - Original HotpotQA IDs are preserved.
    - IDs do NOT have to be sequential.
    - FAISS positions correspond to corpus record positions.
    """

    print("=" * 80)
    print("LOADING CORPUS")
    print("=" * 80)

    print(f"Path: {corpus_path}")
    print()

    corpus = []

    with open(
        corpus_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            tqdm(f, desc="Loading corpus"),
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if "id" not in item:
                raise ValueError(
                    f"Missing 'id' at line {line_number}."
                )

            if "title" not in item:
                raise ValueError(
                    f"Missing 'title' at line {line_number}."
                )

            title = item["title"]

            content = item.get(
                "content",
                item.get("text", ""),
            )

            # Prefer the pre-built R-Search document.
            if "document" in item:

                document = item["document"]

            else:

                document = (
                    f'"{title}"\n'
                    f"{content}"
                )

            corpus.append(
                {
                    "id": item["id"],
                    "title": title,
                    "content": content,
                    "document": document,
                }
            )

    if not corpus:
        raise RuntimeError(
            f"Corpus is empty: {corpus_path}"
        )

    print()
    print(f"Corpus size: {len(corpus):,}")

    print()
    print("First record:")
    print(
        json.dumps(
            corpus[0],
            ensure_ascii=False,
            indent=2,
        )[:1500]
    )

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
    Encode all documents following R-Search's E5 pipeline.

    Steps:
        1. Add "passage: " prefix.
        2. Tokenize with truncation.
        3. Encode.
        4. Mean pooling.
        5. L2 normalization.
        6. Move embeddings to CPU.
        7. Concatenate and cast to float32.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    print("=" * 80)
    print("ENCODING ALL PASSAGES")
    print("=" * 80)

    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print("Prefix:     passage: ")
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

        batch = corpus[start_idx:end_idx]

        # Get document representations.
        batch_data = [
            item["document"]
            for item in batch
        ]

        # E5 passage prefix used by R-Search.
        batch_data = [
            f"passage: {doc}"
            for doc in batch_data
        ]

        # Tokenization.
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

        # Encoder.
        output = encoder(
            **inputs,
            return_dict=True,
        )

        # Mean pooling.
        embeddings = pooling(
            output.pooler_output,
            output.last_hidden_state,
            inputs["attention_mask"],
            "mean",
        )

        # L2 normalization.
        embeddings = torch.nn.functional.normalize(
            embeddings,
            dim=-1,
        )

        # GPU -> CPU -> NumPy.
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
    Save float32 embeddings as a NumPy memmap.
    """

    print("=" * 80)
    print("SAVING EMBEDDINGS")
    print("=" * 80)

    print(f"Path: {embedding_path}")

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
    faiss_type=DEFAULT_FAISS_TYPE,
):
    """
    Build a FAISS index using inner product.

    Default:
        faiss.index_factory(
            dimension,
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
        "Metric:            Inner Product"
    )

    print()

    faiss_index = faiss.index_factory(
        dimension,
        faiss_type,
        faiss.METRIC_INNER_PRODUCT,
    )

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

    if faiss_index.ntotal != len(all_embeddings):

        raise RuntimeError(
            "FAISS index size does not match "
            "embedding count: "
            f"{faiss_index.ntotal} vs "
            f"{len(all_embeddings)}"
        )

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
    Verify corpus/embedding/index consistency.

    FAISS position is NOT assumed to equal the source ID.
    """

    print("=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

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

    # Dtype check.
    print()
    print(
        f"Embedding dtype: {embeddings.dtype}"
    )

    if embeddings.dtype != np.float32:

        raise RuntimeError(
            "Embeddings are not float32."
        )

    print("✓ Float32 verified")

    # Normalization check.
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

    # Self-retrieval check.
    print()
    print(
        "Testing FAISS retrieval using corpus position 0..."
    )

    query_embedding = np.asarray(
        embeddings[0:1],
        dtype=np.float32,
    )

    scores, indices = index.search(
        query_embedding,
        min(3, len(corpus)),
    )

    print()

    for rank, (idx, score) in enumerate(
        zip(
            indices[0],
            scores[0],
        ),
        start=1,
    ):

        if idx < 0:
            continue

        item = corpus[idx]

        print(
            f"{rank}. "
            f"FAISS position={idx} | "
            f"score={score:.6f} | "
            f"source_id={item['id']} | "
            f"title={item['title']}"
        )

    if indices[0][0] != 0:

        print()
        print(
            "WARNING: corpus position 0 was not "
            "the top-1 self-retrieval result."
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
            "HotpotQA-ART-E5-Retrieval-Index"
        ),

        "dataset": "HotpotQA",

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
            "index_type": (
                "IndexFlatIP"
                if faiss_type == "Flat"
                else faiss_type
            ),
            "factory": faiss_type,
            "metric": "inner_product",
            "normalized_embeddings": True,
            "gpu_build": False,
        },

        "document_format": '"<title>"\\n<text>',

        "corpus_order": (
            "Same order as hotpotqa_corpus.jsonl"
        ),

        "batch_size": batch_size,

        "r_search_compatible": True,

        "notes": [
            (
                "FAISS vector position maps to the "
                "corresponding record position in the corpus."
            ),
            (
                "Original source IDs are preserved and "
                "do not have to be sequential."
            ),
            (
                "The E5 passage prefix is added during "
                "embedding, not stored in the corpus."
            ),
        ],

        "files": {
            "corpus": "hotpotqa_corpus.jsonl",
            "embedding": "emb_e5.memmap",
            "index": (
                "e5_Flat.index"
                if faiss_type == "Flat"
                else f"e5_{faiss_type}.index"
            ),
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
    faiss_type,
):
    """
    Save artifact documentation.
    """

    index_type = (
        "IndexFlatIP"
        if faiss_type == "Flat"
        else faiss_type
    )

    readme = f"""# HotpotQA-ART-E5-Retrieval-Index

Frozen retrieval artifacts for the Adaptive Retrieval Timing (ART) project and retrieval baselines on HotpotQA.

## Corpus

- Number of passages: `{corpus_size:,}`
- Input: pre-built HotpotQA corpus in JSONL format.
- Input record order is preserved.
- Original source IDs are preserved and do not have to be sequential.

The document representation follows R-Search:

```text
"<title>"
<text>
```

The E5 passage prefix is NOT stored in the corpus. It is added during embedding.

## Embedding

- Model: `{model_name}`
- Batch size: `{batch_size}`
- Maximum sequence length: `{max_length}`
- Pooling: mean pooling
- Normalization: L2
- Data type: float32
- Document prefix: `passage: `
- Query prefix: `query: `

The tokenizer uses truncation at `{max_length}` tokens.

## FAISS

- Index type: `{index_type}`
- Factory: `{faiss_type}`
- Metric: inner product
- Embeddings are L2-normalized, so inner product corresponds to cosine similarity.
- FAISS is built on CPU.

## Query Encoding

Queries should use the E5 query prefix:

```text
query: <question>
```

Apply the same mean pooling and L2 normalization before searching the FAISS index.

## Corpus / Index Alignment

FAISS vector position `i` corresponds to record position `i` in `hotpotqa_corpus.jsonl`.

Do not assume that a source ID equals the FAISS position. Use the returned FAISS position to retrieve the corresponding record from the corpus in its original order.

## Files

- `hotpotqa_corpus.jsonl` — copy of the input corpus
- `emb_e5.memmap` — float32 E5 embeddings
- `e5_Flat.index` — FAISS index (name varies for non-default index types)
- `metadata.json` — reproducibility metadata
- `README.md` — this file

## Compatibility

This artifact follows the R-Search-style document representation, E5 passage prefix, mean pooling, L2 normalization, and FAISS inner-product search.

Different library/model versions or inference precision may cause small numerical differences.
"""

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
    Upload the complete artifact directory to a
    Hugging Face Dataset repository.
    """

    print("=" * 80)
    print("UPLOADING TO HUGGING FACE")
    print("=" * 80)

    print(
        f"Repository: {repo_id}"
    )

    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    print()
    print("Uploading files...")

    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=(
            "Add HotpotQA E5 retrieval index and embeddings"
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
            "Build R-Search-compatible HotpotQA "
            "E5 + FAISS retrieval artifacts."
        )
    )

    # Corpus.
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help=(
            "Path to the already-built "
            "hotpotqa_corpus.jsonl."
        ),
    )

    # Output.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./HotpotQA-ART-E5-Retrieval-Index",
        help="Local output directory.",
    )

    # Model.
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL,
    )

    # R-Search parameters.
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

    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=False,
        help=(
            "Use FP16 model inference. "
            "May introduce small numerical differences."
        ),
    )

    # Hugging Face.
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
            "to a Hugging Face Dataset repository."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("HotpotQA E5 + FAISS INDEX BUILDER")
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
        "FAISS:       IndexFlatIP / CPU"
    )

    print(
        f"Upload:      {args.upload}"
    )

    print()

    # Check CUDA.
    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "E5 indexing is configured to run on GPU."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch_size must be greater than zero."
        )

    if args.max_length <= 0:
        raise ValueError(
            "--max_length must be greater than zero."
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print()

    # Prepare output directory.
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
        "hotpotqa_corpus.jsonl",
    )

    metadata_path = os.path.join(
        args.output_dir,
        "metadata.json",
    )

    readme_path = os.path.join(
        args.output_dir,
        "README.md",
    )

    # Load corpus.
    corpus = load_corpus(
        args.corpus_path
    )

    # Load model.
    encoder, tokenizer = load_model(
        model_path=args.model_path,
        use_fp16=args.use_fp16,
    )

    # Encode corpus.
    all_embeddings = encode_all(
        corpus=corpus,
        encoder=encoder,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Verify embedding shape.
    if all_embeddings.shape[0] != len(corpus):

        raise RuntimeError(
            "Embedding count does not match corpus size: "
            f"{all_embeddings.shape[0]} vs {len(corpus)}"
        )

    expected_dimension = getattr(
        encoder.config,
        "hidden_size",
        None,
    )

    if (
        expected_dimension is not None
        and all_embeddings.shape[1] != expected_dimension
    ):

        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{all_embeddings.shape[1]} vs "
            f"model hidden size {expected_dimension}"
        )

    # Save embeddings.
    save_embeddings(
        all_embeddings=all_embeddings,
        embedding_path=embedding_path,
    )

    # Build FAISS.
    faiss_index = build_faiss_index(
        all_embeddings=all_embeddings,
        index_path=index_path,
        faiss_type=DEFAULT_FAISS_TYPE,
    )

    # Sanity check.
    sanity_check(
        corpus=corpus,
        embeddings=all_embeddings,
        index=faiss_index,
    )

    # Copy corpus without changing its original line order/content.
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

    # Metadata.
    save_metadata(
        path=metadata_path,
        corpus=corpus,
        model_name=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        faiss_type=DEFAULT_FAISS_TYPE,
    )

    # README.
    save_readme(
        path=readme_path,
        corpus_size=len(corpus),
        model_name=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        faiss_type=DEFAULT_FAISS_TYPE,
    )

    # Free large objects.
    del all_embeddings
    del faiss_index
    del corpus
    del encoder
    del tokenizer

    torch.cuda.empty_cache()

    # Local summary.
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

    # Optional upload.
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