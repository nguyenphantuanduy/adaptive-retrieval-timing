
import json
from typing import List, Dict, Optional, Tuple, Union

import faiss
import torch
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel


def truncate_passages(
    passages: str,
    tokenizer=None,
    max_tokens: int = -1,
) -> str:
    """
    Truncate concatenated passages to at most max_tokens tokens.
    If max_tokens == -1, return the input unchanged without tokenization.
    """
    if not isinstance(passages, str):
        raise TypeError("passages must be a string")

    if max_tokens == -1:
        return passages

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0 or -1")

    if tokenizer is None:
        raise ValueError(
            "tokenizer must be provided when max_tokens != -1"
        )

    tokens = tokenizer.encode(
        passages,
        add_special_tokens=False,
    )

    if len(tokens) <= max_tokens:
        return passages

    return tokenizer.decode(
        tokens[:max_tokens],
        skip_special_tokens=True,
    )


def append_retrieval_observation(observation: str) -> str:
    """
    Wrap retrieved passages in the observation message format.
    """
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "\n"
        "<observation>\n\n"
        + observation
        + "\n\n"
        "</observation>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class HotpotQARetrievalDB:
    """
    HotpotQA retrieval database using E5-base-v2 and FAISS.

    All retrieval assets are downloaded from Hugging Face.

    Repository:
        adaptive-retrieval-timing/HotpotQA-ART-E5-Retrieval-Index

    Files:
        - e5_Flat.index
        - hotpotqa_corpus.jsonl

    Encoder:
        intfloat/e5-base-v2

    No manually prepared local index/corpus directory is required.
    Hugging Face Hub manages the download cache.
    """

    DEFAULT_REPO_ID = (
        "adaptive-retrieval-timing/HotpotQA-ART-E5-Retrieval-Index"
    )

    DEFAULT_REPO_TYPE = "dataset"

    DEFAULT_MODEL_NAME = "intfloat/e5-base-v2"

    DEFAULT_INDEX_FILENAME = "e5_Flat.index"

    DEFAULT_CORPUS_FILENAME = "hotpotqa_corpus.jsonl"

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        repo_type: str = DEFAULT_REPO_TYPE,
        model_name: str = DEFAULT_MODEL_NAME,
        index_filename: str = DEFAULT_INDEX_FILENAME,
        corpus_filename: str = DEFAULT_CORPUS_FILENAME,
        top_k: int = 3,
        max_length: int = 512,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if max_length <= 0:
            raise ValueError("max_length must be > 0")

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.model_name = model_name

        self.index_filename = index_filename
        self.corpus_filename = corpus_filename

        self.top_k = top_k
        self.max_length = max_length

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.revision = revision
        self.cache_dir = cache_dir

        # ----------------------------------------------------------
        # Download retrieval assets from Hugging Face
        # ----------------------------------------------------------

        print(
            f"Downloading FAISS index from Hugging Face: "
            f"{self.repo_id}/{self.index_filename}"
        )

        self.index_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.index_filename,
            repo_type=self.repo_type,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )

        print(
            f"Downloading HotpotQA corpus from Hugging Face: "
            f"{self.repo_id}/{self.corpus_filename}"
        )

        self.corpus_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.corpus_filename,
            repo_type=self.repo_type,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )

        # ----------------------------------------------------------
        # Load FAISS index and corpus
        # ----------------------------------------------------------

        self.index = self._load_index()
        self.corpus = self._load_corpus()

        # FAISS row i must correspond to corpus row i.
        if len(self.corpus) != self.index.ntotal:
            raise RuntimeError(
                "Corpus / FAISS mismatch!\n"
                f"Corpus passages: {len(self.corpus)}\n"
                f"FAISS vectors: {self.index.ntotal}"
            )

        # ----------------------------------------------------------
        # Load E5 model and tokenizer from Hugging Face
        # ----------------------------------------------------------

        print(
            f"Loading E5 tokenizer: {self.model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )

        print(
            f"Loading E5 encoder: {self.model_name}"
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        )

        self.model.eval()
        self.model.to(self.device)

        print("\nHotpotQARetrievalDB initialized successfully.")
        print(f"Repository: {self.repo_id}")
        print(f"Corpus passages: {len(self.corpus):,}")
        print(f"FAISS vectors: {self.index.ntotal:,}")
        print(f"Embedding model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Default top_k: {self.top_k}")

    # --------------------------------------------------------------
    # Loading Hugging Face assets
    # --------------------------------------------------------------

    def _load_index(self):
        print(
            f"Loading FAISS index: {self.index_path}"
        )

        return faiss.read_index(self.index_path)

    def _load_corpus(self) -> List[Dict]:
        print(
            f"Loading corpus: {self.corpus_path}"
        )

        corpus = []

        with open(
            self.corpus_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    doc = json.loads(line)

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}: {exc}"
                    ) from exc

                if not isinstance(doc, dict):
                    raise ValueError(
                        f"Expected a JSON object at line {line_number}"
                    )

                corpus.append(doc)

        return corpus

    # --------------------------------------------------------------
    # Corpus field helpers
    # --------------------------------------------------------------

    @staticmethod
    def _get_passage_text(doc: Dict) -> str:
        if "content" in doc:
            return str(doc["content"])

        if "document" in doc:
            return str(doc["document"])

        raise KeyError(
            "Corpus record must contain either 'content' or 'document'."
        )

    @classmethod
    def _format_passage(cls, doc: Dict) -> str:
        title = str(doc.get("title", ""))
        content = cls._get_passage_text(doc)

        if title:
            return f"Title: {title}\n{content}"

        return content

    # --------------------------------------------------------------
    # E5 embedding
    # --------------------------------------------------------------

    @staticmethod
    def _mean_pooling(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        mask = attention_mask[..., None].bool()

        masked_hidden = last_hidden_state.masked_fill(
            ~mask,
            0.0,
        )

        return masked_hidden.sum(dim=1) / (
            attention_mask.sum(
                dim=1,
                keepdim=True,
            ).clamp(min=1)
        )

    @torch.no_grad()
    def _encode_query(self, query: str):

        # E5 query prefix
        text = f"query: {query}"

        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs,
            return_dict=True,
        )

        embedding = self._mean_pooling(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

        return embedding.cpu().numpy().astype("float32")

    # --------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string"
            )

        k = self.top_k if top_k is None else top_k

        if k <= 0:
            raise ValueError("top_k must be > 0")

        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)

        query_embedding = self._encode_query(query)

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            idx = int(idx)

            if idx < 0:
                continue

            doc = self.corpus[idx]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "index": idx,
                    "id": doc.get("id"),
                    "title": doc.get("title", ""),
                    "document": self._get_passage_text(doc),
                }
            )

        return results

    # --------------------------------------------------------------
    # Public search API
    # --------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_passage_tokens: int = -1,
        tokenizer=None,
        return_passages: bool = False,
    ) -> Union[str, Tuple[str, List[str]]]:

        results = self._retrieve(
            query=query,
            top_k=top_k,
        )

        passage_parts = [
            self._format_passage(
                self.corpus[result["index"]]
            )
            for result in results
        ]

        passages = "\n\n".join(passage_parts)

        passages = truncate_passages(
            passages=passages,
            tokenizer=tokenizer,
            max_tokens=max_passage_tokens,
        )

        observation = append_retrieval_observation(
            passages
        )

        if return_passages:
            return observation, passage_parts

        return observation

    def search_with_metadata(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        return self._retrieve(
            query=query,
            top_k=top_k,
        )

    def __len__(self) -> int:
        return self.index.ntotal
