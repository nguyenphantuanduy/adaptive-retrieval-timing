
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
    Truncate concatenated passages to at most `max_tokens` tokens.

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
    Retrieval database for HotpotQA using E5-base-v2 and FAISS.

    Downloads:
        - e5_Flat.index
        - hotpotqa_corpus.jsonl

    Query embedding:
        "query: <query>"

    Expected corpus fields:
        - id
        - title
        - content

    The implementation also supports "document" as a fallback
    for the passage text field.

    Example:
        db = HotpotQARetrievalDB(top_k=3)

        observation = db.search(
            "Which university did the director of Film X attend?"
        )

        print(observation)
    """

    DEFAULT_REPO_ID = (
        "adaptive-retrieval-timing/HotpotQA-ART-E5-Retrieval-Index"
    )
    DEFAULT_REPO_TYPE = "dataset"
    DEFAULT_MODEL_NAME = "intfloat/e5-base-v2"

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        repo_type: str = DEFAULT_REPO_TYPE,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = 3,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if max_length <= 0:
            raise ValueError("max_length must be > 0")

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.model_name = model_name
        self.top_k = top_k
        self.max_length = max_length

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load FAISS index and corpus.
        self.index = self._load_index()
        self.corpus = self._load_corpus()

        # FAISS row i must correspond to corpus row i.
        if len(self.corpus) != self.index.ntotal:
            raise RuntimeError(
                "Corpus / FAISS mismatch!\n"
                f"Corpus passages: {len(self.corpus)}\n"
                f"FAISS vectors:   {self.index.ntotal}"
            )

        # Load E5 model and tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        )
        self.model.eval()
        self.model.to(self.device)

    # --------------------------------------------------------------
    # Loading
    # --------------------------------------------------------------

    def _load_index(self):
        index_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="e5_Flat.index",
            repo_type=self.repo_type,
        )

        return faiss.read_index(index_path)

    def _load_corpus(self) -> List[Dict]:
        corpus_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="hotpotqa_corpus.jsonl",
            repo_type=self.repo_type,
        )

        corpus = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSONL at line "
                        f"{line_number}: {exc}"
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
        """
        Support either 'content' or 'document' as the passage field.
        """
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
            attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        )

    @torch.no_grad()
    def _encode_query(self, query: str):
        # Keep the E5 query prefix.
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
            raise ValueError("query must be a non-empty string")

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

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_passage_tokens: int = -1,
        tokenizer=None,
        return_passages: bool = False,
    ) -> Union[str, Tuple[str, List[str]]]:
        """
        Retrieve passages and return one observation string.

        Args:
            query:
                Retrieval query, e.g. an ART reasoning sentence.
            top_k:
                Number of passages to retrieve. Defaults to self.top_k.
            max_passage_tokens:
                Maximum token count for the concatenated passage text.
                -1 means no truncation.
            tokenizer:
                Tokenizer for truncation, required if max_passage_tokens
                is not -1.
            return_passages:
                If True, return (observation, passage_parts).

        Returns:
            Observation string, or (observation, passage_parts).
        """
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

        observation = append_retrieval_observation(passages)

        if return_passages:
            return observation, passage_parts

        return observation

    def search_with_metadata(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Return retrieved passage metadata, similarity scores and ranks.
        """
        return self._retrieve(
            query=query,
            top_k=top_k,
        )

    def __len__(self) -> int:
        return self.index.ntotal
