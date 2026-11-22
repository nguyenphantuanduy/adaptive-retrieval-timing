import json
from typing import List, Dict, Optional

import faiss
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel


class MuSiQueRetrievalDB:
    """
    Lightweight retrieval database for the MuSiQue E5 + FAISS artifact.

    The class:
      1. Downloads/loads the FAISS index and MuSiQue JSONL corpus.
      2. Loads E5-base-v2.
      3. Encodes an incoming query using the same E5 convention as the
         original retrieval script: "query: <query>".
      4. Searches FAISS.
      5. Returns all retrieved passages as ONE formatted string.

    The returned string has the following format:

        <observation>
        Title: United States Capitol
        The United States Congress was established ...

        Title: Philadelphia
        Philadelphia ...

        Title: ...
        ...
        </observation>

    Example
    -------
    db = MuSiQueRetrievalDB(top_k=3)

    documents = db.search(
        "To answer this, we first need to identify the city that used "
        "to be the U.S. capitol."
    )

    print(documents)
    """

    DEFAULT_REPO_ID = "adaptive-retrieval-timing/MuSiQue-ART-E5-Retrieval-Index"
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

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.model_name = model_name
        self.top_k = top_k
        self.max_length = max_length

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --------------------------------------------------------------
        # Load retrieval artifacts
        # --------------------------------------------------------------

        self.index = self._load_index()
        self.corpus = self._load_corpus()

        # FAISS row i must correspond to corpus row i.
        if len(self.corpus) != self.index.ntotal:
            raise RuntimeError(
                "Corpus / FAISS mismatch!\n"
                f"Corpus passages: {len(self.corpus)}\n"
                f"FAISS vectors:   {self.index.ntotal}"
            )

        # --------------------------------------------------------------
        # Load E5
        # --------------------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )

        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

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
            filename="musique_corpus.jsonl",
            repo_type=self.repo_type,
        )

        corpus = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    corpus.append(json.loads(line))

        return corpus

    # ------------------------------------------------------------------
    # E5 embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask[..., None].bool()

        masked_hidden = last_hidden_state.masked_fill(
            ~mask,
            0.0,
        )

        return masked_hidden.sum(dim=1) / attention_mask.sum(
            dim=1
        )[..., None]

    @torch.no_grad()
    def _encode_query(self, query: str):
        # Keep exactly the same E5 query convention as the original script.
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

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve the top-k passages for a query and return them as ONE string.

        Parameters
        ----------
        query:
            The incoming retrieval query. In ART this can be one CoT sentence.

        top_k:
            Number of passages to retrieve. If omitted, uses the value
            supplied when constructing the database.

        Returns
        -------
        str
            One formatted observation string:

                <observation>
                Title: ...
                document...

                Title: ...
                document...
                </observation>

            The document text is taken from the JSONL "document" field.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = top_k if top_k is not None else self.top_k

        if k <= 0:
            raise ValueError("top_k must be > 0")

        # FAISS cannot return more rows than exist in the corpus.
        k = min(k, self.index.ntotal)

        query_embedding = self._encode_query(query)

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        document_parts = [
            "<observation>\n"
        ]

        for idx in indices[0]:
            idx = int(idx)

            # Defensive check for FAISS invalid index values.
            if idx < 0:
                continue

            doc = self.corpus[idx]

            document_parts.append(
                f"Title: {doc['title']}\n"
                f"{doc['document']}\n"
            )

        document_parts.append(
            "</observation>\n"
        )

        return "\n".join(document_parts)

    def search_with_metadata(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Same retrieval operation as search(), but returns metadata for each
        retrieved passage.

        This is useful for debugging/evaluation.

        Unlike search(), this method returns a List[Dict].
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = top_k if top_k is not None else self.top_k

        if k <= 0:
            raise ValueError("top_k must be > 0")

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
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["document"],
                }
            )

        return results

    def __len__(self):
        return self.index.ntotal


# ----------------------------------------------------------------------
# Simple standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    db = MuSiQueRetrievalDB(top_k=3)

    query = (
        "To answer this, we first need to identify the city that used "
        "to be the U.S. capitol."
    )

    documents = db.search(query)

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print("RETRIEVED DOCUMENTS")
    print("=" * 80)
    print(documents)
