from collections import deque

import spacy
import torch
from sentencesplit import StreamSegmenter
from transformers import PreTrainedTokenizerBase


class SentenceContextWindow:
    def __init__(self, max_sentences: int):
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")

        self.max_sentences = max_sentences
        self.sentences: deque[torch.LongTensor] = deque(
            maxlen=max_sentences
        )

    def add(self, sentence_ids: torch.LongTensor) -> None:
        if sentence_ids.dtype != torch.long:
            raise TypeError("sentence_ids must be a torch.LongTensor")

        if sentence_ids.ndim != 1:
            raise ValueError("sentence_ids must be a 1D tensor")

        self.sentences.append(sentence_ids)

    def get(self) -> torch.LongTensor:
        if not self.sentences:
            return torch.empty(0, dtype=torch.long)

        return torch.cat(list(self.sentences), dim=0)


def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)

    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)


class RetrievalContextProcessor:
    def __init__(
        self,
        question: str,
        tokenizer: PreTrainedTokenizerBase,
        max_sentences: int = 3,
        language: str = "en",
    ):
        self.question = question
        self.tokenizer = tokenizer

        self.segmenter = StreamSegmenter(language=language)
        self.nlp = load_spacy_model()

        self.context_window = SentenceContextWindow(
            max_sentences=max_sentences
        )

        # Token IDs belonging to the currently incomplete sentence.
        self.buffer: list[int] = []

    def feed_token(self, token_id: torch.LongTensor) -> bool:
        """
        Feed one generated token to the processor.

        Returns:
            True  -> a sentence boundary was detected.
            False -> the current sentence is still incomplete.
        """
        if not isinstance(token_id, torch.Tensor):
            raise TypeError("token_id must be a torch.Tensor")

        if token_id.dtype != torch.long:
            raise TypeError("token_id must have dtype torch.long")

        if token_id.numel() != 1:
            raise ValueError("token_id must contain exactly one token ID")

        token_id = token_id.item()

        # Keep the original token ID as the canonical representation.
        self.buffer.append(token_id)

        # Decode only this token for online sentence boundary detection.
        piece = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        self.segmenter.feed(piece)

        completed_sentences = self.segmenter.get_completed_sentences()

        if not completed_sentences:
            return False

        # A sentence boundary has been confirmed.
        sentence_ids = torch.tensor(
            self.buffer,
            dtype=torch.long,
        )

        self.context_window.add(sentence_ids)

        # The buffer now starts collecting the next sentence.
        self.buffer.clear()

        return True

    def build_retrieval_query(self) -> str:
        """
        Build a retrieval query from the current sentence context window.

        This method should be called when ART decides to retrieve.
        """
        context_ids = self.context_window.get()

        if context_ids.numel() == 0:
            return self.question

        # Decode the entire context at once so that token decoding
        # preserves the original token sequence as faithfully as possible.
        context_text = self.tokenizer.decode(
            context_ids.tolist(),
            skip_special_tokens=False,
        )

        # Re-segment the retrieval context using spaCy.
        doc = self.nlp(context_text)

        sentences = [
            sentence.text.strip()
            for sentence in doc.sents
            if sentence.text.strip()
        ]

        if not sentences:
            return self.question

        last_sentence = sentences[-1]

        return f"{self.question} {last_sentence}"