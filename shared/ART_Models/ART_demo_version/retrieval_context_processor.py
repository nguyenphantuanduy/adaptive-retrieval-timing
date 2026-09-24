from collections import deque

import spacy
import torch
from sentencesplit import StreamSegmenter
from transformers import PreTrainedTokenizerBase


class SentenceContextWindow:
    def __init__(self, max_sentences: int):
        if max_sentences <= 0:
            raise ValueError(
                "max_sentences must be positive"
            )

        self.max_sentences = max_sentences

        self.sentences: deque[torch.LongTensor] = deque(
            maxlen=max_sentences
        )

    def add(self, sentence_ids: torch.LongTensor) -> None:
        if sentence_ids.dtype != torch.long:
            raise TypeError(
                "sentence_ids must be a torch.LongTensor"
            )

        if sentence_ids.ndim != 1:
            raise ValueError(
                "sentence_ids must be a 1D tensor"
            )

        self.sentences.append(sentence_ids)

    def get(self) -> torch.LongTensor:
        if not self.sentences:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return torch.cat(
            list(self.sentences),
            dim=0,
        )

    def get_last(self) -> torch.LongTensor:
        """
        Return the most recently completed sentence.
        """

        if not self.sentences:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return self.sentences[-1]


def load_spacy_model(
    model_name: str = "en_core_web_sm",
):
    try:
        return spacy.load(model_name)

    except OSError:
        print(
            f"spaCy model '{model_name}' not found. "
            "Downloading..."
        )

        spacy.cli.download(model_name)

        return spacy.load(model_name)


class RetrievalContextProcessor:
    """
    Online sentence-level retrieval context processor.

    ART v1 retrieval query policy:

        Before any sentence is completed:
            query = original question

        After a sentence is completed:
            query = latest completed sentence

    The question is used only as the initial fallback query.
    It is NOT concatenated with generated sentences.

    Generation:

        token
          ↓
        StreamSegmenter
          ↓
        sentence boundary?
          │
          ├── No completed sentence yet
          │       ↓
          │    question
          │
          └── Sentence completed
                  ↓
             latest sentence
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        question: str,
        max_sentences: int = 3,
        language: str = "en",
    ):
        self.tokenizer = tokenizer

        self.question = question

        self.segmenter = StreamSegmenter(
            language=language
        )

        self.nlp = load_spacy_model()

        self.context_window = SentenceContextWindow(
            max_sentences=max_sentences
        )

        # Token IDs belonging to the currently incomplete
        # sentence.
        self.buffer: list[int] = []

    def feed_token(
        self,
        token_id: torch.LongTensor,
    ) -> bool:
        """
        Feed one generated token to the processor.

        Returns:
            True:
                A sentence boundary was detected.

            False:
                The current sentence is still incomplete.
        """

        if not isinstance(
            token_id,
            torch.Tensor,
        ):
            raise TypeError(
                "token_id must be a torch.Tensor"
            )

        if token_id.dtype != torch.long:
            raise TypeError(
                "token_id must have dtype torch.long"
            )

        if token_id.numel() != 1:
            raise ValueError(
                "token_id must contain exactly one token ID"
            )

        token_id = token_id.item()

        # Keep the original token ID as the canonical
        # representation.
        self.buffer.append(token_id)

        # Decode only this token for online sentence
        # boundary detection.
        piece = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        self.segmenter.feed(piece)

        completed_sentences = (
            self.segmenter.get_completed_sentences()
        )

        if not completed_sentences:
            return False

        # A sentence boundary has been confirmed.
        sentence_ids = torch.tensor(
            self.buffer,
            dtype=torch.long,
        )

        self.context_window.add(
            sentence_ids
        )

        # Start collecting the next sentence.
        self.buffer.clear()

        return True

    def build_retrieval_query(self) -> str:
        """
        Build the retrieval query.

        If no sentence has been completed yet:
            return the original question.

        Otherwise:
            return the latest completed sentence.

        The question is never concatenated with
        the generated sentence.
        """

        # No completed sentence yet.
        if not self.context_window.sentences:
            return self.question

        sentence_ids = (
            self.context_window.get_last()
        )

        if sentence_ids.numel() == 0:
            return self.question

        sentence_text = self.tokenizer.decode(
            sentence_ids.tolist(),
            skip_special_tokens=False,
        )

        # Re-segment with spaCy so that the returned query
        # corresponds to a clean sentence.
        doc = self.nlp(sentence_text)

        sentences = [
            sentence.text.strip()
            for sentence in doc.sents
            if sentence.text.strip()
        ]

        if not sentences:
            return sentence_text.strip()

        return sentences[-1]

    def get_context(self) -> torch.LongTensor:
        """
        Return all cached completed sentence token IDs.
        """

        return self.context_window.get()

    def get_last_sentence(self) -> torch.LongTensor:
        """
        Return the token IDs of the latest completed sentence.
        """

        return self.context_window.get_last()

    def get_current_buffer(self) -> torch.LongTensor:
        """
        Return the token IDs of the currently incomplete
        sentence.
        """

        if not self.buffer:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return torch.tensor(
            self.buffer,
            dtype=torch.long,
        )