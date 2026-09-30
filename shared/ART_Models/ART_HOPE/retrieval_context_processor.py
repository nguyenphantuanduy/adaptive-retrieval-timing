from collections import deque

import torch
from sentencesplit import Segmenter, StreamSegmenter
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

    def add(
        self,
        sentence_ids: torch.LongTensor,
    ) -> None:
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
        if not self.sentences:
            return torch.empty(
                0,
                dtype=torch.long,
            )

        return self.sentences[-1]


class RetrievalContextProcessor:
    """
    Online sentence-level retrieval context processor.

    There are two separate responsibilities:

    1. Online sentence boundary detection
       ----------------------------------
       self.segmenter receives generated tokens one by one.
       Its only purpose is to detect when retrieval should
       be triggered.

    2. Retrieval query extraction
       ---------------------------
       build_retrieval_query() uses ALL generated tokens and
       performs a fresh sentence segmentation using
       Segmenter.segment_with_lookahead().

       If should_wait_for_more=True, the final segment is
       still incomplete / unstable and is therefore NOT used
       as the retrieval query.
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
        self.language = language

        # Online sentence boundary detector.
        # Used ONLY to decide when retrieval is triggered.
        self.segmenter = StreamSegmenter(
            language=language
        )

        # Stores sentence token IDs detected by the
        # online splitter.
        self.context_window = SentenceContextWindow(
            max_sentences=max_sentences
        )

        # Current online sentence buffer.
        self.buffer: list[int] = []

        # ALL generated token IDs.
        # This list is NEVER cleared during generation.
        self.generated_token_ids: list[int] = []

    def feed_token(
        self,
        token_id: torch.LongTensor,
    ) -> bool:
        """
        Feed one generated token to the processor.

        Returns:
            True:
                The online sentence splitter detected
                a sentence boundary.

            False:
                No sentence boundary was detected.
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

        # Keep token globally.
        self.generated_token_ids.append(
            token_id
        )

        # Keep token in current online buffer.
        self.buffer.append(
            token_id
        )

        # Decode current token for online boundary detection.
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

        # Online boundary detected.
        #
        # The online boundary may be reported on the first
        # token of the next sentence. Therefore this buffer
        # is NOT trusted for retrieval-query extraction.

        sentence_ids = torch.tensor(
            self.buffer,
            dtype=torch.long,
        )

        self.context_window.add(
            sentence_ids
        )

        # Start collecting the next online sentence.
        self.buffer.clear()

        return True

    def build_retrieval_query(self) -> str:
        """
        Build the retrieval query from ALL generated text.

        This method is independent of context_window and
        independent of the timing of online boundary detection.

        The complete generated text is re-segmented from scratch
        using Segmenter.segment_with_lookahead().

        If the final segment is not yet stable
        (should_wait_for_more=True), that final segment is
        excluded from the retrieval query.
        """

        # No generated text yet.
        if not self.generated_token_ids:
            return self.question

        # Decode ALL generated tokens.
        full_text = self.tokenizer.decode(
            self.generated_token_ids,
            skip_special_tokens=False,
        )

        if not full_text.strip():
            return self.question

        # Fresh static sentence segmentation with lookahead.
        segmenter = Segmenter(
            language=self.language
        )

        result = segmenter.segment_with_lookahead(
            full_text
        )

        # Clean segments.
        segments = [
            segment.strip()
            for segment in result.segments
            if segment.strip()
        ]

        # No segments found.
        if not segments:
            return self.question

        # If the splitter says that it needs more text,
        # the final segment is still incomplete / unstable.
        #
        # Example:
        #
        #   ["Richard Allen founded the church.", "The"]
        #
        # "The" must NOT become the retrieval query.
        if result.should_wait_for_more:
            confirmed_segments = segments[:-1]
        else:
            confirmed_segments = segments

        # There is no confirmed sentence yet.
        # Fall back to the original question.
        if not confirmed_segments:
            return self.question

        # Return the latest CONFIRMED completed sentence.
        return confirmed_segments[-1]

    def get_context(self) -> torch.LongTensor:
        """
        Return all cached online-detected sentence token IDs.
        """

        return self.context_window.get()

    def get_last_sentence(self) -> torch.LongTensor:
        """
        Return the token IDs of the latest sentence detected
        by the online boundary detector.
        """

        return self.context_window.get_last()

    def get_current_buffer(self) -> torch.LongTensor:
        """
        Return the token IDs currently stored in the online
        sentence buffer.
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

    def get_generated_text(self) -> str:
        """
        Return the complete generated text accumulated so far.
        """

        if not self.generated_token_ids:
            return ""

        return self.tokenizer.decode(
            self.generated_token_ids,
            skip_special_tokens=False,
        )