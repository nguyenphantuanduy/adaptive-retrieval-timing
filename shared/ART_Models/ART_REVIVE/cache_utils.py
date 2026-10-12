import torch
from transformers import (
    Cache,
    DynamicCache,
)

from .retrieval_context_processor import (
    RetrievalContextProcessor,
)


class ARTCache(Cache):
    """
    Cache for ART's dual-attention paths and
    sentence-level retrieval context.

    Three states are maintained:

        cache_with_doc
            -> KV cache using the document-aware path.

        cache_without_doc
            -> KV cache using the document-free path.

        retrieval_context
            -> Online sentence-level context used to build
               retrieval queries.

    Retrieval query policy for ART v1:

        Before any sentence is completed:
            -> original question

        After a sentence is completed:
            -> latest completed sentence

    The original question is used only as the initial
    fallback query. It is NOT concatenated with
    generated sentences.

    The tokenizer is provided externally and shared with
    the retrieval-context processor.
    """

    def __init__(
        self,
        question: str,
        tokenizer,
        max_sentences: int = 3,
        language: str = "en",
        ddp_cache_data=None,
        config=None,
        offloading=False,
        offload_only_non_sliding=False,
        probability: int = 0.3,
    ):
        # ====================================================
        # CONFIGURATION
        # ====================================================

        self.question = question
        self.max_sentences = max_sentences
        self.language = language
        self.tokenizer = tokenizer
        self.counter = 0
        self.probability = probability

        # ====================================================
        # WITH DOCUMENT
        # ====================================================

        self.cache_with_doc = DynamicCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        # ====================================================
        # WITHOUT DOCUMENT
        # ====================================================

        self.cache_without_doc = DynamicCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        # ====================================================
        # RETRIEVAL CONTEXT
        # ====================================================

        self.retrieval_context = (
            RetrievalContextProcessor(
                tokenizer=self.tokenizer,
                question=self.question,
                max_sentences=self.max_sentences,
                language=self.language,
            )
        )

        self.document_attention_mask = None
        self.retrieval_decision = None
        self.attention_mask = None

    # ========================================================
    # CORE HF CACHE API
    # ========================================================

    def get_seq_length(
        self,
        layer_idx: int = 0,
    ) -> int:
        """
        Return the autoregressive sequence length.

        Both KV caches represent the same generated sequence,
        so they should have identical sequence lengths.
        """

        return self.cache_with_doc.get_seq_length(
            layer_idx
        )

    def reorder_cache(
        self,
        beam_idx: torch.LongTensor,
    ):
        """
        Reorder both cache branches identically during
        beam search / generation reordering.
        """

        self.cache_with_doc.reorder_cache(
            beam_idx
        )

        self.cache_without_doc.reorder_cache(
            beam_idx
        )

        return self

    def to_legacy_cache(self):
        return self

    # ========================================================
    # RETRIEVAL CONTEXT API
    # ========================================================

    def feed_token(
        self,
        token_id: torch.LongTensor,
    ) -> bool:
        """
        Feed one generated token into the retrieval-context
        processor.

        Returns:
            True:
                A sentence boundary was detected.

            False:
                The current sentence is still incomplete.
        """

        return self.retrieval_context.feed_token(
            token_id
        )

    def build_retrieval_query(self) -> str:
        """
        Build the retrieval query.

        Before any sentence is completed:
            -> original question

        After a sentence is completed:
            -> latest completed sentence
        """

        return (
            self.retrieval_context
            .build_retrieval_query()
        )

    # ========================================================
    # UTIL
    # ========================================================

    def update_retrieval_state(
        self,
        document_attention_mask,
        retrieval_decision,
    ) -> None:
        """
        Update the retrieval-related states and synchronize
        the document-free KV cache with the document-aware
        KV cache.

        The document-free cache becomes an independent copy
        of the current document-aware cache.
        """

        self.document_attention_mask = (
            document_attention_mask
        )

        self.retrieval_decision = (
            retrieval_decision
        )

        self.synchronize_cache()

    def synchronize_cache(self) -> None:
        """
        Synchronize the document-free KV cache with the
        current document-aware KV cache.

        The document-free cache becomes an independent copy
        of the current document-aware cache.
        """

        copy_dynamic_cache(
            source_cache=self.cache_with_doc,
            destination_cache=self.cache_without_doc,
        )

    def reset(self):
        """
        Clear both KV cache branches and reset the
        retrieval-context processor.
        """

        self.cache_with_doc.reset()
        self.cache_without_doc.reset()

        self.retrieval_context = (
            RetrievalContextProcessor(
                tokenizer=self.tokenizer,
                question=self.question,
                max_sentences=self.max_sentences,
                language=self.language,
            )
        )

    def get_cache_with_doc(self):
        """
        Return the document-aware KV cache.
        """

        return self.cache_with_doc

    def get_cache_without_doc(self):
        """
        Return the document-free KV cache.
        """

        return self.cache_without_doc

    def has_cache(self) -> bool:
        """
        Whether the KV cache has been initialized.
        """

        return (
            self.cache_with_doc.get_seq_length() > 0
        )

    # ========================================================
    # HF COMPATIBILITY
    # ========================================================

    @property
    def layers(self):
        """
        Expose the document-aware cache layers as the
        primary cache layers.

        This is mainly for compatibility with code expecting
        Cache.layers.
        """

        return self.cache_with_doc.layers


def copy_dynamic_cache(
    source_cache: DynamicCache,
    destination_cache: DynamicCache,
) -> None:
    """
    Reset the destination DynamicCache and copy the complete
    KV state from the source DynamicCache.

    The copied key/value tensors are cloned, so source and
    destination do not share tensor storage.
    """

    destination_cache.reset()

    for source_layer, destination_layer in zip(
        source_cache.layers,
        destination_cache.layers,
    ):
        if not source_layer.is_initialized:
            continue

        destination_layer.keys = source_layer.keys.clone()
        destination_layer.values = source_layer.values.clone()
        destination_layer.is_initialized = True