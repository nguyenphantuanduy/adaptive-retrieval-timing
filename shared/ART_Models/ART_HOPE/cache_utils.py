import copy

import torch

from transformers import (
    Cache,
    DynamicCache,
)

from .retrieval_context_processor import (
    RetrievalContextProcessor,
)


class ARTDecisionCache(Cache):
    """
    Cache for ART's decision branch.

    Two independent KV cache states are maintained:

        cache_with_doc
            -> Decision branch KV cache using the
               document-aware path.

        cache_without_doc
            -> Decision branch KV cache using the
               document-free path.
    """

    def __init__(
        self,
        ddp_cache_data=None,
        config=None,
        offloading=False,
        offload_only_non_sliding=False,
    ):

        # ====================================================
        # WITH DOCUMENT
        # ====================================================

        self.cache_with_doc = DynamicCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=(
                offload_only_non_sliding
            ),
        )

        # ====================================================
        # WITHOUT DOCUMENT
        # ====================================================

        self.cache_without_doc = DynamicCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=(
                offload_only_non_sliding
            ),
        )

    # ========================================================
    # CORE HF CACHE API
    # ========================================================

    def get_seq_length(
        self,
        layer_idx: int = 0,
    ) -> int:
        """
        Return the autoregressive sequence length.

        Both caches represent the same generated sequence,
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
        Reorder both decision-branch cache states
        identically during beam search / generation.
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
    # SYNCHRONIZATION
    # ========================================================

    def synchronize_cache(self) -> None:
        """
        Synchronize the document-free decision cache
        with the document-aware decision cache.

        The copied KV tensors are independent clones.
        """

        copy_dynamic_cache(
            source_cache=self.cache_with_doc,
            destination_cache=self.cache_without_doc,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """
        Clear both decision-branch KV cache states.
        """

        self.cache_with_doc.reset()
        self.cache_without_doc.reset()

    # ========================================================
    # CACHE API
    # ========================================================

    def get_cache_with_doc(self):
        """
        Return the decision branch document-aware KV cache.
        """

        return self.cache_with_doc

    def get_cache_without_doc(self):
        """
        Return the decision branch document-free KV cache.
        """

        return self.cache_without_doc

    # ========================================================
    # STATUS
    # ========================================================

    def has_cache(self) -> bool:
        """
        Whether the decision branch KV cache has been
        initialized.
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
        """

        return self.cache_with_doc.layers


class ARTCache(Cache):
    """
    Cache for ART's dual-branch attention paths and
    sentence-level retrieval context.

    The LM branch keeps the original cache structure:

        cache_with_doc
            -> LM branch KV cache using the document-aware path.

        cache_without_doc
            -> LM branch KV cache using the document-free path.

    The decision branch uses a separate ARTDecisionCache:

        decision_cache
            -> decision_cache.cache_with_doc
            -> decision_cache.cache_without_doc

    The LM branch uses the original model configuration.

    The decision branch uses a copied configuration whose
    num_hidden_layers is replaced by num_decision_layers.

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
    ):
        # ====================================================
        # CONFIGURATION
        # ====================================================

        self.question = question
        self.max_sentences = max_sentences
        self.language = language
        self.tokenizer = tokenizer
        self.counter = 0

        self.config = config

        # ====================================================
        # DECISION BRANCH CONFIGURATION
        # ====================================================

        if config is None:
            raise ValueError(
                "ARTCache requires a model config."
            )

        self.decision_config = copy.deepcopy(config)

        self.decision_config.num_hidden_layers = (
            self.decision_config.num_decision_layers
        )

        # ====================================================
        # LM BRANCH
        # ====================================================

        # ----------------------------------------------------
        # LM branch: WITH DOCUMENT
        # ----------------------------------------------------

        self.cache_with_doc = DynamicCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=(
                offload_only_non_sliding
            ),
        )

        # ----------------------------------------------------
        # LM branch: WITHOUT DOCUMENT
        # ----------------------------------------------------

        self.cache_without_doc = DynamicCache(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=(
                offload_only_non_sliding
            ),
        )

        # ====================================================
        # DECISION BRANCH
        # ====================================================

        self.decision_cache = ARTDecisionCache(
            ddp_cache_data=ddp_cache_data,
            config=self.decision_config,
            offloading=offloading,
            offload_only_non_sliding=(
                offload_only_non_sliding
            ),
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

        # ====================================================
        # RETRIEVAL STATE
        # ====================================================

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

        The LM and decision branches represent the same
        generated sequence, so their corresponding caches
        should have identical sequence lengths.
        """

        return self.cache_with_doc.get_seq_length(
            layer_idx
        )

    def reorder_cache(
        self,
        beam_idx: torch.LongTensor,
    ):
        """
        Reorder the LM and decision branch caches
        identically during beam search / generation
        reordering.
        """

        # ----------------------------------------------------
        # LM branch
        # ----------------------------------------------------

        self.cache_with_doc.reorder_cache(
            beam_idx
        )

        self.cache_without_doc.reorder_cache(
            beam_idx
        )

        # ----------------------------------------------------
        # Decision branch
        # ----------------------------------------------------

        self.decision_cache.reorder_cache(
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
    # RETRIEVAL STATE
    # ========================================================

    def update_retrieval_state(
        self,
        document_attention_mask,
        retrieval_decision,
    ) -> None:
        """
        Update retrieval-related states and synchronize both
        LM and decision document-free caches with their
        corresponding document-aware caches.

        LM:
            cache_without_doc
                <- cache_with_doc

        Decision:
            decision_cache.cache_without_doc
                <- decision_cache.cache_with_doc
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
        Synchronize the document-free caches with their
        corresponding document-aware caches.

        The copied KV tensors are independent clones.
        """

        # ----------------------------------------------------
        # LM branch
        # ----------------------------------------------------

        copy_dynamic_cache(
            source_cache=self.cache_with_doc,
            destination_cache=self.cache_without_doc,
        )

        # ----------------------------------------------------
        # Decision branch
        # ----------------------------------------------------

        self.decision_cache.synchronize_cache()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """
        Clear all LM and decision KV cache branches and
        reset the retrieval-context processor.
        """

        # ----------------------------------------------------
        # LM branch
        # ----------------------------------------------------

        self.cache_with_doc.reset()
        self.cache_without_doc.reset()

        # ----------------------------------------------------
        # Decision branch
        # ----------------------------------------------------

        self.decision_cache.reset()

        # ----------------------------------------------------
        # Retrieval context
        # ----------------------------------------------------

        self.retrieval_context = (
            RetrievalContextProcessor(
                tokenizer=self.tokenizer,
                question=self.question,
                max_sentences=self.max_sentences,
                language=self.language,
            )
        )

        # ----------------------------------------------------
        # Retrieval state
        # ----------------------------------------------------

        self.document_attention_mask = None
        self.retrieval_decision = None
        self.attention_mask = None

    # ========================================================
    # LM CACHE API
    # ========================================================

    def get_cache_with_doc(self):
        """
        Return the LM branch document-aware KV cache.
        """

        return self.cache_with_doc

    def get_cache_without_doc(self):
        """
        Return the LM branch document-free KV cache.
        """

        return self.cache_without_doc

    # ========================================================
    # DECISION CACHE API
    # ========================================================

    def get_decision_cache_with_doc(self):
        """
        Return the decision branch document-aware KV cache.
        """

        return self.decision_cache.get_cache_with_doc()

    def get_decision_cache_without_doc(self):
        """
        Return the decision branch document-free KV cache.
        """

        return self.decision_cache.get_cache_without_doc()

    def get_decision_cache(self):
        """
        Return the complete decision branch cache.
        """

        return self.decision_cache

    # ========================================================
    # STATUS
    # ========================================================

    def has_cache(self) -> bool:
        """
        Whether the LM branch KV cache has been initialized.
        """

        return (
            self.cache_with_doc.get_seq_length() > 0
        )

    def has_decision_cache(self) -> bool:
        """
        Whether the decision branch KV cache has been
        initialized.
        """

        return self.decision_cache.has_cache()

    # ========================================================
    # HF COMPATIBILITY
    # ========================================================

    @property
    def layers(self):
        """
        Expose the LM document-aware cache layers as the
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

        destination_layer.keys = (
            source_layer.keys.clone()
        )

        destination_layer.values = (
            source_layer.values.clone()
        )

        destination_layer.is_initialized = True