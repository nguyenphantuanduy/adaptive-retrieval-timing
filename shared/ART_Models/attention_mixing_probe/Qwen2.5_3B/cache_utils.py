import torch
from transformers import Cache, DynamicCache


class ARTCache(Cache):
    """
    Cache for ART's dual-attention paths.

    Two independent KV caches are maintained:

        cache_with_doc
            -> KV cache using the original document-aware positions.

        cache_without_doc
            -> KV cache using document-removed positional positions.

    This allows ART to maintain two independent autoregressive
    attention states during generation.
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
    # UTIL
    # ========================================================

    def reset(self):
        """
        Clear both cache branches.
        """

        self.cache_with_doc.reset()
        self.cache_without_doc.reset()

    def get_cache_with_doc(self):
        return self.cache_with_doc

    def get_cache_without_doc(self):
        return self.cache_without_doc

    def has_cache(self) -> bool:
        """
        Whether the cache has been initialized.
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