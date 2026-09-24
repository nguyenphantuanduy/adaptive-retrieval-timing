from typing import Callable, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.integrations import use_kernelized_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


# ============================================================
# ATTENTION MIXING
# ============================================================


def attention_mixing_forward(
    module: nn.Module,
    query_with_doc: torch.Tensor,
    key_with_doc: torch.Tensor,
    query_without_doc: torch.Tensor,
    key_without_doc: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    document_attention_mask: torch.Tensor | None = None,
    retrieval_decision: torch.Tensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
):
    # ========================================================
    # REPEAT KV
    # ========================================================

    key_states_with_doc = repeat_kv(
        key_with_doc,
        module.num_key_value_groups,
    )

    key_states_without_doc = repeat_kv(
        key_without_doc,
        module.num_key_value_groups,
    )

    value_states = repeat_kv(
        value,
        module.num_key_value_groups,
    )

    # ========================================================
    # ATTENTION SCORES — WITH DOCUMENT
    # ========================================================

    attn_scores_with_doc = torch.matmul(
        query_with_doc,
        key_states_with_doc.transpose(2, 3),
    ) * scaling

    # ========================================================
    # ATTENTION SCORES — WITHOUT DOCUMENT
    # ========================================================

    attn_scores_without_doc = torch.matmul(
        query_without_doc,
        key_states_without_doc.transpose(2, 3),
    ) * scaling

    # ========================================================
    # CAUSAL MASK
    # ========================================================

    if attention_mask is not None:

        attn_scores_with_doc = (
            attn_scores_with_doc
            + attention_mask
        )

        attn_scores_without_doc = (
            attn_scores_without_doc
            + attention_mask
        )

    # ========================================================
    # SOFTMAX — WITH DOCUMENT
    # ========================================================

    attn_weights_with_doc = (
        nn.functional.softmax(
            attn_scores_with_doc,
            dim=-1,
            dtype=torch.float32,
        ).to(query_with_doc.dtype)
    )

    # ========================================================
    # SOFTMAX — WITHOUT DOCUMENT
    # ========================================================

    if (
        document_attention_mask is not None
        and retrieval_decision is not None
    ):

        attn_weights_without_doc = (
            nn.functional.softmax(
                attn_scores_without_doc
                + document_attention_mask,
                dim=-1,
                dtype=torch.float32,
            ).to(query_without_doc.dtype)
        )

        # ----------------------------------------------------
        # RESHAPE RETRIEVAL DECISION
        # ----------------------------------------------------

        retrieval_decision = (
            retrieval_decision.view(
                -1,
                1,
                1,
                1,
            )
        )

        # ----------------------------------------------------
        # ATTENTION MIXING
        # ----------------------------------------------------

        attn_weights = (
            retrieval_decision
            * attn_weights_with_doc
            + (1.0 - retrieval_decision)
            * attn_weights_without_doc
        )

    else:

        attn_weights = attn_weights_with_doc

    # ========================================================
    # DROPOUT
    # ========================================================

    attn_weights = nn.functional.dropout(
        attn_weights,
        p=dropout,
        training=module.training,
    )

    attn_weights = attn_weights.to(
        value_states.dtype
    )

    # ========================================================
    # ATTENTION OUTPUT
    # ========================================================

    attn_output = torch.matmul(
        attn_weights,
        value_states,
    )

    attn_output = (
        attn_output
        .transpose(1, 2)
        .contiguous()
    )

    return attn_output, attn_weights


# ============================================================
# QWEN2 ATTENTION MIXING PROBE
# ============================================================


@use_kernelized_func(apply_rotary_pos_emb)
class AttentionMixingProbeAttention(Qwen2Attention):

    def forward(
        self,
        hidden_states: torch.Tensor,

        position_embeddings_with_doc: tuple[
            torch.Tensor,
            torch.Tensor,
        ],

        position_embeddings_without_doc: tuple[
            torch.Tensor,
            torch.Tensor,
        ],

        attention_mask: torch.Tensor | None,

        past_key_values: Cache | None = None,

        document_attention_mask: torch.Tensor | None = None,

        retrieval_decision: torch.Tensor | None = None,

        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        # ====================================================
        # INPUT SHAPE
        # ====================================================

        input_shape = hidden_states.shape[:-1]

        hidden_shape = (
            *input_shape,
            -1,
            self.head_dim,
        )

        # ====================================================
        # Q / K / V PROJECTIONS
        # ====================================================

        # Qwen2 does NOT have q_norm / k_norm.
        #
        # Qwen2:
        #   q_proj -> reshape -> transpose
        #   k_proj -> reshape -> transpose
        #   v_proj -> reshape -> transpose

        query_states = (
            self.q_proj(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )

        key_states = (
            self.k_proj(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )

        value_states = (
            self.v_proj(hidden_states)
            .view(hidden_shape)
            .transpose(1, 2)
        )

        # ====================================================
        # ROPE — WITH DOCUMENT
        # ====================================================

        cos_with_doc, sin_with_doc = (
            position_embeddings_with_doc
        )

        (
            query_states_with_doc,
            key_states_with_doc,
        ) = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos_with_doc,
            sin_with_doc,
        )

        # ====================================================
        # ROPE — WITHOUT DOCUMENT
        # ====================================================

        cos_without_doc, sin_without_doc = (
            position_embeddings_without_doc
        )

        (
            query_states_without_doc,
            key_states_without_doc,
        ) = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos_without_doc,
            sin_without_doc,
        )

        # ====================================================
        # CACHE
        # ====================================================

        value_states_with_doc = value_states

        if past_key_values is not None:

            # ------------------------------------------------
            # WITH DOCUMENT CACHE
            # ------------------------------------------------

            (
                key_states_with_doc,
                value_states_with_doc,
            ) = past_key_values.cache_with_doc.update(
                key_states_with_doc,
                value_states,
                self.layer_idx,
            )

            # ------------------------------------------------
            # WITHOUT DOCUMENT CACHE
            # ------------------------------------------------

            (
                key_states_without_doc,
                value_states_without_doc,
            ) = past_key_values.cache_without_doc.update(
                key_states_without_doc,
                value_states,
                self.layer_idx,
            )

        # ====================================================
        # ATTENTION MIXING
        # ====================================================

        attn_output, attn_weights = (
            attention_mixing_forward(
                self,

                query_states_with_doc,
                key_states_with_doc,

                query_states_without_doc,
                key_states_without_doc,

                value_states_with_doc,

                attention_mask,

                dropout=(
                    0.0
                    if not self.training
                    else self.attention_dropout
                ),

                scaling=self.scaling,

                document_attention_mask=(
                    document_attention_mask
                ),

                retrieval_decision=(
                    retrieval_decision
                ),

                **kwargs,
            )
        )

        # ====================================================
        # OUTPUT PROJECTION
        # ====================================================

        attn_output = (
            attn_output
            .reshape(*input_shape, -1)
            .contiguous()
        )

        attn_output = self.o_proj(
            attn_output
        )

        return attn_output, attn_weights