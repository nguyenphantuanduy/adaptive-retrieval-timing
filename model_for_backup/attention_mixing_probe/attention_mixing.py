from typing import Callable, Optional
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.integrations import use_kernelized_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import TransformersKwargs
try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from .utils import remove_document_from_rotary_embeddings

def attention_mixing_forward(
    module: nn.Module,
    query_doc: torch.Tensor,
    key_doc: torch.Tensor,
    query_no_doc: torch.Tensor,
    key_no_doc: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    document_attention_mask: torch.Tensor | None = None,
    retrieval_decision: torch.Tensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
):
    # ========================================================
    # REPEAT KV
    # ========================================================

    key_states_doc = repeat_kv(
        key_doc,
        module.num_key_value_groups,
    )

    key_states_no_doc = repeat_kv(
        key_no_doc,
        module.num_key_value_groups,
    )

    value_states = repeat_kv(
        value,
        module.num_key_value_groups,
    )

    # ========================================================
    # ATTENTION SCORES — WITH DOCUMENT
    # ========================================================

    attn_scores_doc = torch.matmul(
        query_doc,
        key_states_doc.transpose(2, 3),
    ) * scaling

    # ========================================================
    # ATTENTION SCORES — WITHOUT DOCUMENT
    # ========================================================

    attn_scores_no_doc = torch.matmul(
        query_no_doc,
        key_states_no_doc.transpose(2, 3),
    ) * scaling

    # ========================================================
    # CAUSAL MASK
    # ========================================================

    if attention_mask is not None:

        causal_mask_doc = (
            attention_mask[
                :,
                :,
                :,
                :key_states_doc.shape[-2],
            ]
        )

        causal_mask_no_doc = (
            attention_mask[
                :,
                :,
                :,
                :key_states_no_doc.shape[-2],
            ]
        )

        attn_scores_doc = (
            attn_scores_doc
            + causal_mask_doc
        )

        attn_scores_no_doc = (
            attn_scores_no_doc
            + causal_mask_no_doc
        )

    # ========================================================
    # SOFTMAX — WITH DOCUMENT
    # ========================================================

    attn_weights_with_doc = (
        nn.functional.softmax(
            attn_scores_doc,
            dim=-1,
            dtype=torch.float32,
        ).to(query_doc.dtype)
    )

    # ========================================================
    # SOFTMAX — WITHOUT DOCUMENT
    # ========================================================

    if (
        document_attention_mask is not None
        and retrieval_decision is not None
    ):

        attn_weights_no_doc = (
            nn.functional.softmax(
                attn_scores_no_doc
                + document_attention_mask,
                dim=-1,
                dtype=torch.float32,
            ).to(query_no_doc.dtype)
        )

        # ----------------------------------------------------
        # Reshape retrieval decision
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
        # Attention mixing
        # ----------------------------------------------------

        attn_weights = (
            retrieval_decision
            * attn_weights_with_doc
            +
            (1.0 - retrieval_decision)
            * attn_weights_no_doc
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


@use_kernelized_func(apply_rotary_pos_emb)
class AttentionMixingProbeAttention(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,

        document_attention_mask: torch.Tensor | None = None,
        retrieval_decision: torch.Tensor | None = None,

        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states_doc, key_states_doc = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if document_attention_mask is not None:
            cos_no_doc, sin_no_doc = remove_document_from_rotary_embeddings(cos, sin, document_attention_mask)
            query_states_no_doc, key_states_no_doc = apply_rotary_pos_emb(query_states, key_states, cos_no_doc, sin_no_doc)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            
        attn_output, attn_weights = attention_mixing_forward(
            self,
            query_states_doc,
            key_states_doc,
            query_states_no_doc,
            key_states_no_doc,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            document_attention_mask=document_attention_mask,
            retrieval_decision=retrieval_decision,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


if __name__ == "__main__":
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(42)

    B = 1
    L = 4

    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )

    attention = AttentionMixingProbeAttention(
        config=config,
        layer_idx=0,
    )

    hidden_states = torch.randn(
        B,
        L,
        config.hidden_size,
    )

    document_mask = torch.tensor([
        [0, 0, 1, 1],
    ])

    document_attention_mask = (
        document_mask[:, None, None, :]
        .float()
        .masked_fill(
            document_mask[:, None, None, :].bool(),
            float("-inf"),
        )
    )

    retrieval_decision = torch.tensor([0.5])

    cos = torch.ones(
        B,
        L,
        attention.head_dim,
    )

    sin = torch.zeros(
        B,
        L,
        attention.head_dim,
    )

    output, attn_weights = attention(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        document_attention_mask=document_attention_mask,
        retrieval_decision=retrieval_decision,
    )

    print("Output shape :", output.shape)
    print("Attention shape:", attn_weights.shape)

    print("Smoke test passed.")