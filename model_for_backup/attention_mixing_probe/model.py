import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from transformers.utils.generic import (
    merge_with_config_defaults,
)

from transformers.utils.output_capturing import capture_outputs

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from .decoder_layer import AttentionMixingProbeDecoderLayer
from .utils import ensure_document_attention_mask

class AttentionMixingProbePreTrainedModel(
    Qwen3PreTrainedModel
):

    _no_split_modules = [
        "AttentionMixingProbeDecoderLayer"
    ]

    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False

@auto_docstring
class AttentionMixingProbeModel(AttentionMixingProbePreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                AttentionMixingProbeDecoderLayer(
                    config,
                    layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        self.has_sliding_layers = (
            "sliding_attention" in self.config.layer_types
        )

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        document_attention_mask: torch.Tensor | None = None,
        retrieval_decision: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # print("attn implementation:", self.config._attn_implementation)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if document_attention_mask is not None and input_ids is not None:
            document_attention_mask = ensure_document_attention_mask(
                input_ids=input_ids, 
                document_attention_mask=document_attention_mask,
            )

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )

            position_ids = (
                torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
                + past_seen_tokens
            )

            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):

            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            # The sliding window alternating layers are not always activated
            # depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )
        # print(causal_mask_mapping)
        # ============================================================
        # ALIGN DOCUMENT ATTENTION MASK WITH CAUSAL MASK
        # ============================================================

        if document_attention_mask is not None:

            causal_mask = causal_mask_mapping[
                self.config.layer_types[0]
            ]
            if causal_mask is not None:
                causal_length = causal_mask.shape[-1]
                document_length = document_attention_mask.shape[-1]

                if document_length < causal_length:

                    pad_length = causal_length - document_length

                    document_attention_mask = nn.functional.pad(
                        document_attention_mask,
                        (0, pad_length),
                        value=0.0,
                    )

                elif document_length > causal_length:

                    document_attention_mask = (
                        document_attention_mask[..., :causal_length]
                    )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
        )

        for i, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[
                    self.config.layer_types[i]
                ],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                document_attention_mask=document_attention_mask,
                retrieval_decision=retrieval_decision,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=(
                past_key_values
                if use_cache
                else None
            ),
        )


@auto_docstring
class AttentionMixingProbeForCausalLM(
    AttentionMixingProbePreTrainedModel,
    GenerationMixin,
):
    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight"
    }

    _tp_plan = {
        "lm_head": "colwise_gather_output"
    }

    _pp_plan = {
        "lm_head": (["hidden_states"], ["logits"])
    }

    _fsdp_plan = {
        "lm_head": "keep_full_weight"
    }

    def __init__(self, config):
        super().__init__(config)

        self.model = AttentionMixingProbeModel(config)

        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        document_attention_mask: torch.Tensor | None = None,
        retrieval_decision: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            document_attention_mask=document_attention_mask,
            retrieval_decision=retrieval_decision,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float
        # if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        logits = self.lm_head(
            hidden_states[:, slice_indices, :]
        )

        loss = None

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )