import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
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

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from .cache_utils import ARTCache
from .decoder_layer import AttentionMixingProbeDecoderLayer
from .utils import (
    ensure_document_attention_mask,
    create_position_ids_without_doc,
    align_document_attention_mask,
)


class AttentionMixingProbePreTrainedModel(Qwen2PreTrainedModel):
    _no_split_modules = [
        "AttentionMixingProbeDecoderLayer",
    ]

    # The custom attention-mixing decoder layer uses the eager/custom
    # attention path rather than HF's flash/sdpa/flex backends.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False


@auto_docstring
class AttentionMixingProbeModel(AttentionMixingProbePreTrainedModel):
    def __init__(self, config: Qwen2Config):
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

        self.norm = Qwen2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Newer Transformers Qwen2 configs may expose layer_types.
        # For ordinary Qwen2/Qwen2.5 configurations, all layers are full
        # attention, so fall back to full_attention if it is absent.
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * config.num_hidden_layers

        self.layer_types = layer_types
        self.has_sliding_layers = "sliding_attention" in layer_types

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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # ARTCache stores the cache state corresponding to the
        # with-document attention path.
        if use_cache:
            if past_key_values is None or not isinstance(
                past_key_values,
                ARTCache,
            ):
                past_key_values = ARTCache(
                    config=self.config,
                )

        if document_attention_mask is not None and input_ids is not None:
            document_attention_mask = ensure_document_attention_mask(
                document_attention_mask
            )

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.cache_with_doc.get_seq_length()
                if past_key_values is not None
                else 0
            )

            position_ids_with_doc = (
                torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
                + past_seen_tokens
            ).unsqueeze(0)
        else:
            position_ids_with_doc = position_ids

        if document_attention_mask is not None:
            position_ids_without_doc = create_position_ids_without_doc(
                position_ids_with_doc=position_ids_with_doc,
                document_attention_mask=document_attention_mask,
            )
        else:
            position_ids_without_doc = position_ids_with_doc

        # It may already have been prepared by e.g. generate().
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": (
                    past_key_values.cache_with_doc
                    if past_key_values is not None
                    else None
                ),
                "position_ids": position_ids_with_doc,
            }

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        # Align the user-provided document mask with the actual causal
        # attention mask produced by Transformers.
        if document_attention_mask is not None:
            causal_mask = causal_mask_mapping[
                self.layer_types[0]
            ]

            document_attention_mask = align_document_attention_mask(
                document_attention_mask=document_attention_mask,
                causal_mask=causal_mask,
            )

        hidden_states = inputs_embeds

        position_embeddings_with_doc = self.rotary_emb(
            hidden_states,
            position_ids_with_doc,
        )

        position_embeddings_without_doc = self.rotary_emb(
            hidden_states,
            position_ids_without_doc,
        )

        for i, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[
                    self.layer_types[i]
                ],
                position_embeddings_with_doc=position_embeddings_with_doc,
                position_embeddings_without_doc=position_embeddings_without_doc,
                position_ids_with_doc=position_ids_with_doc,
                position_ids_without_doc=position_ids_without_doc,
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
        "lm_head.weight": "model.embed_tokens.weight",
    }

    _tp_plan = {
        "lm_head": "colwise_gather_output",
    }

    _pp_plan = {
        "lm_head": (["hidden_states"], ["logits"]),
    }

    _fsdp_plan = {
        "lm_head": "keep_full_weight",
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
        # if we are not computing the loss.
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
