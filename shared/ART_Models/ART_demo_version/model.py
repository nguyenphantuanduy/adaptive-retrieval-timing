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
    create_document_attention_mask,
    update_attention_mask,
)

THRESHOLD = 0.5

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
        # if use_cache:
        #     if past_key_values is None or not isinstance(
        #         past_key_values,
        #         ARTCache,
        #     ):
        #         past_key_values = ARTCache(
        #             config=self.config,
        #         )

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


import torch
import torch.nn as nn

class RetrievalDecisionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.projection = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden_state = hidden_states[:, -1, :]  # (B, H)
        logits = self.projection(last_hidden_state)  # (B, 1)
        decision = torch.sigmoid(logits).squeeze(-1) # (B,)

        return decision

from .musique_retrieval_db import MuSiQueRetrievalDB


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

        self.retrieval_decision_head = RetrievalDecisionHead(
            hidden_size=config.hidden_size,
        )

        # =====================================================================
        # ART flow state
        # =====================================================================

        self.isTraining = False

        self.max_retrievals = 4
        self.retrieval_count = 0

        self.retriever = MuSiQueRetrievalDB(
            top_k=3,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def enable_training_mode(self):
        self.isTraining = True

    def disable_training_mode(self):
        self.isTraining = False

    # =========================================================================
    # Forward dispatcher
    # =========================================================================
    # =========================================================================
    # Forward dispatcher
    # =========================================================================

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,

        # =====================================================================
        # Standard forward inputs
        # =====================================================================

        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,

        # =====================================================================
        # ART-specific inputs
        # =====================================================================

        question: str | None = None,
        question_input_ids: torch.LongTensor | None = None,
        answer_input_ids: torch.LongTensor | None = None,

        document_attention_mask: torch.Tensor | None = None,
        retrieval_decision: torch.Tensor | None = None,

        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        # =====================================================================
        # TRAINING
        # =====================================================================

        if self.isTraining:

            return self.forward_training(
                question=question,
                question_input_ids=question_input_ids,
                answer_input_ids=answer_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        # =====================================================================
        # GENERATION
        # =====================================================================

        return self.forward_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            question=question,
            document_attention_mask=document_attention_mask,
            retrieval_decision=retrieval_decision,
            **kwargs,
        )
    
    # =========================================================================
    # ART training forward
    # =========================================================================

    def forward_training(
        self,
        question: str | None = None,
        question_input_ids: torch.LongTensor | None = None,
        answer_input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        # =====================================================================
        # 0. Basic validation
        # =====================================================================

        if question is None:
            raise ValueError(
                "question must be provided when training mode is enabled."
            )

        if question_input_ids is None:
            raise ValueError(
                "question_input_ids must be provided when training mode "
                "is enabled."
            )

        if answer_input_ids is None:
            raise ValueError(
                "answer_input_ids must be provided when training mode "
                "is enabled."
            )

        if attention_mask is None:
            raise ValueError(
                "attention_mask must be provided when training mode "
                "is enabled."
            )

        if labels is None:
            raise ValueError(
                "labels must be provided when training mode is enabled."
            )

        # =====================================================================
        # 1. Reset per-step retrieval statistics
        # =====================================================================

        self.retrieval_count = 0

        # =====================================================================
        # 2. Initial sequence length
        # =====================================================================

        current_sequence_length = question_input_ids.shape[1]

        # =====================================================================
        # 3. Create / validate ART cache
        # =====================================================================

        if past_key_values is None or not isinstance(past_key_values, ARTCache):

            past_key_values = ARTCache(
                question=question,
                config=self.config,
            )

        # =====================================================================
        # 4. Forward QUESTION
        #
        # Question is processed without document context.
        # =====================================================================
        attention_mask = attention_mask[:, :current_sequence_length]

        question_outputs: BaseModelOutputWithPast = self.model(
            input_ids=question_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            document_attention_mask=None,
            retrieval_decision=None,
            **kwargs,
        )

        question_hidden_states = (
            question_outputs.last_hidden_state
        )

        past_key_values = (
            question_outputs.past_key_values
        )

        # =====================================================================
        # 5. Question logits
        # =====================================================================

        question_logits = self.lm_head(
            question_hidden_states
        )

        all_logits = [
            question_logits
        ]

        # =====================================================================
        # 6. Decision after QUESTION
        # =====================================================================

        question_decision_score = (
            self.retrieval_decision_head(
                question_hidden_states
            )
        )

        # decision_score = question_decision_score
        decision_score = None

        # No active document context initially.
        document_attention_mask = None

        # =====================================================================
        # 7. Retrieval after QUESTION
        # =====================================================================

        if (
            question_decision_score.item() >= THRESHOLD
            and self.retrieval_count < self.max_retrievals
        ):

            # -----------------------------------------------------------------
            # 7.1 Retrieve using original question
            # -----------------------------------------------------------------

            retrieval_text = self.retriever.search(
                question
            )

            self.retrieval_count += 1

            # -----------------------------------------------------------------
            # 7.2 Tokenize retrieved document
            # -----------------------------------------------------------------

            document_inputs = self.retriever.tokenizer(
                retrieval_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            document_input_ids = (
                document_inputs["input_ids"]
                .to(question_input_ids.device)
            )

            document_length = (
                document_input_ids.shape[1]
            )

            # -----------------------------------------------------------------
            # 7.3 Create document attention mask
            # -----------------------------------------------------------------

            document_attention_mask = (
                create_document_attention_mask(
                    current_sequence_len=current_sequence_length,
                    document_len=document_length,
                ).to(question_input_ids.device)
            )

            decision_score = question_decision_score
            # -----------------------------------------------------------------
            # 7.4 Extend normal attention mask
            # -----------------------------------------------------------------

            attention_mask = update_attention_mask(
                attention_mask,
                num_tokens=document_length,
            )

            # -----------------------------------------------------------------
            # 7.5 Insert -100 labels for document tokens
            # -----------------------------------------------------------------

            ignore_labels = torch.full(
                (
                    labels.shape[0],
                    document_length,
                ),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )

            labels = torch.cat(
                [
                    labels[:, :current_sequence_length],
                    ignore_labels,
                    labels[:, current_sequence_length:],
                ],
                dim=1,
            )

            # -----------------------------------------------------------------
            # 7.6 Forward DOCUMENT
            # -----------------------------------------------------------------

            document_outputs: BaseModelOutputWithPast = self.model(
                input_ids=document_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                document_attention_mask=document_attention_mask,
                retrieval_decision=decision_score,
                **kwargs,
            )

            past_key_values = (
                document_outputs.past_key_values
            )

            # -----------------------------------------------------------------
            # 7.7 Document logits
            # -----------------------------------------------------------------

            document_hidden_states = (
                document_outputs.last_hidden_state
            )

            document_logits = self.lm_head(
                document_hidden_states
            )

            all_logits.append(
                document_logits
            )

            current_sequence_length += document_length

        # =====================================================================
        # 8. Forward ANSWER ONLINE, SENTENCE BY SENTENCE
        # =====================================================================

        sentence_buffer = []

        for token_id in answer_input_ids[0]:

            # -----------------------------------------------------------------
            # 8.1 Add current token to the local sentence buffer
            # -----------------------------------------------------------------

            sentence_buffer.append(token_id)

            # -----------------------------------------------------------------
            # 8.2 Feed token to the online sentence boundary detector
            # -----------------------------------------------------------------

            sentence_completed = (
                past_key_values.feed_token(token_id)
            )

            # Sentence is not complete yet.
            if not sentence_completed:
                continue

            # -----------------------------------------------------------------
            # 8.3 A sentence has just been completed
            # -----------------------------------------------------------------

            sentence_tokens = (
                torch.stack(sentence_buffer)
                .unsqueeze(0)
            )

            # Reset local buffer immediately.
            sentence_buffer = []

            sentence_length = (
                sentence_tokens.shape[1]
            )

            # -----------------------------------------------------------------
            # 8.4 Extend normal attention mask
            # -----------------------------------------------------------------

            attention_mask = update_attention_mask(
                attention_mask,
                num_tokens=sentence_length,
            )

            # -----------------------------------------------------------------
            # 8.5 Forward CURRENT sentence
            #
            # Uses the currently active document context.
            # -----------------------------------------------------------------

            sentence_outputs: BaseModelOutputWithPast = (
                self.model(
                    input_ids=sentence_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    document_attention_mask=document_attention_mask,
                    retrieval_decision=decision_score,
                    **kwargs,
                )
            )

            past_key_values = (
                sentence_outputs.past_key_values
            )

            current_sequence_length += sentence_length

            # -----------------------------------------------------------------
            # 8.6 Sentence logits
            # -----------------------------------------------------------------

            sentence_hidden_states = (
                sentence_outputs.last_hidden_state
            )

            sentence_logits = self.lm_head(
                sentence_hidden_states
            )

            all_logits.append(
                sentence_logits
            )

            # -----------------------------------------------------------------
            # 8.7 New retrieval decision after CURRENT sentence
            # -----------------------------------------------------------------

            new_decision_score = (
                self.retrieval_decision_head(
                    sentence_hidden_states
                )
            )

            # -----------------------------------------------------------------
            # 8.8 Retrieval after CURRENT sentence
            # -----------------------------------------------------------------

            if (
                new_decision_score.item() >= THRESHOLD
                and self.retrieval_count < self.max_retrievals
            ):

                # -------------------------------------------------------------
                # 8.8.1 Build retrieval query from CURRENT completed sentence
                # -------------------------------------------------------------

                retrieval_query = (
                    past_key_values.build_retrieval_query()
                )

                # -------------------------------------------------------------
                # 8.8.2 Retrieve
                # -------------------------------------------------------------

                retrieval_text = (
                    self.retriever.search(
                        retrieval_query
                    )
                )

                self.retrieval_count += 1

                # -------------------------------------------------------------
                # 8.8.3 Tokenize retrieved document
                # -------------------------------------------------------------

                document_inputs = (
                    self.retriever.tokenizer(
                        retrieval_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                )

                document_input_ids = (
                    document_inputs["input_ids"]
                    .to(question_input_ids.device)
                )

                document_length = (
                    document_input_ids.shape[1]
                )

                # -------------------------------------------------------------
                # 8.8.4 Create new document attention mask
                # -------------------------------------------------------------

                document_attention_mask = (
                    create_document_attention_mask(
                        current_sequence_len=current_sequence_length,
                        document_len=document_length,
                    ).to(question_input_ids.device)
                )

                # -------------------------------------------------------------
                # 8.8.5 Extend normal attention mask
                # -------------------------------------------------------------

                attention_mask = update_attention_mask(
                    attention_mask,
                    num_tokens=document_length,
                )

                # -------------------------------------------------------------
                # 8.8.6 Insert -100 labels for document tokens
                # -------------------------------------------------------------

                ignore_labels = torch.full(
                    (
                        labels.shape[0],
                        document_length,
                    ),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )

                labels = torch.cat(
                    [
                        labels[:, :current_sequence_length],
                        ignore_labels,
                        labels[:, current_sequence_length:],
                    ],
                    dim=1,
                )

                # -------------------------------------------------------------
                # 8.8.7 Forward NEW DOCUMENT
                # -------------------------------------------------------------

                document_outputs: BaseModelOutputWithPast = (
                    self.model(
                        input_ids=document_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        document_attention_mask=document_attention_mask,
                        retrieval_decision=new_decision_score,
                        **kwargs,
                    )
                )

                past_key_values = (
                    document_outputs.past_key_values
                )

                # -------------------------------------------------------------
                # 8.8.8 Document logits
                # -------------------------------------------------------------

                document_hidden_states = (
                    document_outputs.last_hidden_state
                )

                document_logits = (
                    self.lm_head(
                        document_hidden_states
                    )
                )

                all_logits.append(
                    document_logits
                )

                current_sequence_length += document_length

                # -------------------------------------------------------------
                # 8.8.9 NEW document becomes active
                # -------------------------------------------------------------

                decision_score = (
                    new_decision_score
                )

            # -----------------------------------------------------------------
            # If no retrieval:
            #
            #   document_attention_mask stays unchanged.
            #   decision_score stays unchanged.
            #
            # Therefore the next sentence continues using the same
            # active document context.
            # -----------------------------------------------------------------


        # =====================================================================
        # 9. Remaining incomplete sentence
        #
        # If the answer does not end with a detected sentence boundary,
        # process the remaining tokens as the final sentence.
        # =====================================================================

        if sentence_buffer:

            sentence_tokens = (
                torch.stack(sentence_buffer)
                .unsqueeze(0)
            )

            sentence_length = (
                sentence_tokens.shape[1]
            )

            # -----------------------------------------------------------------
            # 9.1 Extend normal attention mask
            # -----------------------------------------------------------------

            attention_mask = update_attention_mask(
                attention_mask,
                num_tokens=sentence_length,
            )

            # -----------------------------------------------------------------
            # 9.2 Forward final incomplete sentence
            # -----------------------------------------------------------------

            sentence_outputs: BaseModelOutputWithPast = (
                self.model(
                    input_ids=sentence_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    document_attention_mask=document_attention_mask,
                    retrieval_decision=decision_score,
                    **kwargs,
                )
            )

            past_key_values = (
                sentence_outputs.past_key_values
            )

            current_sequence_length += sentence_length

            # -----------------------------------------------------------------
            # 9.3 Final sentence logits
            # -----------------------------------------------------------------

            sentence_hidden_states = (
                sentence_outputs.last_hidden_state
            )

            sentence_logits = (
                self.lm_head(
                    sentence_hidden_states
                )
            )

            all_logits.append(
                sentence_logits
            )

            # -----------------------------------------------------------------
            # 9.4 Decision after final sentence
            # -----------------------------------------------------------------

            new_decision_score = (
                self.retrieval_decision_head(
                    sentence_hidden_states
                )
            )

            # -----------------------------------------------------------------
            # 9.5 No retrieval after final sentence
            #
            # This is intentionally the final answer segment, so there is
            # no subsequent sentence that would consume a newly retrieved
            # document.
            # -----------------------------------------------------------------

        # =====================================================================
        # 10. Concatenate logits
        # =====================================================================

        logits = torch.cat(
            all_logits,
            dim=1,
        )

        # =====================================================================
        # 11. Compute loss
        # =====================================================================

        loss = None

        if labels is not None:

            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # =====================================================================
        # 12. Return
        # =====================================================================

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def forward_generation(
        self,
        input_ids: torch.LongTensor | None = None,

        # =====================================================================
        # Standard forward inputs
        # =====================================================================

        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,

        # =====================================================================
        # ART-specific inputs
        # =====================================================================

        question: str | None = None,

        document_attention_mask: torch.Tensor | None = None,
        retrieval_decision: torch.Tensor | None = None,

        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        # =====================================================================
        # 0. Validation
        # =====================================================================

        if input_ids is None:
            raise ValueError(
                "input_ids must be provided during generation."
            )

        if question is None:
            raise ValueError(
                "question must be provided during generation."
            )

        if attention_mask is None:
            raise ValueError(
                "attention_mask must be provided during generation."
            )

        # =====================================================================
        # 1. Create / validate ART cache
        # =====================================================================

        first_token_generation = False

        if (
            past_key_values is None
            or not isinstance(past_key_values, ARTCache)
        ):
            past_key_values = ARTCache(
                question=question,
                config=self.config,
            )

            first_token_generation = True

            # Reset retrieval quota for a new generation.
            self.retrieval_count = 0

        # =====================================================================
        # 2. Recover persistent ART retrieval state from cache
        # =====================================================================

        if document_attention_mask is None:
            document_attention_mask = (
                past_key_values.document_attention_mask
            )

        if retrieval_decision is None:
            retrieval_decision = (
                past_key_values.retrieval_decision
            )

        # =====================================================================
        # 3. Current sequence length
        #
        # attention_mask represents the sequence before adding the current
        # input tokens.
        #
        #   question + documents + generated tokens
        # =====================================================================

        current_sequence_length = attention_mask.shape[1]

        # =====================================================================
        # 4. Forward current input
        # =====================================================================

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            document_attention_mask=document_attention_mask,
            retrieval_decision=retrieval_decision,
            **kwargs,
        )

        past_key_values = outputs.past_key_values

        hidden_states = outputs.last_hidden_state

        # # Current input is now part of the sequence.
        # current_sequence_length += input_ids.shape[1]

        # =====================================================================
        # 5. Determine whether to make a retrieval decision
        #
        # First generation token:
        #     always make a decision.
        #
        # Later tokens:
        #     make a decision only when a sentence has just completed.
        # =====================================================================

        sentence_completed = False

        if not first_token_generation:

            for token_id in input_ids[0]:

                sentence_completed = (
                    past_key_values.feed_token(token_id)
                )

        should_decide = (
            first_token_generation
            or sentence_completed
        )

        # =====================================================================
        # 6. Retrieval decision
        # =====================================================================

        if should_decide:

            # -----------------------------------------------------------------
            # 6.1 Compute decision score
            # -----------------------------------------------------------------

            new_decision_score = (
                self.retrieval_decision_head(
                    hidden_states
                )
            )

            # -----------------------------------------------------------------
            # 6.2 Check retrieval condition
            # -----------------------------------------------------------------

            if (
                new_decision_score.item() >= THRESHOLD
                and self.retrieval_count < self.max_retrievals
            ):

                # -------------------------------------------------------------
                # 6.2.1 Build retrieval query
                # -------------------------------------------------------------

                if first_token_generation:

                    retrieval_query = question

                else:

                    retrieval_query = (
                        past_key_values.build_retrieval_query()
                    )

                # -------------------------------------------------------------
                # 6.2.2 Retrieve
                # -------------------------------------------------------------

                retrieval_text = self.retriever.search(
                    retrieval_query
                )

                self.retrieval_count += 1

                # -------------------------------------------------------------
                # 6.2.3 Tokenize retrieved document
                # -------------------------------------------------------------

                document_inputs = self.retriever.tokenizer(
                    retrieval_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

                document_input_ids = (
                    document_inputs["input_ids"]
                    .to(input_ids.device)
                )

                document_length = (
                    document_input_ids.shape[1]
                )

                # -------------------------------------------------------------
                # 6.2.4 Create document attention mask
                # -------------------------------------------------------------

                new_document_attention_mask = (
                    create_document_attention_mask(
                        current_sequence_len=current_sequence_length,
                        document_len=document_length,
                    ).to(input_ids.device)
                )

                # -------------------------------------------------------------
                # 6.2.5 Extend normal attention mask
                # -------------------------------------------------------------

                attention_mask = update_attention_mask(
                    attention_mask,
                    num_tokens=document_length,
                )

                # -------------------------------------------------------------
                # 6.2.6 Forward retrieved document
                #
                # The newly retrieved document becomes available to
                # subsequent generated tokens.
                # -------------------------------------------------------------

                document_outputs: BaseModelOutputWithPast = (
                    self.model(
                        input_ids=document_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        document_attention_mask=(
                            new_document_attention_mask
                        ),
                        retrieval_decision=new_decision_score,
                        **kwargs,
                    )
                )

                past_key_values = (
                    document_outputs.past_key_values
                )

                # -------------------------------------------------------------
                # 6.2.7 Update sequence length
                # -------------------------------------------------------------

                current_sequence_length += document_length

                # -------------------------------------------------------------
                # 6.2.8 Store new document context in ART cache
                # -------------------------------------------------------------

                past_key_values.document_attention_mask = (
                    new_document_attention_mask
                )

                past_key_values.retrieval_decision = (
                    new_decision_score
                )

                # Update local references as well.
                document_attention_mask = (
                    new_document_attention_mask
                )

                retrieval_decision = (
                    new_decision_score
                )

        # =====================================================================
        # 7. Logits
        # =====================================================================

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        logits = self.lm_head(
            hidden_states[:, slice_indices, :]
        )

        # =====================================================================
        # 8. Return
        # =====================================================================

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )