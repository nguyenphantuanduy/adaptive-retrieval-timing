import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from typing import Any
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

THRESHOLD = 0.4297
RETRIEVAL_COST = 0.1
# THRESHOLD = 0.5
# RETRIEVAL_COST = 0.05
MAX_RETRIEVAL_CONTEXT_TOKENS = 600
MAX_RETRIEVAL_COUNT = 4
MAX_EXPLORATION_PROBABILITY = 0.5
MIN_EXPLORATION_PROBABILITY = 0.1
INIT_EXPLORATION_PROBABILITY = 0.3

import random

SEED = 42
exploration_rng = random.Random(SEED)


def flip_coin(probability: float = 0.1) -> bool:
    return exploration_rng.random() < probability

def increase_probability(probability: float) -> float:
    return min(probability + 0.05, MAX_EXPLORATION_PROBABILITY)

def decrease_probability(
    probability: float,
) -> float:
    return MIN_EXPLORATION_PROBABILITY


# ============================================================
# Exponential saturation retrieval cost
# ============================================================

# RETRIEVAL_COST_L = -3.697402
# RETRIEVAL_COST_U = 0.278405
# RETRIEVAL_COST_K = 4.572680
# RETRIEVAL_COST_D0 = 0.545356


# def compute_retrieval_cost(
#         total_decision_score: torch.Tensor,
#     ) -> torch.Tensor:

#         d = total_decision_score

#         # ====================================================
#         # Antiderivative G(d)
#         #
#         # g(d) = L + (U - L) * (1 - exp(-k*d))
#         #       = U - (U - L) * exp(-k*d)
#         #
#         # G'(d) = g(d)
#         # ====================================================

#         exp_d = torch.exp(
#             -RETRIEVAL_COST_K * d
#         )

#         exp_d0 = torch.exp(
#             torch.tensor(
#                 -RETRIEVAL_COST_K
#                 * RETRIEVAL_COST_D0,
#                 device=d.device,
#                 dtype=d.dtype,
#             )
#         )

#         G_d = (
#             RETRIEVAL_COST_U
#             * (
#                 d
#                 - RETRIEVAL_COST_D0
#             )
#             + (
#                 (
#                     RETRIEVAL_COST_U
#                     - RETRIEVAL_COST_L
#                 )
#                 / RETRIEVAL_COST_K
#             )
#             * (
#                 exp_d
#                 - exp_d0
#             )
#         )

#         # ====================================================
#         # Piecewise retrieval cost
#         #
#         # d < D0  -> -G(d)
#         # d >= D0 -> 0.05 * d
#         # ====================================================

#         retrieval_cost = torch.where(
#             d < RETRIEVAL_COST_D0,
#             -G_d,
#             0.05 * d,
#         )

#         return retrieval_cost.sum()

def compute_retrieval_cost(
        total_decision_score: torch.Tensor,
    ) -> torch.Tensor:

    return RETRIEVAL_COST * total_decision_score.sum()

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

        self.network = nn.Sequential(
            nn.Linear(hidden_size, 1),  # 2048 → 1
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden_state = hidden_states[:, -1, :]   # (B, 2048)

        logits = self.network(last_hidden_state)      # (B, 1)

        decision = torch.sigmoid(logits).squeeze(-1)  # (B,)

        return decision
    

from .musique_retrieval_db import MuSiQueRetrievalDB


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

        self.max_retrievals = MAX_RETRIEVAL_COUNT
        self.retrieval_count = 0

        self.retriever = None
        self.tokenizer = None

        self.retrieval_count_for_logging = []
        self.query = []
        self.doc = []
        self.retrieve_segment_indices = []
        self.first_segment_retrieval_decision_score = None

        # Initialize weights and apply final processing
        self.post_init()

    def enable_training_mode(self):
        self.isTraining = True

    def disable_training_mode(self):
        self.isTraining = False

    def _get_retriever(self):
        if self.retriever is None:
            self.retriever = MuSiQueRetrievalDB(
                top_k=3,
            )
        return self.retriever

    def _get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B-Instruct",
                trust_remote_code=True,
            )
        return self.tokenizer

    def _append_trajectory_step(
        self,
        trajectory: list[dict],
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_attention_mask: torch.Tensor | None = None,
        retrieval_decision: torch.Tensor | None = None,
    ) -> None:
        trajectory.append(
            {
                "input_ids": (
                    input_ids.clone()
                    if input_ids is not None
                    else None
                ),
                "attention_mask": (
                    attention_mask.clone()
                    if attention_mask is not None
                    else None
                ),
                "document_attention_mask": (
                    document_attention_mask.clone()
                    if document_attention_mask is not None
                    else None
                ),
                "retrieval_decision": (
                    retrieval_decision.clone()
                    if retrieval_decision is not None
                    else None
                ),
            }
        )

    # =========================================================================
    # Forward dispatcher
    # =========================================================================

    @can_return_tuple
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
        retrieval_topk: int = 3,
        max_retrieval_context_tokens: int = MAX_RETRIEVAL_CONTEXT_TOKENS,
        question: str | None = None,
        # question_input_ids: torch.LongTensor | None = None,
        # answer_input_ids: torch.LongTensor | None = None,
        collect_outputs: dict[str, Any] | None = None,

        document_attention_mask: torch.Tensor | None = None,
        retrieval_decision: torch.Tensor | None = None,

        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        # =====================================================================
        # TRAINING
        # =====================================================================

        if self.isTraining:

            # return self.forward_training(
            #     question=question,
            #     question_input_ids=question_input_ids,
            #     answer_input_ids=answer_input_ids,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     inputs_embeds=inputs_embeds,
            #     labels=labels,
            #     logits_to_keep=logits_to_keep,
            #     **kwargs,
            # )

            return self.forward_training(
                collect_outputs=collect_outputs,
            )

        # =====================================================================
        # GENERATION
        # =====================================================================

        return self.forward_generate(
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
            retrieval_topk=retrieval_topk,
            max_retrieval_context_tokens=max_retrieval_context_tokens,
            **kwargs,
        )
    
    # =========================================================================
    # ART training forward
    # =========================================================================

    def forward_collect(
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
    ) -> dict:

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
        input_ids_buffer = None
        trajectory = []
        probability = INIT_EXPLORATION_PROBABILITY
        self.retrieve_segment_indices = []
        self.first_segment_retrieval_decision_score = None

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
                tokenizer=self._get_tokenizer(),
            )

        # =====================================================================
        # 4. Forward QUESTION
        #
        # Question is processed without document context.
        # =====================================================================
        attention_mask = attention_mask[:, :current_sequence_length]
        input_ids_buffer = question_input_ids

        question_outputs: BaseModelOutputWithPast = self.model(
            input_ids=question_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            document_attention_mask=None,
            retrieval_decision=None,
        )

        question_hidden_states = (
            question_outputs.last_hidden_state
        )

        past_key_values = (
            question_outputs.past_key_values
        )


        # =====================================================================
        # 5. Decision after QUESTION
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
        # 6. Retrieval after QUESTION
        # =====================================================================

        if (
            ( question_decision_score.item() >= THRESHOLD )
            and self.retrieval_count < self.max_retrievals
        ):
            self.first_segment_retrieval_decision_score = question_decision_score.item()
            self.retrieve_segment_indices.append(past_key_values.counter)
            probability = decrease_probability(probability)
            
            self._append_trajectory_step(
                trajectory=trajectory,
                input_ids=input_ids_buffer,
                attention_mask=attention_mask,
                document_attention_mask=document_attention_mask,
                retrieval_decision=decision_score,
            )

            # -----------------------------------------------------------------
            # 6.1 Retrieve using original question
            # -----------------------------------------------------------------

            
            retrieval_text = self._get_retriever().search(
                query=question,
                tokenizer=self._get_tokenizer(),
                max_passage_tokens=MAX_RETRIEVAL_CONTEXT_TOKENS,
            )

            self.retrieval_count += 1

            # -----------------------------------------------------------------
            # 6.2 Tokenize retrieved document
            # -----------------------------------------------------------------

            document_inputs = self._get_tokenizer()(
                retrieval_text,
                return_tensors="pt",
            )

            document_input_ids = (
                document_inputs["input_ids"]
                .to(question_input_ids.device)
            )

            document_length = (
                document_input_ids.shape[1]
            )

            # -----------------------------------------------------------------
            # 6.3 Create document attention mask
            # -----------------------------------------------------------------

            new_document_attention_mask = (
                create_document_attention_mask(
                    current_sequence_len=current_sequence_length,
                    document_len=document_length,
                ).to(question_input_ids.device)
            )

            new_decision_score = question_decision_score

            # -----------------------------------------------------------------
            # 6.4 Extend normal attention mask
            # -----------------------------------------------------------------

            attention_mask = update_attention_mask(
                attention_mask,
                num_tokens=document_length,
            )

            # -----------------------------------------------------------------
            # 6.5 Insert -100 labels for document tokens
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
            # 6.6 Forward DOCUMENT
            # -----------------------------------------------------------------
            input_ids_buffer = torch.cat(
                [
                    input_ids_buffer,
                    document_input_ids,
                ],
                dim=1,
            )

            document_outputs: BaseModelOutputWithPast = self.model(
                input_ids=document_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                document_attention_mask=document_attention_mask,
                retrieval_decision=decision_score,
            )

            past_key_values = (
                document_outputs.past_key_values
            )

            current_sequence_length += document_length
            document_attention_mask = new_document_attention_mask
            decision_score = new_decision_score
            past_key_values.synchronize_cache()

        else:
            probability = increase_probability(probability)

        # =====================================================================
        # 7. Forward ANSWER ONLINE, SENTENCE BY SENTENCE
        # =====================================================================

        sentence_buffer = []

        for token_id in answer_input_ids[0]:

            # -----------------------------------------------------------------
            # 7.1 Add current token to the local sentence buffer
            # -----------------------------------------------------------------

            sentence_buffer.append(token_id)

            # -----------------------------------------------------------------
            # 7.2 Feed token to the online sentence boundary detector
            # -----------------------------------------------------------------

            sentence_completed = (
                past_key_values.feed_token(token_id)
            )

            # Sentence is not complete yet.
            if not sentence_completed:
                continue

            # -----------------------------------------------------------------
            # 7.3 A sentence has just been completed
            # -----------------------------------------------------------------
            past_key_values.counter += 1

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
            # 7.4 Extend normal attention mask
            # -----------------------------------------------------------------

            attention_mask = update_attention_mask(
                attention_mask,
                num_tokens=sentence_length,
            )

            # -----------------------------------------------------------------
            # 7.5 Forward CURRENT sentence
            #
            # Uses the currently active document context.
            # -----------------------------------------------------------------
            input_ids_buffer = torch.cat(
                [
                    input_ids_buffer,
                    sentence_tokens,
                ],
                dim=1,
            )

            sentence_outputs: BaseModelOutputWithPast = (
                self.model(
                    input_ids=sentence_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    document_attention_mask=document_attention_mask,
                    retrieval_decision=decision_score,
                )
            )

            past_key_values = (
                sentence_outputs.past_key_values
            )

            current_sequence_length += sentence_length

            # -----------------------------------------------------------------
            # 7.6 Sentence hidden states
            # -----------------------------------------------------------------

            sentence_hidden_states = (
                sentence_outputs.last_hidden_state
            )

            # -----------------------------------------------------------------
            # 7.7 New retrieval decision after CURRENT sentence
            # -----------------------------------------------------------------

            new_decision_score = (
                self.retrieval_decision_head(
                    sentence_hidden_states
                )
            )

            # retrieval_scores.append(
            #     new_decision_score
            # )

            # -----------------------------------------------------------------
            # 7.8 Retrieval after CURRENT sentence
            # -----------------------------------------------------------------

            if (
                ( new_decision_score.item() >= THRESHOLD or flip_coin(probability) )
                and self.retrieval_count < self.max_retrievals
            ):
                if self.first_segment_retrieval_decision_score is None:
                    self.first_segment_retrieval_decision_score = new_decision_score.item()
                self.retrieve_segment_indices.append(past_key_values.counter)
                probability = decrease_probability(probability)

                self._append_trajectory_step(
                    trajectory=trajectory,
                    input_ids=input_ids_buffer,
                    attention_mask=attention_mask,
                    document_attention_mask=document_attention_mask,
                    retrieval_decision=decision_score,
                )

                # -------------------------------------------------------------
                # 7.8.1 Build retrieval query from CURRENT completed sentence
                # -------------------------------------------------------------

                retrieval_query = (
                    past_key_values.build_retrieval_query()
                )

                # -------------------------------------------------------------
                # 7.8.2 Retrieve
                # -------------------------------------------------------------

                retrieval_text = (
                    self._get_retriever().search(
                        query=retrieval_query,
                        tokenizer=self._get_tokenizer(),
                        max_passage_tokens=MAX_RETRIEVAL_CONTEXT_TOKENS,
                    )
                )

                self.retrieval_count += 1

                # -------------------------------------------------------------
                # 7.8.3 Tokenize retrieved document
                # -------------------------------------------------------------

                document_inputs = (
                    self._get_tokenizer()(
                        retrieval_text,
                        return_tensors="pt",
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
                # 7.8.4 Create new document attention mask
                # -------------------------------------------------------------

                new_document_attention_mask = (
                    create_document_attention_mask(
                        current_sequence_len=current_sequence_length,
                        document_len=document_length,
                    ).to(question_input_ids.device)
                )

                # -------------------------------------------------------------
                # 7.8.5 Extend normal attention mask
                # -------------------------------------------------------------

                attention_mask = update_attention_mask(
                    attention_mask,
                    num_tokens=document_length,
                )

                # -------------------------------------------------------------
                # 7.8.6 Insert -100 labels for document tokens
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
                # 7.8.7 Forward NEW DOCUMENT
                # -------------------------------------------------------------
                input_ids_buffer = torch.cat(
                    [
                        input_ids_buffer,
                        document_input_ids,
                    ],
                    dim=1,
                )

                document_outputs: BaseModelOutputWithPast = (
                    self.model(
                        input_ids=document_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        document_attention_mask=document_attention_mask,
                        retrieval_decision=decision_score,
                    )
                )

                past_key_values = (
                    document_outputs.past_key_values
                )

                current_sequence_length += document_length

                # -------------------------------------------------------------
                # 7.8.8 NEW document becomes active
                # -------------------------------------------------------------

                decision_score = (
                    new_decision_score
                )

                document_attention_mask = new_document_attention_mask
                past_key_values.synchronize_cache()

            else:
                probability = increase_probability(probability)

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
        # 8. Remaining incomplete sentence
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
            # 8.1 Extend normal attention mask
            # -----------------------------------------------------------------

            attention_mask = update_attention_mask(
                attention_mask,
                num_tokens=sentence_length,
            )

            # -----------------------------------------------------------------
            # 8.2 Forward final incomplete sentence
            # -----------------------------------------------------------------

            input_ids_buffer = torch.cat(
                [
                    input_ids_buffer,
                    sentence_tokens,
                ],
                dim=1,
            )
            
            sentence_outputs: BaseModelOutputWithPast = (
                self.model(
                    input_ids=sentence_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    document_attention_mask=document_attention_mask,
                    retrieval_decision=decision_score,
                )
            )

            past_key_values = (
                sentence_outputs.past_key_values
            )

            sentence_hidden_states = (
                sentence_outputs.last_hidden_state
            )

            current_sequence_length += sentence_length
            self._append_trajectory_step(
                trajectory=trajectory,
                input_ids=input_ids_buffer,
                attention_mask=attention_mask,
                document_attention_mask=document_attention_mask,
                retrieval_decision=decision_score,
            )

            # -----------------------------------------------------------------
            # 8.3 Decision after final sentence
            # -----------------------------------------------------------------

            new_decision_score = (
                self.retrieval_decision_head(
                    sentence_hidden_states
                )
            )

            if (
                ( new_decision_score.item() >= THRESHOLD )
                and self.retrieval_count < self.max_retrievals
            ):
                self._append_trajectory_step(
                    trajectory=trajectory,
                    retrieval_decision=decision_score,
                )

        self.retrieval_count_for_logging.append(
            self.retrieval_count
        )
        
        return {
            "labels": labels,
            "trajectory": trajectory,
        }

    def forward_generate(
        self,
        input_ids: torch.LongTensor | None = None,
        retrieval_topk: int = 3,
        max_retrieval_context_tokens: int = MAX_RETRIEVAL_CONTEXT_TOKENS,

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
                tokenizer=self._get_tokenizer(),
                config=self.config,
            )

            first_token_generation = True

            # Reset retrieval quota for a new generation.
            self.retrieval_count = 0
            self.query = []
            self.doc = []

        else:
            attention_mask = past_key_values.attention_mask

            attention_mask = update_attention_mask(
                attention_mask,
                num_tokens=1,
            )

        # =====================================================================
        # 2. Recover persistent ART retrieval state from cache
        # =====================================================================

        document_attention_mask = (
            past_key_values.document_attention_mask
        )

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
            past_key_values=past_key_values,
            use_cache=True,
            document_attention_mask=document_attention_mask,
            retrieval_decision=retrieval_decision,
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

        if sentence_completed:
            past_key_values.counter = past_key_values.counter + 1

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
                ( new_decision_score.item() >= THRESHOLD )
                and self.retrieval_count < self.max_retrievals
            ):

                # -------------------------------------------------------------
                # 6.2.1 Build retrieval query
                # -------------------------------------------------------------

                retrieval_query = (
                    past_key_values.build_retrieval_query()
                )
                self.query.append(retrieval_query)
                # -------------------------------------------------------------
                # 6.2.2 Retrieve
                # -------------------------------------------------------------

                retrieval_text, passages = self._get_retriever().search(
                    query=retrieval_query,
                    max_passage_tokens=max_retrieval_context_tokens,
                    tokenizer=self._get_tokenizer(),
                    top_k=retrieval_topk,
                    return_passages=True,
                )
                self.doc.extend(passages)

                self.retrieval_count += 1

                # -------------------------------------------------------------
                # 6.2.3 Tokenize retrieved document
                # -------------------------------------------------------------

                document_inputs = self._get_tokenizer()(
                    retrieval_text,
                    return_tensors="pt",
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
                            document_attention_mask
                        ),
                        retrieval_decision=retrieval_decision,
                    )
                )

                past_key_values = (
                    document_outputs.past_key_values
                )

                document_hidden_states = (
                    document_outputs.last_hidden_state
                )

                hidden_states = torch.cat(
                    [
                        hidden_states,
                        document_hidden_states,
                    ],
                    dim=1,
                )

                # -------------------------------------------------------------
                # 6.2.7 Update sequence length
                # -------------------------------------------------------------

                current_sequence_length += document_length

                # -------------------------------------------------------------
                # 6.2.8 Store new document context in ART cache
                # -------------------------------------------------------------

                past_key_values.update_retrieval_state(
                    document_attention_mask=new_document_attention_mask,
                    retrieval_decision=new_decision_score,
                )

                # # Update local references as well.
                # document_attention_mask = (
                #     new_document_attention_mask
                # )

                # retrieval_decision = (
                #     new_decision_score
                # )

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

        past_key_values.attention_mask = attention_mask

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def forward_training(
    #     self,
    #     question: str | None = None,
    #     question_input_ids: torch.LongTensor | None = None,
    #     answer_input_ids: torch.LongTensor | None = None,
    #     attention_mask: torch.Tensor | None = None,
    #     position_ids: torch.LongTensor | None = None,
    #     past_key_values: Cache | None = None,
    #     inputs_embeds: torch.FloatTensor | None = None,
    #     labels: torch.LongTensor | None = None,
    #     logits_to_keep: int | torch.Tensor = 0,
    #     **kwargs: Unpack[TransformersKwargs],
    # ) -> CausalLMOutputWithPast:

    #     # ============================================================
    #     # 1. Collect trajectory
    #     # ============================================================

    #     self.gradient_checkpointing_disable()

    #     with torch.no_grad():

    #         collect_outputs = self.forward_collect(
    #             question=question,
    #             question_input_ids=question_input_ids,
    #             answer_input_ids=answer_input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             labels=labels,
    #             logits_to_keep=logits_to_keep,
    #             **kwargs,
    #         )

    #     trajectory = collect_outputs["trajectory"]
    #     labels = collect_outputs["labels"]
    #     self.retrieval_count_for_logging.append(
    #         self.retrieval_count
    #     )

    #     # ============================================================
    #     # 2. Enable gradient checkpointing
    #     # ============================================================

    #     self.gradient_checkpointing_enable()

    #     # ============================================================
    #     # 3. Replay trajectory with gradient
    #     # ============================================================

    #     logits_buffer = []

    #         # Number of tokens for which logits have already been collected.
    #     computed_logits_length = 0

    #     # Retrieval decision from the previous trajectory step.
    #     decision_score = None

    #     # Sum of all retrieval decision scores.
    #     total_decision_score = None

    #     for item in trajectory:

    #         # --------------------------------------------------------
    #         # 3.1 Accumulate previous decision score
    #         # --------------------------------------------------------

    #         if decision_score is not None:

    #             if total_decision_score is None:
    #                 total_decision_score = decision_score
    #             else:
    #                 total_decision_score = (
    #                     total_decision_score
    #                     + decision_score
    #                 )

    #         # --------------------------------------------------------
    #         # 3.2 Get trajectory state
    #         # --------------------------------------------------------

    #         input_ids = item["input_ids"]
    #         attention_mask = item["attention_mask"]
    #         document_attention_mask = (
    #             item["document_attention_mask"]
    #         )

    #         # --------------------------------------------------------
    #         # 3.3 Skip decision-only trajectory state
    #         # --------------------------------------------------------

    #         if (
    #             input_ids is None
    #             or attention_mask is None
    #         ):
    #             continue

    #         # --------------------------------------------------------
    #         # 3.4 Forward model
    #         # --------------------------------------------------------
    #         outputs: BaseModelOutputWithPast = self.model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             inputs_embeds=inputs_embeds,
    #             past_key_values=None,
    #             use_cache=False,
    #             document_attention_mask=(
    #                 document_attention_mask
    #             ),
    #             retrieval_decision=decision_score,
    #             **kwargs,
    #         )

    #         hidden_states = outputs.last_hidden_state

    #         # --------------------------------------------------------
    #         # 3.5 Get logits for newly computed tokens only
    #         # --------------------------------------------------------

    #         current_sequence_length = (
    #             hidden_states.shape[1]
    #         )

    #         new_hidden_states = hidden_states[
    #             :,
    #             computed_logits_length:current_sequence_length,
    #             :,
    #         ]

    #         new_logits = self.lm_head(
    #             new_hidden_states
    #         )

    #         logits_buffer.append(new_logits)

    #         computed_logits_length = (
    #             current_sequence_length
    #         )

    #         # --------------------------------------------------------
    #         # 3.6 Compute new retrieval decision
    #         # --------------------------------------------------------

    #         decision_score = (
    #             self.retrieval_decision_head(
    #             hidden_states
    #             )
    #         )

    #     # ============================================================
    #     # 4. Accumulate final decision score
    #     #
    #     # If the last trajectory item is decision-only,
    #     # its decision score has not been accumulated by
    #     # the loop body, so add it here.
    #     # ============================================================

    #     if trajectory:

    #         last_item = trajectory[-1]

    #         if (
    #             last_item["input_ids"] is None
    #             or last_item["attention_mask"] is None
    #             or last_item["document_attention_mask"] is None
    #         ):

    #             if decision_score is not None:

    #                 if total_decision_score is None:
    #                     total_decision_score = decision_score
    #                 else:
    #                     total_decision_score = (
    #                         total_decision_score
    #                         + decision_score
    #                     )

    #     # ============================================================
    #     # 5. Combine logits
    #     # ============================================================

    #     if not logits_buffer:
    #         raise ValueError(
    #             "Trajectory does not contain any valid "
    #             "states for training."
    #         )

    #     logits = torch.cat(
    #         logits_buffer,
    #         dim=1,
    #     )

    #     assert logits.shape[1] == labels.shape[1], (
    #         "Logits and labels must have the same sequence length: "
    #         f"logits.shape={logits.shape}, "
    #         f"labels.shape={labels.shape}"
    #     )


    #     # ============================================================
    #     # 6. Handle empty retrieval cost
    #     # ============================================================

    #     if total_decision_score is None:
    #         total_decision_score = logits.new_zeros(())

    #     # ============================================================
    #     # 7. Compute language modeling loss
    #     # ============================================================

    #     lm_loss = self.loss_function(
    #         logits=logits,
    #         labels=labels,
    #         vocab_size=self.config.vocab_size,
    #         **kwargs,
    #     )

    #     # ============================================================
    #     # 8. Add retrieval cost
    #     # ============================================================

    #     loss = (
    #         lm_loss
    #         + RETRIEVAL_COST * total_decision_score.sum()
    #     )

    #     # ============================================================
    #     # 9. Return
    #     # ============================================================

    #     return CausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=None,
    #         hidden_states=None,
    #         attentions=None,
    #     )
    def forward_training(
            self,
            collect_outputs,
            **kwargs: Unpack[TransformersKwargs],
        ) -> CausalLMOutputWithPast:
    
            # ============================================================
            # 1. Get trajectory data
            # ============================================================
    
            trajectory_items = collect_outputs["trajectory"]
            labels = collect_outputs["labels"]
    
            # ============================================================
            # 2. Replay trajectory with gradient
            # ============================================================
    
            logits_buffer = []
    
            # Number of tokens for which logits have already been collected.
            computed_logits_length = 0
    
            # Retrieval decision from the previous trajectory step.
            decision_score = None
    
            # Sum of all retrieval decision scores.
            total_decision_score = None
    
            for item in trajectory_items:
    
                # --------------------------------------------------------
                # 2.1 Accumulate previous decision score
                # --------------------------------------------------------
    
                if decision_score is not None:
    
                    if total_decision_score is None:
                        total_decision_score = decision_score
                    else:
                        total_decision_score = torch.cat(
                            [
                                total_decision_score,
                                decision_score,
                            ],
                            dim=0,
                        )
    
                # --------------------------------------------------------
                # 2.2 Get trajectory state
                # --------------------------------------------------------
    
                input_ids = item["input_ids"]
                attention_mask = item["attention_mask"]
                document_attention_mask = (
                    item["document_attention_mask"]
                )
    
                # --------------------------------------------------------
                # 2.3 Skip decision-only trajectory state
                # --------------------------------------------------------
    
                if (
                    input_ids is None
                    or attention_mask is None
                ):
                    continue
    
                # --------------------------------------------------------
                # 2.4 Forward model
                # --------------------------------------------------------
    
                outputs: BaseModelOutputWithPast = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    document_attention_mask=(
                        document_attention_mask
                    ),
                    retrieval_decision=decision_score,
                )
    
                hidden_states = outputs.last_hidden_state
    
                # --------------------------------------------------------
                # 2.5 Get logits for newly computed tokens only
                # --------------------------------------------------------
    
                current_sequence_length = (
                    hidden_states.shape[1]
                )
    
                new_hidden_states = hidden_states[
                    :,
                    computed_logits_length:current_sequence_length,
                    :,
                ]
    
                new_logits = self.lm_head(
                    new_hidden_states
                )
    
                logits_buffer.append(new_logits)
    
                computed_logits_length = (
                    current_sequence_length
                )
    
                # --------------------------------------------------------
                # 2.6 Compute new retrieval decision
                # --------------------------------------------------------
    
                decision_score = (
                    self.retrieval_decision_head(
                        new_hidden_states
                    )
                )
    
            # ============================================================
            # 3. Accumulate final decision score
            #
            # If the last trajectory item is decision-only,
            # its decision score has not been accumulated by
            # the loop body, so add it here.
            # ============================================================
    
            # ============================================================
            # 4. Combine logits
            # ============================================================
    
            if not logits_buffer:
                raise ValueError(
                    "Trajectory does not contain any valid "
                    "states for training."
                )
    
            logits = torch.cat(
                logits_buffer,
                dim=1,
            )
    
            assert logits.shape[1] == labels.shape[1], (
                "Logits and labels must have the same sequence length: "
                f"logits.shape={logits.shape}, "
                f"labels.shape={labels.shape}"
            )
    
            # ============================================================
            # 5. Handle empty retrieval cost
            # ============================================================
    
            if total_decision_score is None:
                retrieval_cost = logits.new_zeros(()).sum()
            else:
                retrieval_cost = compute_retrieval_cost(
                    total_decision_score
                )
    
            # ============================================================
            # 6. Compute language modeling loss
            # ============================================================
    
            lm_loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
            )
    
            # ============================================================
            # 7. Compute retrieval cost
            # ============================================================
    
            if trajectory_items:
            
                last_item = trajectory_items[-1]
            
                if (
                    last_item["input_ids"] is None
                    or last_item["attention_mask"] is None
                ):
            
                    if decision_score is not None:
                        retrieval_cost += RETRIEVAL_COST * decision_score.sum()    
    
            loss = (
                lm_loss
                + retrieval_cost
            )
    
            # ============================================================
            # 8. Return
            # ============================================================
    
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )