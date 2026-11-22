"""ART model configuration"""

from huggingface_hub.dataclasses import strict

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


@strict
class ARTConfig(Qwen2Config):

    model_type = "art"

    keys_to_ignore_at_inference = ["past_key_values"]

    # ==========================================================
    # Tensor Parallel
    # ==========================================================

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    # ==========================================================
    # Pipeline Parallel
    # ==========================================================

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    # ==========================================================
    # ART
    # ==========================================================

    # Layer index where the backbone splits into:
    #   1. next-token prediction branch
    #   2. retrieval-decision branch
    decision_branch_start_layer: int = 22

    # Number of Qwen layers used by the retrieval-decision branch
    num_decision_layers: int = 14


__all__ = ["ARTConfig"]