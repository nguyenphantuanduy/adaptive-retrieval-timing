from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import Trainer


class ARTTrainer(Trainer):
    """
    Hugging Face Trainer with automatic logging of ART retrieval statistics.

    The model is expected to expose:

        model.model_retrieval_count_for_logging: list[int]
        model.exploration_retrieval_count_for_logging: list[int]
        model.retrieval_count_for_logging: list[int]
        model.retrieve_segment_indices: list[int]
        model.first_segment_retrieval_decision_score: float | None
    """

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ================================================================
    # LOGGING
    # ================================================================

    def log(
        self,
        logs: dict[str, Any],
        start_time: float | None = None,
    ) -> None:
        """
        Compute and log ART retrieval statistics since the previous
        logging step, then reset the logging values.
        """

        model = self.model

        # Unwrap if needed (DDP / other wrappers).
        if hasattr(model, "module"):
            model = model.module

        # ============================================================
        # MODEL RETRIEVAL COUNT
        # ============================================================

        if hasattr(model, "model_retrieval_count_for_logging"):

            model_retrieval_counts = (
                model.model_retrieval_count_for_logging
            )

            if model_retrieval_counts:

                average_model_retrieval_count = (
                    sum(model_retrieval_counts)
                    / len(model_retrieval_counts)
                )

                logs["model_retrieval_count"] = (
                    average_model_retrieval_count
                )

                model_retrieval_counts.clear()

        # ============================================================
        # EXPLORATION RETRIEVAL COUNT
        # ============================================================

        if hasattr(model, "exploration_retrieval_count_for_logging"):

            exploration_retrieval_counts = (
                model.exploration_retrieval_count_for_logging
            )

            if exploration_retrieval_counts:

                average_exploration_retrieval_count = (
                    sum(exploration_retrieval_counts)
                    / len(exploration_retrieval_counts)
                )

                logs["exploration_retrieval_count"] = (
                    average_exploration_retrieval_count
                )

                exploration_retrieval_counts.clear()

        # ============================================================
        # RETRIEVAL COUNT
        # ============================================================

        if hasattr(model, "retrieval_count_for_logging"):

            retrieval_counts = model.retrieval_count_for_logging

            if retrieval_counts:

                average_retrieval_count = (
                    sum(retrieval_counts)
                    / len(retrieval_counts)
                )

                logs["retrieval_count"] = average_retrieval_count

                retrieval_counts.clear()

        # ============================================================
        # RETRIEVED SEGMENT INDICES
        # ============================================================

        if hasattr(model, "retrieve_segment_indices"):

            retrieve_segment_indices = (
                model.retrieve_segment_indices
            )

            if retrieve_segment_indices:

                logs["retrieve_segment_indices"] = list(
                    retrieve_segment_indices
                )

                retrieve_segment_indices.clear()

        # ============================================================
        # FIRST SEGMENT RETRIEVAL DECISION SCORE
        # ============================================================

        if hasattr(model, "first_segment_retrieval_decision_score"):

            first_segment_score = (
                model.first_segment_retrieval_decision_score
            )

            if first_segment_score is not None:

                logs["first_segment_retrieval_decision_score"] = (
                    first_segment_score
                )

                model.first_segment_retrieval_decision_score = None

        # ============================================================
        # PARENT TRAINER LOG
        # ============================================================

        super().log(
            logs,
            start_time=start_time,
        )

    # ================================================================
    # TRAINING STEP
    # ================================================================

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # ============================================================
        # 1. COLLECT TRAJECTORY
        # ============================================================

        # Preserve the original gradient-checkpointing state.
        was_gradient_checkpointing_enabled = getattr(
            model,
            "is_gradient_checkpointing",
            False,
        )

        if was_gradient_checkpointing_enabled:
            model.gradient_checkpointing_disable()

        try:

            with torch.no_grad():

                collect_outputs = model.forward_collect(
                    **inputs,
                )

        finally:

            # Restore gradient checkpointing even if collection fails.
            if was_gradient_checkpointing_enabled:
                model.gradient_checkpointing_enable()

        # ============================================================
        # 2. PREPARE TRAINING INPUTS
        # ============================================================

        training_inputs = {
            "collect_outputs": collect_outputs,
        }

        # ============================================================
        # 3. NORMAL TRAINING STEP
        # ============================================================

        return super().training_step(
            model,
            training_inputs,
            num_items_in_batch,
        )