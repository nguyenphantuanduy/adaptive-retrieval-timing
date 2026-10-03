from __future__ import annotations

from transformers import Trainer
from typing import Any, Optional, Union

import torch
from torch import nn

class ARTTrainer(Trainer):
    """
    Hugging Face Trainer with automatic logging of ART retrieval statistics.

    The model is expected to expose:

        model.retrieval_count_for_logging: list[int]
        model.retrieve_segment_indices: list[int]
        model.first_segment_retrieval_decision_score: float | None

    At each Trainer logging step:

        - Average retrieval count is logged as `retrieval_count`.
        - Retrieved segment indices are logged as
          `retrieve_segment_indices`.
        - First segment retrieval decision score is logged as
          `first_segment_retrieval_decision_score` if not None.
    """

    def log(
        self,
        logs: dict[str, Any],
        start_time: float | None = None,
    ) -> None:
        """
        Compute and log ART retrieval statistics since the
        previous logging step, then reset the logging values.
        """

        model = self.model

        # Unwrap if needed (Accelerate / DDP / DeepSpeed / FSDP ...)
        if hasattr(model, "module"):
            model = model.module

        # ============================================================
        # RETRIEVAL COUNT
        # ============================================================

        if hasattr(model, "retrieval_count_for_logging"):
            retrieval_counts = model.retrieval_count_for_logging

            if retrieval_counts:
                average_retrieval_count = (
                    sum(retrieval_counts) / len(retrieval_counts)
                )

                logs["retrieval_count"] = average_retrieval_count

                retrieval_counts.clear()

        # ============================================================
        # RETRIEVED SEGMENT INDICES
        # ============================================================

        if hasattr(model, "retrieve_segment_indices"):
            retrieve_segment_indices = model.retrieve_segment_indices

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
        # CALL PARENT TRAINER LOG
        # ============================================================

        super().log(
            logs,
            start_time=start_time,
        )

    def training_step(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            num_items_in_batch: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # ============================================================
            # 1. Collect trajectory
            # ============================================================
            model.gradient_checkpointing_disable()
            
            with torch.no_grad():
                collect_outputs = model.forward_collect(
                    **inputs,
                )
    
            model.gradient_checkpointing_enable()
    
            training_inputs = {
                "collect_outputs": collect_outputs,
            }
    
            return super().training_step(
                model,
                training_inputs,
                num_items_in_batch,
            )