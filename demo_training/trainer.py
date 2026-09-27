from __future__ import annotations

from transformers import Trainer


class ARTTrainer(Trainer):
    """
    Hugging Face Trainer with automatic logging of ART retrieval statistics.

    The model is expected to expose:

        model.retrieval_count_for_logging: list[int]
        model.retrieve_segment_indices: list[int]
        model.first_segment_retrieval_decision_score: float | None

    During each training forward:

        - retrieval_count_for_logging records the number of
          retrievals for that forward pass.
        - retrieve_segment_indices records the segment indices
          at which retrieval occurred.
        - first_segment_retrieval_decision_score records the
          retrieval decision score of the first segment.

    At each Trainer logging step:

        - Average retrieval count is logged as `retrieval_count`.
        - Retrieved segment indices are logged as
          `retrieve_segment_indices`.
        - First segment retrieval decision score is logged as
          `first_segment_retrieval_decision_score` if it is not None.
        - Logging values are reset after logging.
    """

    def log(
        self,
        logs: dict[str, float],
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
                # Average retrieval count across all forward passes
                # since the previous logging step.
                average_retrieval_count = (
                    sum(retrieval_counts) / len(retrieval_counts)
                )

                logs["retrieval_count"] = average_retrieval_count

                # Reset after logging so the next log interval
                # starts with an empty list.
                retrieval_counts.clear()

        # ============================================================
        # RETRIEVED SEGMENT INDICES
        # ============================================================

        if hasattr(model, "retrieve_segment_indices"):
            retrieve_segment_indices = model.retrieve_segment_indices

            if retrieve_segment_indices:
                # Store the segment indices observed since the
                # previous logging step.
                #
                # Example:
                # [0, 2, 5, 0, 3]
                #
                # means retrieval occurred at those segment indices
                # across the forward passes in this logging interval.
                logs["retrieve_segment_indices"] = list(
                    retrieve_segment_indices
                )

                # Reset after logging.
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

                # Reset after logging so the next logging interval
                # does not reuse the previous value.
                model.first_segment_retrieval_decision_score = None

        # ============================================================
        # CALL PARENT TRAINER LOG
        # ============================================================

        super().log(
            logs,
            start_time=start_time,
        )