from __future__ import annotations

from transformers import Trainer


class ARTTrainer(Trainer):
    """
    Hugging Face Trainer with automatic logging of
    the average ART retrieval count.

    The model is expected to expose:

        model.retrieval_count_for_logging: list[int]

    Each training forward appends its retrieval count to this list.
    At each Trainer logging step, the average is computed and the
    list is reset.
    """

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        """
        Compute and log the average retrieval count since the
        previous logging step, then reset the logging list.
        """

        model = self.model

        # Unwrap if needed (Accelerate / DDP / DeepSpeed / FSDP ...)
        if hasattr(model, "module"):
            model = model.module

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
                model.retrieval_count_for_logging.clear()

        super().log(
            logs,
            start_time=start_time,
        )