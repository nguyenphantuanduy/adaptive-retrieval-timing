from __future__ import annotations

from transformers import Trainer


class ARTTrainer(Trainer):
    """
    Hugging Face Trainer with automatic logging of
    the ART retrieval count stored in model.retrieval_count.
    """

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        """
        Append the model's retrieval count to Trainer logs.

        The model is expected to expose:

            model.retrieval_count: int
        """

        model = self.model

        # Unwrap if needed (Accelerate / DDP / DeepSpeed / FSDP ...)
        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "retrieval_count"):
            logs["retrieval_count"] = model.retrieval_count

        super().log(
            logs,
            start_time=start_time,
        )