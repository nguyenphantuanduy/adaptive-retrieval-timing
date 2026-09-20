"""
ART H1 Gradient Probe
=====================

For ONE MuSiQue validation sample:

    sample
      ↓
    create_lm_loss_samples(sample)
      ↓
    each document ratio
      ↓
    33 decision levels:
        d = 0/32, 1/32, ..., 32/32
      ↓
    create ONE HF Dataset with 33 samples
      ↓
    sample 0 → d=0/32
    sample 1 → d=1/32
    ...
    sample 32 → d=32/32
      ↓
    one HF Trainer.train()
      ↓
    batch_size=1
      ↓
    capture ∂L/∂d with Tensor.register_hook()
      ↓
    print gradient + timing

Important:
- d is an ordinary Tensor, NOT nn.Parameter.
- d is taken directly from the HF Dataset batch.
- d.requires_grad_(True) is enabled inside forward().
- Hook is registered on the actual Tensor used by ART forward().
- The actual ART model is used directly.
- No GradientProbeModel wrapper.
- No custom Trainer.
- No custom Dataset.
- HF Dataset is used directly.
- ART model parameters are frozen.
- learning_rate = 0.
- Trainer uses gradient checkpointing.
"""

import argparse
import gc
import time
from typing import Any

import torch
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from shared.MuSiQue.load_hypothesis_validation import (
    load_teacher_generated_hypothesis_validation,
)

from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.build_lm_loss_samples import (
    create_lm_loss_samples,
)

from phase_1.stage_2_hypothesis_verification.H1_attention_modulation.model_input import (
    create_model_input_for_lm_loss,
)


# ============================================================
# CONFIG
# ============================================================

NUM_DECISION_LEVELS = 32

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

HF_REPO = (
    "adaptive-retrieval-timing/"
    "ART-Attention-Mixing-Probe"
)

OUTPUT_DIR = "./art_gradient_probe_tmp"

DEFAULT_MODE = "2"


# ============================================================
# CUDA CLEANUP
# ============================================================

def cleanup_cuda() -> None:

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================
# GRADIENT CAPTURE
# ============================================================

class GradientCapture:

    def __init__(self):

        self.gradients = []

    def hook(self, grad):

        self.gradients.append(
            grad.detach().clone()
        )


# ============================================================
# BUILD HF DATASET
# ============================================================

def build_dataset(
    model_input: dict[str, Any],
) -> Dataset:
    """
    Convert the existing 33-row model_input directly
    into an HF Dataset with 33 samples.

    IMPORTANT:
    No new values are generated here.

    Every Dataset column is taken directly from
    model_input.

    Expected shapes:

        input_ids:
            (33, seq_len)

        attention_mask:
            (33, seq_len)

        document_attention_mask:
            (33, seq_len)

        retrieval_decision:
            (33,)

        labels:
            (33, seq_len)

    Therefore:

        Dataset[0]  -> d = 0/32
        Dataset[1]  -> d = 1/32
        ...
        Dataset[32] -> d = 32/32
    """

    num_samples = (
        model_input["input_ids"].shape[0]
    )

    assert (
        num_samples ==
        NUM_DECISION_LEVELS + 1
    )

    # --------------------------------------------------------
    # Verify all tensors have the same first dimension
    # --------------------------------------------------------

    for key, value in model_input.items():

        if torch.is_tensor(value):

            assert value.shape[0] == num_samples, (
                f"{key}: expected first dimension "
                f"{num_samples}, got {value.shape[0]}"
            )

    # --------------------------------------------------------
    # Take values directly from model_input.
    #
    # No d is manually created here.
    # No decision values are regenerated here.
    # --------------------------------------------------------

    dataset_dict = {}

    for key, value in model_input.items():

        if not torch.is_tensor(value):
            continue

        dataset_dict[key] = (
            value.detach()
            .cpu()
            .tolist()
        )

    dataset = Dataset.from_dict(
        dataset_dict
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert len(dataset) == 33

    assert (
        dataset[0]["retrieval_decision"]
        == 0.0
    )

    assert (
        abs(
            dataset[32]["retrieval_decision"]
            - 1.0
        )
        < 1e-7
    )

    print(
        f"HF Dataset created: "
        f"{len(dataset)} samples"
    )

    print(
        "Decision range: "
        f"{dataset[0]['retrieval_decision']:.6f}"
        " → "
        f"{dataset[32]['retrieval_decision']:.6f}"
    )

    return dataset


# ============================================================
# PATCH ART FORWARD
# ============================================================

def install_gradient_hook(
    model,
    capture: GradientCapture,
):
    """
    Patch model.forward() so that the exact
    retrieval_decision Tensor received from Trainer is:

        requires_grad_(True)
        register_hook(...)

    No replacement d value is created.

    Returns the original forward function so it can
    be restored afterward.
    """

    original_forward = model.forward

    def forward_with_gradient_hook(
        *args,
        **kwargs,
    ):

        # ----------------------------------------------------
        # Get the actual retrieval_decision Tensor
        # coming from Trainer's batch.
        # ----------------------------------------------------

        d = kwargs["retrieval_decision"]

        assert torch.is_tensor(d)

        # ----------------------------------------------------
        # Make the actual batch Tensor differentiable.
        #
        # No clone()
        # No detach()
        # No new d
        # ----------------------------------------------------

        d.requires_grad_(True)

        # ----------------------------------------------------
        # Register hook on this exact Tensor.
        # ----------------------------------------------------

        d.register_hook(
            capture.hook
        )

        # ----------------------------------------------------
        # Run the real ART forward.
        # ----------------------------------------------------

        return original_forward(
            *args,
            **kwargs,
        )

    model.forward = forward_with_gradient_hook

    return original_forward


# ============================================================
# MEASURE ALL 33 DECISIONS FOR ONE RATIO
# ============================================================

def measure_ratio_gradients(
    model,
    model_input: dict[str, Any],
    ratio,
    device: torch.device,
) -> list[float]:

    num_decisions = (
        model_input["input_ids"].shape[0]
    )

    assert (
        num_decisions ==
        NUM_DECISION_LEVELS + 1
    )

    print()
    print("=" * 70)

    print(
        f"RATIO {ratio[0]}:{ratio[1]}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Build ONE Dataset containing all 33 decisions.
    # --------------------------------------------------------

    dataset = build_dataset(
        model_input
    )

    # --------------------------------------------------------
    # Gradient capture
    # --------------------------------------------------------

    capture = GradientCapture()

    # --------------------------------------------------------
    # Patch model.forward()
    # --------------------------------------------------------

    original_forward = (
        install_gradient_hook(
            model=model,
            capture=capture,
        )
    )

    # --------------------------------------------------------
    # Trainer arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        # Exactly ONE decision per Trainer step.
        per_device_train_batch_size=1,

        # Dataset contains exactly 33 samples.
        # Therefore exactly 33 steps.
        max_steps=33,

        # We do NOT want to update any model parameter.
        learning_rate=0.0,

        logging_steps=1,

        report_to="none",

        remove_unused_columns=False,

        save_strategy="no",

        eval_strategy="no",

        gradient_accumulation_steps=1,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset,
    )

    try:

        start = time.perf_counter()

        # ----------------------------------------------------
        # ONE Trainer.train()
        #
        # 33 samples
        # batch_size=1
        # => 33 forward/backward steps
        # ----------------------------------------------------

        trainer.train()

        elapsed = (
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # Verify exactly 33 gradients
        # ----------------------------------------------------

        if len(capture.gradients) != 33:

            raise RuntimeError(
                "Expected 33 captured gradients, "
                f"got {len(capture.gradients)}."
            )

        # ----------------------------------------------------
        # Extract one scalar gradient per decision.
        #
        # Each captured gradient corresponds to:
        #
        #   d = 0/32
        #   d = 1/32
        #   ...
        #   d = 32/32
        #
        # Because batch_size = 1, each Tensor has
        # one decision value.
        # ----------------------------------------------------

        gradients = []

        for decision_index, grad in enumerate(
            capture.gradients
        ):

            gradient_value = (
                grad
                .float()
                .mean()
                .item()
            )

            gradients.append(
                gradient_value
            )

            decision_score = (
                decision_index /
                NUM_DECISION_LEVELS
            )

            print(
                f"d={decision_score:.4f} "
                f"({decision_index:2d}/32) "
                f"grad={gradient_value:+.8e}"
            )

        print()
        print(
            f"Trainer time for 33 decisions: "
            f"{elapsed:.3f} seconds"
        )

        return gradients

    finally:

        # ----------------------------------------------------
        # Restore original ART forward.
        # ----------------------------------------------------

        model.forward = original_forward

        # ----------------------------------------------------
        # Cleanup Trainer / Dataset.
        # ----------------------------------------------------

        del trainer
        del dataset
        del capture

        cleanup_cuda()


# ============================================================
# EVALUATE ONE SAMPLE
# ============================================================

def evaluate_one_sample(
    model,
    tokenizer,
    sample: dict[str, Any],
    sample_index: int,
    device: torch.device,
):

    sample_id = sample["id"]

    print()
    print("#" * 80)
    print("ART H1 GRADIENT PROBE")
    print("#" * 80)

    print(
        f"sample_index = {sample_index}"
    )

    print(
        f"sample_id    = {sample_id}"
    )

    print()

    # --------------------------------------------------------
    # Create LM-loss samples
    # --------------------------------------------------------

    lm_loss_samples = (
        create_lm_loss_samples(
            sample
        )
    )

    print(
        f"Number of document ratios: "
        f"{len(lm_loss_samples)}"
    )

    all_gradients = {}

    sample_start = time.perf_counter()

    # --------------------------------------------------------
    # Loop over document ratios
    # --------------------------------------------------------

    for ratio, lm_sample in (
        lm_loss_samples.items()
    ):

        print()
        print(
            f"Creating model input "
            f"for ratio={ratio[0]}:{ratio[1]}"
        )

        # ----------------------------------------------------
        # Same input construction as H1 LM-loss experiment.
        #
        # This already contains 33 decision levels.
        # ----------------------------------------------------

        model_input = (
            create_model_input_for_lm_loss(
                lm_sample,
                tokenizer,
            )
        )

        # ----------------------------------------------------
        # Verify 33 decisions.
        # ----------------------------------------------------

        assert (
            model_input[
                "retrieval_decision"
            ].shape
            == (33,)
        )

        # ----------------------------------------------------
        # Measure all 33 gradients using ONE Trainer.
        # ----------------------------------------------------

        gradients = (
            measure_ratio_gradients(

                model=model,

                model_input=model_input,

                ratio=ratio,

                device=device,
            )
        )

        all_gradients[ratio] = gradients

        del model_input
        del lm_sample

        cleanup_cuda()

    sample_elapsed = (
        time.perf_counter()
        - sample_start
    )

    del lm_loss_samples

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print()
    print("#" * 80)
    print("FINAL GRADIENT RESULTS")
    print("#" * 80)

    for ratio, gradients in (
        all_gradients.items()
    ):

        print()

        print(
            f"Ratio {ratio[0]}:{ratio[1]}"
        )

        print("-" * 70)

        for decision_index, gradient in (
            enumerate(gradients)
        ):

            decision_score = (
                decision_index /
                NUM_DECISION_LEVELS
            )

            print(
                f"d={decision_score:.4f} "
                f"grad={gradient:+.10e}"
            )

    print()

    print("#" * 80)

    print(
        f"TOTAL TIME FOR ONE SAMPLE: "
        f"{sample_elapsed:.3f} seconds"
    )

    print("#" * 80)

    return (
        all_gradients,
        sample_elapsed,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model_and_tokenizer():

    print(
        "Loading tokenizer...",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "Loading ART probe...",
        flush=True,
    )

    model = (
        AutoModelForCausalLM.from_pretrained(

            HF_REPO,

            torch_dtype=torch.bfloat16,

            device_map={"": 0},

            trust_remote_code=True,

            attn_implementation="eager",
        )
    )

    # --------------------------------------------------------
    # Freeze ART model parameters.
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.eval()

    print(
        "Model loaded and frozen.",
        flush=True,
    )

    return model, tokenizer


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "ART H1 gradient probe "
            "for one sample"
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["2", "3", "4"],
        help=(
            "Which hop dataset to use: "
            "2, 3, or 4."
        ),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help=(
            "Index of the sample "
            "to evaluate."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}",
        flush=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, tokenizer = (
        load_model_and_tokenizer()
    )

    # --------------------------------------------------------
    # MuSiQue validation dataset
    # --------------------------------------------------------

    print(
        "Loading MuSiQue validation dataset...",
        flush=True,
    )

    all_datasets = (
        load_teacher_generated_hypothesis_validation()
    )

    hop = {
        "2": "2-hop",
        "3": "3-hop",
        "4": "4-hop",
    }[args.mode]

    samples = all_datasets[hop]

    # --------------------------------------------------------
    # Validate sample index
    # --------------------------------------------------------

    if args.sample_index >= len(samples):

        raise IndexError(

            f"sample-index={args.sample_index} "
            f"but {hop} only contains "
            f"{len(samples)} samples."
        )

    sample = samples[
        args.sample_index
    ]

    print(
        f"Using {hop}, "
        f"sample {args.sample_index}/"
        f"{len(samples) - 1}",
        flush=True,
    )

    # --------------------------------------------------------
    # Run gradient probe
    # --------------------------------------------------------

    evaluate_one_sample(

        model=model,

        tokenizer=tokenizer,

        sample=sample,

        sample_index=args.sample_index,

        device=device,
    )