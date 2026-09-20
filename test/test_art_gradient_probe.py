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
    HF Trainer.train()
      ↓
    batch_size=1
      ↓
    capture ∂L/∂d with Tensor.register_hook()
      ↓
    map each gradient to its ACTUAL decision value d
      ↓
    print gradients in ordered d = 0/32 ... 32/32

Important:
- d is an ordinary Tensor, NOT nn.Parameter.
- d is taken directly from the HF Dataset batch.
- d.requires_grad_(True) is enabled inside forward().
- Hook is registered on the exact Tensor used by ART forward().
- The actual ART model is used directly.
- No GradientProbeModel wrapper.
- No custom Trainer.
- No custom Dataset.
- HF Dataset is used directly.
- ART model parameters are frozen.
- learning_rate = 0.
- Trainer uses gradient checkpointing.
- Gradient results are indexed by the ACTUAL d value,
  not by Trainer iteration order.
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
NUM_DECISIONS = NUM_DECISION_LEVELS + 1

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

        # IMPORTANT:
        # Store gradient by ACTUAL decision index.
        #
        # Example:
        #   gradients[0]  = gradient for d=0/32
        #   gradients[1]  = gradient for d=1/32
        #   ...
        #   gradients[32] = gradient for d=32/32
        #
        # This makes the result independent of
        # Trainer's sampling order.

        self.gradients: dict[int, torch.Tensor] = {}

    def hook(
        self,
        decision_index: int,
        grad: torch.Tensor,
    ) -> None:

        # Each Trainer batch has batch_size=1,
        # therefore grad should contain exactly one d.

        gradient_value = (
            grad.detach()
            .clone()
            .float()
            .mean()
            .item()
        )

        self.gradients[decision_index] = gradient_value


# ============================================================
# GET DECISION INDEX FROM ACTUAL d
# ============================================================

def get_decision_index(
    d: torch.Tensor,
) -> int:
    """
    Convert the ACTUAL retrieval_decision Tensor
    received by model.forward() into its decision index.

    Expected:

        d = 0.0       -> 0
        d = 1/32      -> 1
        ...
        d = 1.0       -> 32

    No new d value is created.
    This only reads the actual Tensor value.
    """

    assert torch.is_tensor(d)

    # batch_size = 1
    assert d.numel() == 1, (
        "Expected exactly one decision value "
        f"because batch_size=1, got shape={tuple(d.shape)}"
    )

    decision_score = (
        d.detach()
        .float()
        .reshape(-1)[0]
        .item()
    )

    decision_index = int(
        round(
            decision_score *
            NUM_DECISION_LEVELS
        )
    )

    # --------------------------------------------------------
    # Validate that d really corresponds to one of:
    #
    # 0/32, 1/32, ..., 32/32
    # --------------------------------------------------------

    expected_score = (
        decision_index /
        NUM_DECISION_LEVELS
    )

    assert (
        abs(
            decision_score -
            expected_score
        )
        < 1e-6
    ), (
        "Unexpected retrieval_decision value: "
        f"d={decision_score}"
    )

    assert (
        0 <= decision_index <= NUM_DECISION_LEVELS
    )

    return decision_index


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
    """

    num_samples = (
        model_input["input_ids"].shape[0]
    )

    assert (
        num_samples ==
        NUM_DECISIONS
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

    assert len(dataset) == NUM_DECISIONS

    assert (
        dataset[0]["retrieval_decision"]
        == 0.0
    )

    assert (
        abs(
            dataset[NUM_DECISIONS - 1][
                "retrieval_decision"
            ]
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
        f"{dataset[NUM_DECISIONS - 1]['retrieval_decision']:.6f}"
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

    The actual d value is read and used to identify
    which decision the captured gradient belongs to.

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
        # Identify this batch by its ACTUAL d value.
        #
        # This is the important fix.
        #
        # We do NOT assume:
        #
        #   first Trainer step  -> d=0
        #   second Trainer step -> d=1/32
        #
        # Instead we read d itself.
        # ----------------------------------------------------

        decision_index = get_decision_index(d)

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
            lambda grad,
            idx=decision_index:
                capture.hook(idx, grad)
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
        NUM_DECISIONS
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
        max_steps=NUM_DECISIONS,

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
        #
        # Trainer may process the dataset in an order that
        # is not guaranteed to match dataset index order.
        #
        # That is OK because GradientCapture stores each
        # gradient using the ACTUAL decision index.
        # ----------------------------------------------------

        trainer.train()

        elapsed = (
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # Verify exactly 33 UNIQUE gradients.
        # ----------------------------------------------------

        if len(capture.gradients) != NUM_DECISIONS:

            missing = [
                i
                for i in range(NUM_DECISIONS)
                if i not in capture.gradients
            ]

            raise RuntimeError(
                "Expected 33 unique captured gradients, "
                f"got {len(capture.gradients)}. "
                f"Missing decisions: {missing}"
            )

        # ----------------------------------------------------
        # Extract gradients in CORRECT decision order.
        #
        # IMPORTANT:
        #
        # We explicitly access:
        #
        #   gradients[0]
        #   gradients[1]
        #   ...
        #   gradients[32]
        #
        # instead of relying on capture order.
        # ----------------------------------------------------

        gradients = []

        for decision_index in range(
            NUM_DECISIONS
        ):

            gradient_value = (
                capture.gradients[
                    decision_index
                ]
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
            == (NUM_DECISIONS,)
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