# ART methodology

For layer-by-layer and token-by-token details, see [REVIVE architecture](REVIVE_ARCHITECTURE.md). The [training implementation](REVIVE_TRAINING_EVALUATION.md) covers replay accounting and its edge cases. The [hypothesis chapter](HYPOTHESIS_VERIFICATION.md) reports the saved H1 and H2 measurements and their interpretation.

## Retrieval decision and attention

In [REVIVE model.py](../shared/ART_Models/ART_REVIVE/model.py#L296), the decision head takes the final hidden vector of the current block. Its linear layer and sigmoid give \(d_t=\sigma(w^\top h_t+b)\). At inference, the model evaluates the score at prompt prefill and after a completed sentence. Local `forward_generate` retrieves when \(d_t\ge\tau\) and the constructed query lacks the blocked answer/observation markers. The current model constant is 0.4297. The training cap and exploration hooks are commented out at the inference gate.

[RetrievalContextProcessor](../shared/ART_Models/ART_REVIVE/retrieval_context_processor.py#L59) feeds tokens to a streaming sentence splitter. Query construction decodes generated tokens and resegments them with lookahead. It returns the latest confirmed sentence, falling back to the original question if none is available. Synthetic observation tokens are inserted through the model's retrieval path. `ARTCache` holds generation state, retrieval score/mask and the two rotary-position cache views, and `utils.py` provides document masks and no-document position IDs. These paths are implemented in [cache_utils.py](../shared/ART_Models/ART_REVIVE/cache_utils.py), [utils.py](../shared/ART_Models/ART_REVIVE/utils.py) and [decoder_layer.py](../shared/ART_Models/ART_REVIVE/decoder_layer.py).

For the same value tensor \(V\), [attention_mixing_forward](../shared/ART_Models/ART_REVIVE/attention_mixing.py#L26) implements:

- \(A_+=\operatorname{softmax}(Q_+K_+^\top/\sqrt{d_k}+M_{\rm causal})\).
- \(A_-=\operatorname{softmax}(Q_-K_-^\top/\sqrt{d_k}+M_{\rm causal}+M_{\rm doc})\), where \(M_{\rm doc}\) masks document keys and the minus branch uses document-free rotary positions.
- \(O=\operatorname{Dropout}(d_t A_+ + (1-d_t)A_-)V\) when both document mask and decision are available. Otherwise it uses ordinary with-document attention weights.

The interpolation acts on attention probabilities within the same attention computation. The branch formula is not a separate end-to-end forward pass with retrieved text physically removed. The current retrieval score and document mask are carried forward in the cache after insertion. The inference path concatenates retrieved document hidden states with the block before computing output logits.

## Collect and replay

1. [ARTTrainer.training_step](../REVIVE_training/trainer.py#L164) temporarily disables gradient checkpointing and collects under no-grad. [forward_collect](../shared/ART_Models/ART_REVIVE/model.py#L531) consumes the teacher continuation as the token stream. It records accumulating input prefixes and masks at decision boundaries. Initial retrieval is threshold-gated. At later sentence boundaries, eligible retrieval can be threshold-triggered or exploration-triggered. The local collection cap is four, and the exploration probability begins at 0.3, increases by 0.05 up to 0.5 and halves down to 0.1 on the implemented update paths. The final incomplete block can create a terminal decision-only entry.
2. The trainer restores checkpointing and passes the fixed trajectory to [forward_training](../shared/ART_Models/ART_REVIVE/model.py#L1757). Replay recomputes states with gradients, using the previous replay score as the attention mixing coefficient. It adds only the newly computed token logits per prefix, concatenates them and checks their length against labels. The hard search decision and dense lookup happened in collection. Gradients flow through the replay computation and score penalty.
3. The objective is \(\mathcal L=\mathcal L_{\rm LM}+0.05\sum_{j\in J}d_j\) for the current local implementation. Here \(\mathcal L_{\rm LM}\) is the causal LM loss over unmasked teacher continuation labels, and \(J\) follows the scores accumulated by replay, including the conditional terminal charge. The penalty uses scores, rather than a count of retrieval calls. The teacher gives text supervision. No binary retrieval targets or RL reward enter this loss.

## Research probes

[H1](../phase_1/stage_2_hypothesis_verification/H1_attention_modulation/README.md) proposes checking language preservation, ordered information control and answer behavior. Its `model_input.py` constructs document spans and label regions. `run_lm_loss_evaluation.py` sweeps forced mixing settings and computes regional LM losses. `plot_lm_loss_vs_decision_score.py` aggregates and plots losses.

[H2](../phase_1/stage_2_hypothesis_verification/H2_gradient_direction/run_gradient_evaluation.py) installs a gradient hook on a decision tensor and measures loss sensitivity across mixing ratios. A negative gradient with respect to a score favors increasing it locally under gradient descent, and a positive gradient favors decreasing it. This interpretation is conditional on the input, region and probe setting. Probe code establishes the measurement procedure. It does not establish empirical verification of the hypotheses by itself.
