# REVIVE training and evaluation implementation

See [training provenance and teacher datasets](TRAINING.md) for the source defaults and author-confirmed settings. Both demo checkpoints trained for 0.75 epochs. The current v1 script displays 0.1 after the accidental later edit.

## Student inputs and parameters

[training.py](../REVIVE_training/training.py#L152) and [training_v2.py](../REVIVE_training/training_v2.py#L144) load per-hop teacher train splits, concatenate and deterministically shuffle. V2 switches dataset repository, adds string and nonempty validation for teacher text, and names the layer-selection region explicitly. Both apply the same last-40%-rounded-up rule and also train the decision head. The model is loaded with eager attention and trusted custom code. Both preserve the loaded head and enable ART's explicit training dispatcher. The source diff does not show a replacement loss or trainer.

In [v2 tokenization and collation](../REVIVE_training/training_v2.py#L230), the student sees the raw question in a chat template with assistant generation prompt. The teacher output plus EOS is separately tokenized without automatic specials. The dataset map drops original columns after constructing raw question, separate token streams, a concatenated attention mask and labels. The collator requires one sample, with `-100` for question labels and actual teacher token IDs for continuation labels. Per-device microbatch is one and the source accumulation factor is 32. Both freeze embeddings, output head and all other weights outside the selected decoder layers and decision head. See [freeze_backbone](../REVIVE_training/training_v2.py#L489).

## Collection fixes the retrieval trajectory

[ARTTrainer.training_step](../REVIVE_training/trainer.py#L164) disables gradient checkpointing for collection, runs `forward_collect` under no-grad, restores checkpointing in a finally block, and passes only the collected object to Trainer's ordinary training step. The local collection path uses the teacher sequence and original question.

1. Prefill the question and score the final hidden vector. Initial search is a hard threshold comparison.
2. Feed teacher token IDs through the sentence splitter and process each buffered block. Capture a prefix snapshot immediately before each accepted retrieval. A later boundary can trigger either via the head or via exploration. The local cap is four and the exploration probability increases after no search and decreases after search within the configured bounds.
3. Tokenize retrieved observations, insert them into the accumulated decoder prefix, and insert an equal number of ignored labels. Maintain the latest document span and active decision score. `isCritic=True` substitutes query-plus-observation into the collected buffer while cache forwarding and mask/length bookkeeping use the observation-only sequence. `ARTTrainer` leaves this option at its default false.
4. A remaining incomplete block gets a final prefix snapshot. A final threshold crossing can increment retrieval logging and add a decision-only marker without making a search. Thus the collection counters need not equal the number of search calls. See [model.py:531–1175](../shared/ART_Models/ART_REVIVE/model.py#L531).

**Static edge case.** The final prefix snapshot is inside `if sentence_buffer`. If the stream ends on a detected boundary, a post-retrieval observation or trailing nonretrieval block may lack a final snapshot. Replay asserts equal logits and label lengths. This is a reachable control-flow concern conditional on token and splitter behavior. No run was executed to measure its frequency.

## Differentiable replay

[forward_training](../shared/ART_Models/ART_REVIVE/model.py#L1757) re-forwards each stored full prefix without KV caching, with the stored document mask and the previously recomputed score. It takes the suffix of hidden states beyond the prior covered token count and concatenates the corresponding vocabulary logits. Prefix snapshots mark retrieval boundaries, which lets replay reconstruct a separate score and context for each next block. Saved detached scores serve as trajectory metadata. Replay recomputes scores so they can receive gradients.

Let \(y_t\) be continuation labels after inserting ignored entries at document positions. The LM term scores each unmasked \(y_t\) from logits at \(t-1\). The active total loss is LM cross-entropy plus \(0.05\sum_{j\in J} d_j\) for the local model. \(J\) follows the loop's previous-score accumulation and special terminal branch. In the current code a terminal decision-only entry lets the previous score enter the loop accumulator and then adds that same score again in the terminal branch. This is a static implementation finding. Its frequency and historical presence are unknown.

The score gradient reaches the head through attention and the explicit penalty. A score \(d=\sigma(z)\) gives a local chain factor \(d(1-d)\) to the head logit. The hard search comparison, query selection, document identities and exploration coin were fixed during collection and are outside the gradient graph. On a trajectory with no differentiably used or charged decisions the LM term can train the enabled backbone layers without a decision-head gradient. Final model/tokenizer saves occur after Trainer training and diagnostics. See [trainer.py:34–161](../REVIVE_training/trainer.py#L34) for logging windows, including separate head and exploration counts.

## Generation and metric pipeline

[evaluate_art_pure_generation_top5.py:24–137, 195–351](../REVIVE_training/evaluate_art_pure_generation_top5.py#L24) currently selects the R-Search 2Wiki test loader; commented alternatives select MuSiQue or HotpotQA. The loader resolves its packaged test path and reads JSONL into sample dictionaries. Generation loads the unpinned ART repo and Qwen tokenizer in BF16, formats the question with the ART system prompt, seeds once, and samples with temperature 0.1, top-p 0.9, sampling top-k 0, one beam and up to 1,024 generated tokens. The call requests retrieval top-k 5 and passes -1 for no passage-text truncation. Seed-once means run order matters for sampling reproducibility. Only tokens after the initial chat prompt are decoded. Responses and flat diagnostics are saved as `pure_art_answer`, `pure_art_passages`, `pure_art_queries`, and `pure_art_retrieval_scores`.

[evaluate_art_pure.py:115–185, 247–311, 392–474](../REVIVE_training/evaluate_art_pure.py#L115) excludes blank lines, malformed JSON and objects missing the answer field. For scored records it takes the last complete answer tag pair and falls back to the entire stripped output when no pair matches. It counts tag coverage separately and uses each record's `golden_answers`. The denominator is the included record count. Missing tags can therefore affect substring Acc and normalized EM/F1 differently.

[utils.py:18–91](../REVIVE_training/utils.py#L18) computes:

- Acc as the proportion where any gold alias is a case-insensitive substring.
- EM as the maximum exact equality across normalized aliases, averaged per record.
- F1 as the best token multiset F1 across normalized aliases, averaged per record. With overlap \(c\) and predicted and gold token counts \(p,g\), F1 is \(2c/(p+g)\) when \(c>0\), otherwise zero.

Normalization lowercases, removes ASCII punctuation and English articles, and collapses whitespace. All three metrics are percentages rounded to two decimals. The metrics do not directly measure retrieval timing, evidence recall, generation coherence, or whether the answer was derived from the passages.
