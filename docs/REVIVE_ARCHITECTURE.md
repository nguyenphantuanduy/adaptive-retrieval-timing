# REVIVE architecture and inference

This chapter describes the local REVIVE implementation. Evaluation scripts load custom code through Hugging Face. A repository name without an immutable revision does not establish which implementation an archived run executed.

## Model forward

[model.py:93–290](../shared/ART_Models/ART_REVIVE/model.py#L93) builds a Qwen2-style embedding, stack of customized layers, rotary embeddings and final RMSNorm. Each [decoder layer](../shared/ART_Models/ART_REVIVE/decoder_layer.py#L21) performs pre-normalized attention with a residual, then pre-normalized Qwen2 MLP with another residual. The causal head projects hidden states into vocabulary logits, with a tied-weight mapping to token embeddings. A separate head reads the last hidden vector in a block and computes \(d=\sigma(w^\top h_{\mathrm{last}}+b)\). The head supplies a scalar attention coefficient and a thresholdable decision score. The sigmoid alone does not calibrate retrieval benefit.

[attention_mixing.py:26–175, 190–379](../shared/ART_Models/ART_REVIVE/attention_mixing.py#L26) projects one residual stream to Q/K/V, applies RoPE twice to Q and K, and expands grouped KV heads to the query head count. Softmax is evaluated in float32 and converted back. With causal mask \(C\) and document-key mask \(D\),

- \(A_+=\mathrm{softmax}(Q_+K_+^\top/\sqrt{d_k}+C)\).
- \(A_-=\mathrm{softmax}(Q_-K_-^\top/\sqrt{d_k}+C+D)\).
- \(O=\mathrm{Dropout}(dA_++(1-d)A_-)V_+\), followed by the output projection. With either score or document mask missing, the weights are \(A_+\).

The same incoming hidden states generate both projections. Both branches share values from the document-aware cache. With fixed projections and masks, direct attention mass on masked keys scales with \(d\). Across repeated layers the incoming hidden states can themselves depend on \(d\), so an ordered end-to-end information or accuracy response needs empirical evidence. The mask helper treats the marked span as contiguous. It subtracts its length from the absolute positions of tokens after it, while attention masking separately suppresses access to its keys in the minus branch. The alignment helper expands or slices the key axis to the causal mask. [utils.py:2–253](../shared/ART_Models/ART_REVIVE/utils.py#L2)

## Inference state machine

[model.py:452–529, 1178–1515](../shared/ART_Models/ART_REVIVE/model.py#L1178) routes normal calls to `forward_generate`. It requires input IDs, the raw question and an attention mask.

~~~text
on prefill or a non-ART cache:
    create ARTCache and reset per-generation diagnostics
on a subsequent decode step:
    restore ART's internal mask and append one valid position

hidden = decoder(current tokens, cache, previous mask, previous score)
if prefill or streaming splitter reports a sentence boundary:
    candidate = sigmoid(head(last hidden))
    log candidate
    if candidate >= threshold:
        query = latest confirmed sentence, or original question
        if query contains no blocked answer or observation marker:
            retrieve ranked passages and format an observation
            extend the valid-token mask by the observation length
            forward observation using the previous document mask and score
            append observation hidden states to current hidden block
            store the new document span and score, synchronize caches
return vocabulary logits from the requested hidden-state slice and ARTCache
~~~

A search can change the logits used for the immediately following generated token because the appended observation ends the block passed to the LM head. Synthetic observation IDs are internal decoder inputs. The generation wrapper's decoded continuation and separately logged passages are distinct. The local inference threshold is 0.4297 and the inference quota and exploration clauses are commented out. The search count and query array only advance for actual calls, while the score array includes rejected decisions. A flat passage array cannot give exact query-to-passage boundaries when fewer than top-k records are returned. A low candidate score keeps the previous active document mask and score.

[retrieval_context_processor.py:112–260](../shared/ART_Models/ART_REVIVE/retrieval_context_processor.py#L112) decodes each emitted token for the streaming splitter and buffers all model-generated token IDs. Boundary detection may fire on a token belonging to the next sentence. Query extraction therefore separately decodes the whole emitted stream, resegments it with lookahead and discards an unstable trailing segment. It selects the last confirmed sentence, using the raw question when no sentence is confirmed. That can make the hidden state used by the trigger slightly later than the text chosen for the query. Only row zero is fed into the splitter and one score is converted via `.item()`, so the implementation targets single-example inference.

## Cache semantics

[cache_utils.py:45–150, 192–228, 288–311](../shared/ART_Models/ART_REVIVE/cache_utils.py#L45) stores two Transformers `DynamicCache` instances and logical sequence state. Each layer appends keys rotated according to its branch and appends shared projected values. Cached length and generation-facing cache access are anchored to the document-aware view. At retrieval, the model clones that view's initialized K/V tensors into the second cache. That costs cache-sized storage and copying and means the second cache starts from the same prefix immediately following insertion. It is not a counterfactual full decoder run with evidence removed.

The new document mask marks only the newly appended observation, while previous tokens get zeros in that mask. Prior observations remain in the KV prefix and can still affect both attention views. Between retrievals the previous mask is padded with zeros as generated tokens arrive. The next synchronization copies the entire state again. The decoder constructs causality from the with-document timeline and applies the document mask only to the minus-key scores.

## Retrieval adapter contracts

All three adapters load dataset-specific corpus and FAISS artifacts, check corpus/index row count, use a prefixed E5 query, masked mean pooling and L2 normalization, and pass a float32 CPU vector to FAISS. Ranked corpus row IDs select the passages. An optional limit truncates the combined passage text with the generator tokenizer before observation wrapping. The returned passage list can still describe the pre-truncation hits.

| Adapter | Key implementation | Contract |
|---|---|---|
| 2Wiki | [two_wikimultihopqa_db.py:155–241, 384–585](../shared/ART_Models/ART_REVIVE/two_wikimultihopqa_db.py#L384) | Active local default. Regular search returns observation and query-plus-observation. `return_passages=True` returns observation and passage list. |
| HotpotQA | [hotpotqa_retrieval_db.py:99–175, 290–444](../shared/ART_Models/ART_REVIVE/hotpotqa_retrieval_db.py#L290) | Default returns observation string. Optional flag returns observation and passages. |
| MuSiQue | [musique_retrieval_db.py:91–125, 177–332](../shared/ART_Models/ART_REVIVE/musique_retrieval_db.py#L177) | Default returns observation string. Optional flag returns observation and passages. |

Collection currently unpacks two strings from default search, while inference passes `return_passages=True`. Swapping 2Wiki for either other adapter in the current training path therefore requires adapting the call contract. Retrievers and tokenizer are lazy-loaded.

The [initializer](../shared/ART_Models/ART_REVIVE/initialize_art_revive.py#L889) and [seeded initializer](../shared/ART_Models/ART_REVIVE/initialize_art_revive_with_seed.py#L713) transfer compatible Qwen weights, initialize and check the new linear decision head, package model/config/tokenizer and custom code, and upload. These are initialization workflows. The training scripts load a head from their configured checkpoint.
