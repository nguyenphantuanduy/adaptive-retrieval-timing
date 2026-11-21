# Task: Implement No-RAG and Vanilla RAG Evaluation Baselines for ART

Implement two baseline generation scripts using the original `Qwen/Qwen2.5-3B-Instruct` model, each supporting MuSiQue, HotpotQA, and 2WikiMultiHopQA.

## 1. Inspect existing code before implementing

Read `REVIVE_training/evaluate_art_pure_generation_top5.py` as the authoritative reference for test-set loading, sample ordering/selection, random seed, tokenizer/chat template, generation settings, result schema, and output conventions. Inspect related dataset loaders, evaluators, and `REVIVE_training/utils.py` as needed. Do not guess values; extract them from code.

Inspect `shared/ART_Models/ART_REVIVE/`, especially:
- `musique_retrieval_db.py`
- `hotpotqa_retrieval_db.py`
- `two_wikimultihopqa_db.py`

Reuse their Hugging Face index loading and E5/FAISS retrieval interfaces where possible. Do not rebuild indexes. Verify how to query and obtain title/text/ordered top-k passages.

## 2. Create two scripts

- `REVIVE_training/evaluate_no_rag_generation.py`
- `REVIVE_training/evaluate_vanilla_rag_generation.py`

Both must accept `--dataset musique|hotpotqa|2wikimultihopqa`, optional `--limit`, and optional `--output`.

### No-RAG
1. Load the exact matching ART test set and model/tokenizer.
2. Prompt with the question only, without retrieval.
3. Generate once and write JSONL with sample ID, question, gold answer, generated text, and compatible ART evaluation fields.
4. Never initialize or call a retriever.

FlashRAG example template (not yet verified as the exact R-Search evaluation prompt):

```python
system_prompt = (
    "You are a friendly AI Assistant."
    "Respond to the input as a friendly AI assistant, "
    "generating human-like text, and follow the "
    "instructions in the input if applicable.\n"
)
user_prompt = "{question}"
```

### Vanilla RAG
1. Load the same test set and base model/tokenizer.
2. Retrieve once with the original question using the matching ART database and `top_k=5`.
3. Format ordered retrieved passages, preserving titles and text, into `{reference}`.
4. Prompt with question and references; generate once (no iterative search, rewriting, or further retrieval).
5. Save the same sample fields plus retrieved passages.

FlashRAG example template (not yet verified as the exact R-Search evaluation prompt):

```python
system_prompt = (
    "You are a friendly AI Assistant."
    "Respond to the input as a friendly AI assistant, "
    "generating human-like text, and follow the "
    "instructions in the input if applicable."
    "\nThe following are provided references. "
    "You can use them for answering question.\n\n"
    "{reference}"
)
user_prompt = "{question}"
```

Use the same chat-template conventions as the ART evaluator. Verify exact whitespace and reference formatting from available sources; document any deviations. Do not describe these prompts as confirmed R-Search prompts without checking its source.

## 3. Ensure fair comparison

- Match the ART evaluation samples, ordering, seed, generation settings, and compatible output format.
- Use original Qwen2.5-3B-Instruct, **not** an ART checkpoint.
- Preserve IDs, questions, gold answers, and metadata for paired comparison.
- Default output paths:
  - `REVIVE_training/result/baselines/{dataset}/no_rag.jsonl`
  - `REVIVE_training/result/baselines/{dataset}/vanilla_rag_top5.jsonl`
- Create directories; never overwrite existing ART outputs.
- Reuse existing answer-extraction/Acc/F1/EM evaluators when compatible; avoid duplicating scoring logic.
- Explicitly document unavoidable differences from the ART evaluation.

## 4. Validate and document

Create `docs/RAG_BASELINE_IMPLEMENTATION.md` explaining the pipeline, prompts and their provenance, dataset loaders, HF retrieval/index sources, generation settings, JSONL schema, and commands for all six dataset/baseline combinations.

Run syntax/import tests, dataset loading, prompt construction, retrieval formatting, and JSONL serialization checks. Optionally run a one-example generation smoke test if inexpensive. **Do not run full evaluation or expensive GPU inference without permission.**

Do not change existing ART scripts, model architecture, training code, datasets, indexes, or prior experiment results. If a source is incompatible or an essential detail cannot be verified, report it instead of silently substituting behavior.

## 5. Final report

Summarize files changed, inherited generation configuration, each dataset loader and retrieval class, verified HF sources, all six run commands, test results, and remaining uncertainties.
