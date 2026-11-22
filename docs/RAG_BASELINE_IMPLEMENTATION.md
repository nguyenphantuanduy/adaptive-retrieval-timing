# No-RAG and Vanilla RAG baselines

Both entry points use the original `Qwen/Qwen2.5-3B-Instruct` for the model and tokenizer and preserve ART loader order. `--limit N` selects the first N records. No-RAG formats the question with the R-Search baseline template. Vanilla RAG makes one search with the preprocessed question and inserts up to five ranked passages into the baseline template before generating once. No-RAG does not import or initialize any database.

## Datasets and retrieval sources

All loaders target dataset repo `qingfei1/R-Search_datasets` and file `data/eval/{dataset}/test.jsonl`. The existing loaders prefer local copies and download the corresponding file when absent. The table records the Hugging Face dataset IDs and filenames verified in the local adapter source. The index repositories were unavailable to independent inspection during validation. Revisions are unpinned.

| Dataset | Shared loader | ART class | Hugging Face index dataset | Corpus file |
|---|---|---|---|---|
| `musique` | `load_rsearch_musique` | `MuSiQueRetrievalDB` | `adaptive-retrieval-timing/MuSiQue-ART-E5-Retrieval-Index` | `musique_corpus.jsonl` |
| `hotpotqa` | `load_rsearch_hotpotqa` | `HotpotQARetrievalDB` | `adaptive-retrieval-timing/HotpotQA-ART-E5-Retrieval-Index` | `hotpotqa_corpus.jsonl` |
| `2wikimultihopqa` | `load_rsearch_2wikimultihopqa` | `TwoWikiMultiHopQARetrievalDB` | `adaptive-retrieval-timing/2WikiMultiHopQA-ART-E5-Retrieval-Index` | `2wikimultihopqa_corpus.jsonl` |

Each adapter loads `e5_Flat.index` and the associated corpus, and encodes queries using `intfloat/e5-base-v2`. Vanilla RAG calls `search_with_metadata(preprocessed_question, top_k=5)` once per sample and preserves returned rank order, title and document text without a passage-text cap. Dense similarities remain in the returned passage metadata.

## Prompt and decoding protocol

Both templates reproduce the parsed R-Search YAML values. Because the source uses single-quoted YAML scalars, `\n` in the templates is a literal backslash followed by `n`. No-RAG retains the space before the first colon. Vanilla RAG retains the extra double quote after its answer-only instruction. Reference formatting is `(Title: {title}) {text}` with one actual LF appended per ranked hit. The runner inserts the corresponding ART adapter's title and text in returned order, including body line breaks. [Official prompt configuration](https://github.com/QingFei1/R-Search/blob/main/src/eval/config.yaml#L19-L20) ? [Official reference formatter](https://github.com/QingFei1/R-Search/blob/main/src/eval/main.py#L57-L66)

Both modes trim the question and append `?` if absent. Vanilla RAG uses the same preprocessed question for retrieval and prompting, and the saved `question` remains the original dataset field. The rendered prompt is one user message with `tokenize=False` and `add_generation_prompt=True`. If a tokenizer has no chat template, the formatted prompt is used directly. [Official baseline evaluator](https://github.com/QingFei1/R-Search/blob/main/src/eval/main.py#L25-L47) ? [Question preprocessing](https://github.com/QingFei1/R-Search/blob/main/src/eval/main.py#L142-L152)

The runner keeps ART's separate tokenizer pass and seeds once with 44. Generation uses BF16 eager attention, sampling with temperature 0.1, top-p 0.9, sampling top-k 0, repetition penalty 1.0, one beam, one returned sequence and up to 1,024 new tokens. It decodes only the continuation with `skip_special_tokens=False`. This is the official baseline prompt and prediction protocol on the local ART data, retrievers and Transformers runner. Equality with the published R-Search numbers would require further verification.

## Run from the repository root

```powershell
python -m REVIVE_training.evaluate_no_rag_generation --dataset musique
python -m REVIVE_training.evaluate_no_rag_generation --dataset hotpotqa
python -m REVIVE_training.evaluate_no_rag_generation --dataset 2wikimultihopqa
python -m REVIVE_training.evaluate_vanilla_rag_generation --dataset musique
python -m REVIVE_training.evaluate_vanilla_rag_generation --dataset hotpotqa
python -m REVIVE_training.evaluate_vanilla_rag_generation --dataset 2wikimultihopqa
```

Defaults are `REVIVE_training/result/baselines/{dataset}/no_rag.jsonl` and `REVIVE_training/result/baselines/{dataset}/vanilla_rag_top5.jsonl`. Each run writes to a new temporary file in the destination directory and atomically replaces the selected JSONL after successful completion. A failed run preserves any previous output. An abrupt process termination can leave a temporary file beside it. `--output` still accepts a custom JSONL path and applies the same replacement behavior. Use custom paths dedicated to the selected baseline and dataset. The runner rejects symbolic-link destinations, other managed baseline outputs, and the project's ART result directories.

Every output row preserves the input record including `question`, `golden_answers`, and metadata. It preserves the original ID or supplies ART's zero-based row fallback. The raw continuation is in `pure_art_answer`. ART-compatible fields include `pure_art_queries`, `pure_art_passages` as formatted strings, and an empty `pure_art_retrieval_scores` because these baselines have no ART decision scores. Vanilla RAG also saves structured `retrieved_passages` with the adapter's metadata.

Score the five result files currently available from the repository root.

```powershell
python -m REVIVE_training.evaluate_baseline --filename REVIVE_training/result/baselines/2wikimultihopqa/no_rag.jsonl
python -m REVIVE_training.evaluate_baseline --filename REVIVE_training/result/baselines/2wikimultihopqa/vanilla_rag_top5.jsonl
python -m REVIVE_training.evaluate_baseline --filename REVIVE_training/result/baselines/hotpotqa/no_rag.jsonl
python -m REVIVE_training.evaluate_baseline --filename REVIVE_training/result/baselines/musique/no_rag.jsonl
python -m REVIVE_training.evaluate_baseline --filename REVIVE_training/result/baselines/musique/vanilla_rag_top5.jsonl
```

When the HotpotQA Vanilla RAG result arrives, run:

```powershell
python -m REVIVE_training.evaluate_baseline --filename REVIVE_training/result/baselines/hotpotqa/vanilla_rag_top5.jsonl
```

The scorer removes only the literal text/chat-control tokens `<|im_end|>`, `<|endoftext|>`, and `<|im_start|>` from predictions in memory. These match the text-control entries in the Qwen2.5-3B-Instruct tokenizer configuration. It leaves the raw JSONL, ordinary text, punctuation, whitespace, and answer tags intact. It passes the cleaned string to ART's existing Acc, F1 and EM functions, which still apply their normal metric-level normalization.

Each command scores one file independently and writes `REVIVE_training/metrics/baselines/{dataset}/{variant}_metrics.json`. A missing input produces a CLI error without writing metrics. Existing metrics files are preserved. Move an earlier metrics file aside if you want to save a new score for the same input.

## Validation

Run `python -m unittest test.test_rag_baselines` for prompt, output and scoring regression tests. Scoring cases cover clean text, multiple delimiters, raw answer-tag preservation and a missing input. The following checks were run during the initial baseline implementation.

- Syntax, imports and both help commands passed in the project virtual environment.
- Loaded 500 local MuSiQue and 500 local HotpotQA records with the real loaders. The 2Wiki parser and dispatch were checked with a one-record fixture because the upstream test file was absent locally and the public file could not be inspected.
- Rendered prompts with the cached Qwen tokenizer. Checked ordered passages, multiline body text and the assistant generation marker.
- Six one-record mocked runs verified one generation per record, one top-5 metadata call for Vanilla RAG, no retriever for No-RAG, metadata preservation and scorer-compatible JSONL.
- Output regression tests verify that a successful rerun replaces the selected JSONL without appending, a failed rerun preserves it, and protected destinations are rejected.
- The project virtual environment lacks the `faiss` module, so adapter routing used mocked database construction. Live index loading, retrieval and model generation remain unverified. No full evaluations ran.
