# No-RAG and Vanilla RAG baselines

Both entry points use the original `Qwen/Qwen2.5-3B-Instruct` for the model and tokenizer. The shared runner uses the evaluation loaders listed in [ART's top-5 generator](../REVIVE_training/evaluate_art_pure_generation_top5.py). It preserves loader order, with no shuffling or resampling. `--limit N` selects the first N records. No-RAG sends the question as the user message with a fixed system prompt. Vanilla RAG queries the selected database once with the original question, adds up to five ranked passages to the system message, then generates once. No-RAG does not import or initialize any database.

## Datasets and retrieval sources

All loaders target dataset repo `qingfei1/R-Search_datasets` and file `data/eval/{dataset}/test.jsonl`. The existing loaders prefer local copies and download the corresponding file when absent. The table records the Hugging Face dataset IDs and filenames verified in the local adapter source. The index repositories were unavailable to independent inspection during validation. Revisions are unpinned.

| Dataset | Shared loader | ART class | Hugging Face index dataset | Corpus file |
|---|---|---|---|---|
| `musique` | `load_rsearch_musique` | `MuSiQueRetrievalDB` | `adaptive-retrieval-timing/MuSiQue-ART-E5-Retrieval-Index` | `musique_corpus.jsonl` |
| `hotpotqa` | `load_rsearch_hotpotqa` | `HotpotQARetrievalDB` | `adaptive-retrieval-timing/HotpotQA-ART-E5-Retrieval-Index` | `hotpotqa_corpus.jsonl` |
| `2wikimultihopqa` | `load_rsearch_2wikimultihopqa` | `TwoWikiMultiHopQARetrievalDB` | `adaptive-retrieval-timing/2WikiMultiHopQA-ART-E5-Retrieval-Index` | `2wikimultihopqa_corpus.jsonl` |

Each adapter loads `e5_Flat.index` and the associated corpus, and encodes queries using `intfloat/e5-base-v2`. Vanilla RAG calls `search_with_metadata(question, top_k=5)` once per sample and preserves returned rank order, title and document text without a passage-text cap. Dense similarities remain in the returned passage metadata.

## Prompt and decoding protocol

The system prompts follow the supplied [FlashRAG English demo](https://github.com/RUC-NLPIR/FlashRAG/blob/main/examples/quick_start/demo_en.py), including consecutive sentences with no space between the first period and the next sentence. No-RAG ends its system prompt with one LF. Vanilla RAG adds the reference introduction, a blank line, and ranked references in FlashRAG's `Doc N(Title: TITLE) TEXT\n` format, retaining body line breaks. The user message is exactly the question. This reuses the reference format without FlashRAG's own prompt truncation. [FlashRAG reference formatter](https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/prompt/base_prompt.py#L196-L206)

R-Search's public baseline code uses distinct answer-only instructions in a single user message and formats references without the numbered prefix. The checked public sources do not establish a match to the published benchmark runs. These scripts generate local ART comparison baselines. [R-Search baseline evaluator](https://github.com/QingFei1/R-Search/blob/main/src/eval/main.py#L25-L66) · [R-Search prompt configuration](https://github.com/QingFei1/R-Search/blob/main/src/eval/config.yaml#L15-L20)

The runner follows ART's system/user chat template, `tokenize=False`, `add_generation_prompt=True`, and separate tokenizer pass. It seeds once with 44. Generation uses BF16 eager attention, sampling with temperature 0.1, top-p 0.9, sampling top-k 0, repetition penalty 1.0, one beam, one returned sequence and up to 1,024 new tokens. It decodes only the continuation with `skip_special_tokens=False`. The generator weights, system prompts, context insertion and retrieval timing differ from ART. Matching the loaders does not independently establish the revisions or IDs used in historical runs.

## Run from the repository root

```powershell
python -m REVIVE_training.evaluate_no_rag_generation --dataset musique
python -m REVIVE_training.evaluate_no_rag_generation --dataset hotpotqa
python -m REVIVE_training.evaluate_no_rag_generation --dataset 2wikimultihopqa
python -m REVIVE_training.evaluate_vanilla_rag_generation --dataset musique
python -m REVIVE_training.evaluate_vanilla_rag_generation --dataset hotpotqa
python -m REVIVE_training.evaluate_vanilla_rag_generation --dataset 2wikimultihopqa
```

Defaults are `REVIVE_training/result/baselines/{dataset}/no_rag.jsonl` and `REVIVE_training/result/baselines/{dataset}/vanilla_rag_top5.jsonl`. `--output` accepts a new JSONL path. The runner creates parent directories and opens the file exclusively. An interrupted run can leave a partial file. Use a new output path on retry.

Every output row preserves the input record including `question`, `golden_answers`, and metadata. It preserves the original ID or supplies ART's zero-based row fallback. The raw continuation is in `pure_art_answer`. ART-compatible fields include `pure_art_queries`, `pure_art_passages` as formatted strings, and an empty `pure_art_retrieval_scores` because these baselines have no ART decision scores. Vanilla RAG also saves structured `retrieved_passages` with the adapter's metadata.

Reuse the unchanged scorer, for example:

```powershell
python -m REVIVE_training.evaluate_art_pure --filename REVIVE_training/result/baselines/musique/vanilla_rag_top5.jsonl
```

The scorer uses the last complete answer tag pair or falls back to the stripped continuation, then computes Acc, F1 and EM. The supplied baseline prompts do not request tags. Untagged prose can therefore affect EM, F1 and substring Acc differently from ART. The example scores into `REVIVE_training/metrics/baselines/musique/vanilla_rag_top5_metrics.json`.

## Validation

- Syntax, imports and both help commands passed in the project virtual environment.
- Loaded 500 local MuSiQue and 500 local HotpotQA records with the real loaders. The 2Wiki parser and dispatch were checked with a one-record fixture because the upstream test file was absent locally and the public file could not be inspected.
- Rendered prompts with the cached Qwen tokenizer. Checked ordered passages, multiline body text and the assistant generation marker.
- Six one-record mocked runs verified one generation per record, one top-5 metadata call for Vanilla RAG, no retriever for No-RAG, metadata preservation, scorer-compatible JSONL and refusal to overwrite an existing path.
- The project virtual environment lacks the `faiss` module, so adapter routing used mocked database construction. Live index loading, retrieval and model generation remain unverified. No full evaluations ran.
