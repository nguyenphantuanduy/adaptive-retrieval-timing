# Adaptive Retrieval Timing (ART)

ART studies when a language model should retrieve evidence while solving multi-hop questions. The REVIVE implementation adds a retrieval decision head to a Qwen2.5-3B-Instruct-derived causal LM. It tests the question and completed reasoning segments for a retrieval trigger, inserts retrieved passages into the model context, and mixes attention with and without access to document positions. Training uses teacher-generated reasoning and answers, a language-modeling loss, and a penalty on retrieval decision scores. The training objective does not require labels specifying when to retrieve.

| Start here | Content |
|---|---|
| [Codebase map and workflows](WORKFLOWS.md) | Data, corpus/index, model initialization, training and evaluation entry points |
| [Codebase reading coverage](CODEBASE_COVERAGE.md) | Recursive inventory, file-by-file review depth, exclusions and remaining gaps |
| [Methodology](METHODOLOGY.md) | Score, attention, cache, trajectory collection and replay, loss, probes |
| [REVIVE architecture and inference](REVIVE_ARCHITECTURE.md) | Decoder math, retrieval state machine, mask scope, cache synchronization and retriever contracts |
| [REVIVE training and evaluation implementation](REVIVE_TRAINING_EVALUATION.md) | Student inputs, collection/replay, gradient routes, and metrics |
| [H1 and H2 hypothesis verification](HYPOTHESIS_VERIFICATION.md) | Probe design, saved aggregate results, threshold derivation and evidentiary limits |
| [Training and teacher datasets](TRAINING.md) | Script comparison, historical epoch correction and prompt changes |
| [ART experiments](EXPERIMENTS.md) | Metrics, provenance, demo comparison, threshold sweeps and missing evidence |
| [R-Search benchmark reference](R_SEARCH_BENCHMARK_REFERENCE.md) | Published numbers and settings |
| [ART and R-Search comparability](R_SEARCH_COMPARISON.md) | Configuration matches, differences and evidence limits |
| [Academic report preparation](REPORT_PREPARATION.md) | Reusable explanations, figures, tables and claim boundaries |

For future sessions, read this index and the linked chapter for the task before tracing the cited functions. **Source** describes the current checkout. **Author** identifies historical settings supplied by the project author. **Artifact** identifies values read from aggregate metrics. **Interpretation** identifies a conclusion drawn from those sources. A current script default does not establish a historical run setting. Keep archived result JSONL files closed unless the author grants permission to inspect them. This documentation was prepared through static inspection and aggregate metrics review.
