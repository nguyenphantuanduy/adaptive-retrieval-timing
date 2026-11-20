# Academic report preparation

## Reusable explanations

- **Problem and scope.** Multi-hop QA can require external evidence at different points in reasoning. ART studies a learned retrieval gate evaluated at prompt and sentence boundaries. It derives retrieval queries from confirmed reasoning text or the initial question.
- **Mechanism.** A scalar computed from the latest hidden state triggers retrieval at inference and controls document attention through a probability-space mixture. The language-model loss supervises the teacher continuation and the retrieval score penalty adds pressure against high scores.
- **Training evidence.** The two historical demos trained for equal epoch fractions, according to the author. Demo-3 yields higher archived answer scores on the two corresponding MuSiQue settings. Multiple teacher-pipeline changes accompany the checkpoint transition.
- **Threshold evidence.** On one author-confirmed demo-3 checkpoint, the saved MuSiQue results peak at 0.60 over three thresholds. HotpotQA peaks at 0.50 over two observed points. These are sample- and configuration-specific observations.
- **External context.** R-Search provides a published multi-reward RL benchmark with shared task names, model family at 3B and dense retriever family. The methods and evaluation evidence differ as documented in [benchmark comparability](R_SEARCH_COMPARISON.md).

## Proposed figures and tables

| Deliverable | What to show | Evidence |
|---|---|---|
| Architecture diagram | Prompt → decoder → sentence boundary and decision head → query/retriever → cache and attention mixture → next token | [Methodology](METHODOLOGY.md) |
| Training diagram | Teacher target → no-grad trajectory collection → replay → LM loss and score penalty | [Workflows](WORKFLOWS.md) |
| Training provenance table | Historical and current epochs, teacher datasets, changed prompt rules and available checkpoint evidence | [Training](TRAINING.md) |
| ART results table | Matched demo rows, sample counts and metric definitions, deltas in percentage points | [Experiments](EXPERIMENTS.md) |
| Threshold plot | EM and F1 at the saved points with dataset labels, leaving unobserved points blank | [Experiments](EXPERIMENTS.md) |
| Related-work benchmark table | Published 3B EM/F1 and comparison conditions | [Benchmark reference](R_SEARCH_BENCHMARK_REFERENCE.md) |

Claims about optimal thresholds, efficiency, statistically reliable gains, teacher quality, or causal effects of attention mixing need additional evidence. Artifact aggregates support descriptive comparisons. The [outstanding evidence](EXPERIMENTS.md#outstanding-evidence) section records what would be needed to support broader claims. Cite source functions for mechanisms, author provenance for historical settings, and paper sections or tables for external numbers.
