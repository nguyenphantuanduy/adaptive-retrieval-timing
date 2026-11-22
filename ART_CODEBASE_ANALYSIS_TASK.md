# Task: Understand and Document the Entire ART Project

You are working on **ART (Adaptive Retrieval Timing)**, an academic research project that has already been implemented. Your task is to understand the existing codebase and experimental work, then create comprehensive technical documentation for future sessions and an academic report (Đồ án chuyên ngành).

## 1. Explore the repository

Read and understand all relevant source code, configurations, scripts, existing documentation, training, inference, evaluation, dataset-processing, and retrieval components. Follow dependencies, function calls, and data flows rather than relying on README files alone.

**Exception — large results:** Do **not** read or load the contents of long experimental `result` JSONL files. Use the corresponding `metrics` files instead. You may inspect filenames and directory structure to map results to metrics. If raw results are indispensable, ask me for permission before reading them.

## 2. Experimental context

The `art-backup` directory contains artifacts from earlier demonstrations. These mappings are confirmed:

- `demo-2` corresponds to `REVIVE_training/training.py`.
- `demo-3` corresponds to `REVIVE_training/training_v2.py`.

Compare these training implementations and their corresponding metrics carefully. Do not infer causality or superiority without evidence. Identify any missing experimental configuration.

## 3. Clarify uncertainties immediately

**If something important is unclear, ambiguous, contradictory, or undocumented, stop and ask me before proceeding with assumptions.** This includes research motivations, design choices, module relationships, experiment configurations, metric definitions, and differences between training versions. You may group closely related questions. Resume after my answer and incorporate the clarification into the documentation. Never invent missing explanations.

## 4. Write documentation under `docs/`

Once the relevant questions have been resolved, create a structured set of Markdown files in the repository's `docs/` directory, with `docs/README.md` as the entry point. Cover:

1. Project overview: problem, motivation, goals, scope, architecture.
2. Codebase map: directories, modules, key functions/classes, dependencies.
3. End-to-end workflows: data preparation, training, retrieval, inference, evaluation.
4. Research methodology: ART mechanisms, models, losses, algorithms, equations grounded in code.
5. Training comparison: `training.py` versus `training_v2.py`.
6. Experiments: datasets, configurations, demo-2/demo-3 metrics and comparisons.
7. Limitations, missing evidence, and unresolved questions.
8. Preparation for the academic report: reusable explanations, proposed figures/tables, substantiated findings.

The documentation should be detailed enough for a future Codex session or another assistant to start from `docs/` instead of rediscovering the repository. Cite relative source paths, functions/classes, and line numbers when useful. Distinguish **confirmed code behavior**, **observed experimental results**, and **interpretations**.

## 5. Constraints

- Do not modify existing source code, datasets, checkpoints, logs, or experimental artifacts.
- Do not run training/evaluation, install packages, or download models/datasets.
- Only create or update files within `docs/`.
- Do not write the final academic report yet.
- Prefer accuracy, traceability, and completeness over speed.

## 6. Final response

Summarize your understanding of ART, list documentation files, explain key differences between demo-2 and demo-3, identify unresolved uncertainties, and tell me how future Codex sessions should use `docs/README.md`.

**Understand first. Ask immediately when uncertain. Document accurately.**
