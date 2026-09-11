# IRCoT — Interleaving Retrieval with Chain-of-Thought Reasoning

**Paper:** Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal
**Title:** _Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions_
**Venue:** ACL 2023 Long Paper
**Paper:** [ACL Anthology](https://aclanthology.org/2023.acl-long.557.pdf)
**Code:** [GitHub — StonyBrookNLP/ircot](https://github.com/StonyBrookNLP/ircot)

---

## 1. Problem

Knowledge-intensive multi-step questions require the model to combine information from multiple sources and perform reasoning across several steps.

A conventional retrieve-then-read pipeline can be represented as:

```text
Question → Retrieve → Reason → Answer
```

This approach has an important limitation: the original question may not contain enough information to retrieve all documents required for solving a multi-hop question.

For example, the question may require:

```text
Question
   ↓
Document A
   ↓
Reasoning
   ↓
Document B
   ↓
Reasoning
   ↓
Answer
```

However, if retrieval is performed only once using the original question, Document B may be difficult to retrieve because the information needed to formulate a useful query for Document B is only discovered after reasoning over Document A.

IRCoT addresses this problem by interleaving reasoning and retrieval.

Instead of:

```text
Question → Retrieve → Reason → Answer
```

IRCoT uses:

```text
Reason → Retrieve → Reason → Retrieve → ... → Answer
```

The main research question is therefore how intermediate reasoning can be used to guide subsequent retrieval for knowledge-intensive multi-step questions.

---

## 2. Core Idea

The core idea of IRCoT is to **interleave Chain-of-Thought reasoning with retrieval**.

The process is approximately:

```text
Question
   ↓
Initial Retrieval
   ↓
Reasoning
   ↓
Retrieve using reasoning
   ↓
More Reasoning
   ↓
Retrieve again
   ↓
...
   ↓
Final QA
```

The initial retrieval uses the original question.

After documents are retrieved, the language model generates a reasoning sentence. This reasoning sentence is then used as the query for the next retrieval step.

Therefore, the retrieval process is guided by the model's evolving reasoning state.

### Important detail

At each reasoning step, the LM may generate multiple sentences, but IRCoT **keeps only the first generated sentence and discards the rest**.

That first reasoning sentence becomes the query for the next retrieval step.

Therefore, the procedure is effectively:

```text
Current context
      ↓
Generate CoT
      ↓
Keep first sentence
      ↓
Use it as retrieval query
      ↓
Retrieve new paragraphs
      ↓
Continue reasoning
```

The process terminates when:

- the generated CoT contains `"answer is:"`, or
- the maximum number of reasoning steps is reached.

After retrieval terminates, **all collected paragraphs are passed to a separate QA reader** to produce the final answer.

Thus, IRCoT can be viewed as two major stages:

```text
Stage 1:
IRCoT Retriever
Reason ↔ Retrieve
        ↓
Collected paragraphs

Stage 2:
QA Reader
Question + collected paragraphs
        ↓
Final Answer
```

---

## 3. Model, Retriever

### Retriever

IRCoT uses **BM25 implemented through Elasticsearch** as the underlying retriever.

Two retrieval settings are compared:

### OneR

OneR performs a single retrieval using the original question.

The number of retrieved paragraphs `K` is tuned on the development set:

```text
K ∈ {5, 7, 9, 11, 13, 15}
```

### IRCoT

IRCoT performs iterative retrieval.

At each retrieval step, the current reasoning sentence is used as the query.

The number of paragraphs retrieved per step is tuned from:

```text
K ∈ {2, 4, 6, 8}
```

The total number of retrieved paragraphs is capped at 15 during evaluation.

### Language models

The paper evaluates:

- GPT-3 `code-davinci-002` (~175B parameters)
- Flan-T5 Base (~0.2B)
- Flan-T5 Large (~0.7B)
- Flan-T5 XL (~3B)
- Flan-T5 XXL (~11B)

For each model size, the same LM family/model is used for the reasoning component and the QA reader.

### Datasets

IRCoT is evaluated on:

- HotpotQA
- 2WikiMultihopQA
- MuSiQue
- IIRC

The associated Wikipedia corpora are used as the retrieval corpus.

For HotpotQA, the associated Wikipedia corpus is directly available.

For 2WikiMultihopQA and MuSiQue, the authors convert the original reading-comprehension setting into an open-domain retrieval setting by combining supporting and non-supporting paragraphs from the questions.

For IIRC, the corpus is constructed from paragraphs from Wikipedia pages appearing in the dataset.

The supporting paragraphs are mixed into the retrieval corpus rather than being directly provided to the retriever.

---

## 4. How to Make Query

Query construction is one of the central ideas of IRCoT.

### Initial query

The first retrieval uses the original question:

```text
Query₀ = Question
```

### Subsequent queries

After retrieval, the LM reasons over the question, previously retrieved paragraphs, and previous reasoning.

The first generated reasoning sentence becomes the next retrieval query:

```text
Queryₜ = First CoT sentenceₜ
```

Therefore:

```text
Question
   ↓
Retrieve
   ↓
Reasoning sentence₁
   ↓
Retrieve using sentence₁
   ↓
Reasoning sentence₂
   ↓
Retrieve using sentence₂
   ↓
...
```

This creates a feedback loop:

```text
Evidence
   ↓
Reasoning
   ↓
New query
   ↓
New evidence
   ↓
More reasoning
```

### Few-shot demonstrations

IRCoT does not train a dedicated query-generation model.

Instead, few-shot demonstrations are used to teach the LM how to perform the desired reasoning behavior.

The authors manually annotate 20 questions per dataset with Chain-of-Thought reasoning.

They construct three demonstration sets by sampling 15 questions per dataset.

For IRCoT reasoning demonstrations, the prompts contain:

- gold supporting paragraphs
- `M ∈ {1, 2, 3}` randomly selected distractor paragraphs
- shuffled/concatenated context

At test time, as many demonstrations as possible are packed into the available context.

The GPT-3 setting uses an approximately 8K-token context, while the Flan-T5 experiments use a shorter context of around 6K tokens due to memory limitations.

---

## 5. Training

IRCoT does **not require additional training of the retrieval mechanism**.

The method is primarily an inference-time framework built around:

- a language model,
- BM25 retrieval,
- few-shot demonstrations.

There is no dedicated retrieval-decision supervision.

There is also no separately trained query-generation model.

The language model is prompted to reason step-by-step and the generated reasoning sentence is reused as the next retrieval query.

For Flan-T5, the prompts are adapted to an instruction-following format.

The paper uses formats such as:

```text
Q: Answer the following question by reasoning step-by-step. <question>
```

for Chain-of-Thought reasoning, and:

```text
Q: Answer the following question. <question>
```

for direct answering.

The same LM family/model is used for both reasoning and final QA.

---

## 6. Evaluation

IRCoT is evaluated from both **retrieval** and **question-answering** perspectives.

### 6.1 Retrieval evaluation

The retrieval evaluation measures whether the gold supporting paragraphs are included in the retrieved set.

The maximum number of retrieved paragraphs is capped at 15.

The authors measure recall of gold paragraphs.

They do not use MAP/DCG as the primary retrieval metrics because the retrieved documents are not treated as a conventional ranked list for a single final retrieval operation.

The number of retrieved documents can also vary depending on how many reasoning steps are performed.

The retrieval hyperparameters are tuned on the development set and then fixed for test evaluation.

---

### 6.2 QA evaluation

The main systems compared are:

- **NoR:** No retrieval
- **OneR:** One-step retrieval using the question
- **IRCoT:** Iterative reasoning-guided retrieval

The general result is:

```text
IRCoT > OneR > NoR
```

although the magnitude of the improvement varies across datasets.

### GPT-3 CoT QA F1

| System | HotpotQA | 2Wiki | MuSiQue | IIRC |
| ------ | -------: | ----: | ------: | ---: |
| NoR    |     47.5 |  41.2 |    25.2 | 52.1 |
| OneR   |     53.6 |  54.8 |    29.4 | 49.8 |
| IRCoT  |     60.7 |  68.0 |    36.5 | 49.9 |

IRCoT substantially improves over OneR on HotpotQA, 2WikiMultihopQA, and MuSiQue.

IIRC is an exception: although IRCoT improves retrieval recall, the final QA improvement over OneR is very small.

---

### 6.3 Retrieval improvement

For Flan-T5-XXL, the retrieval recall improvement of IRCoT over OneR is approximately:

| Dataset         | IRCoT improvement |
| --------------- | ----------------: |
| HotpotQA        |              +7.9 |
| 2WikiMultihopQA |             +14.3 |
| MuSiQue         |              +3.5 |
| IIRC            |             +10.2 |

For GPT-3:

| Dataset         | IRCoT improvement |
| --------------- | ----------------: |
| HotpotQA        |             +11.3 |
| 2WikiMultihopQA |             +22.6 |
| MuSiQue         |             +12.5 |
| IIRC            |             +21.2 |

This demonstrates that intermediate reasoning can substantially improve the ability to discover relevant documents in multi-hop settings.

---

### 6.4 Factuality

The paper also evaluates factual errors in the generated reasoning.

They randomly sample 40 questions from each of the four datasets.

A factual error is counted when any fact stated before the final `"answer is:"` is incorrect.

IRCoT produces fewer factual errors than the corresponding baselines.

The paper reports approximately:

- 50% reduction in factual errors over OneR on HotpotQA
- 40% reduction over OneR on 2WikiMultihopQA

Thus, retrieval guided by intermediate reasoning does not only improve document recall; it can also reduce factual errors during reasoning.

---

### 6.5 Scaling

The authors evaluate IRCoT across different Flan-T5 model sizes.

IRCoT consistently outperforms OneR across model sizes.

The improvement generally becomes larger as the model size increases.

An important observation is that **IRCoT with a 3B-parameter Flan-T5 model can outperform OneR/NoR using the much larger 175B GPT-3 model on all evaluated datasets**.

This suggests that retrieval strategy can compensate for some of the limitations of simply scaling the language model.

---

### 6.6 Out-of-distribution demonstrations

The paper also evaluates whether demonstrations from one dataset can be used when solving another dataset.

The general finding is that IRCoT remains effective under these cross-dataset demonstration settings.

For most dataset pairs:

```text
IRCoT retrieval > OneR retrieval
IRCoT QA > OneR QA > NoR QA
```

This suggests that the reasoning-guided retrieval behavior is not completely tied to the exact dataset on which the demonstrations were constructed.

---

### 6.7 Comparison with other methods

The paper compares IRCoT with methods including:

- ReAct
- Self-Ask
- DecomP
- RECITE
- Internet-Augmented QA
- DSP

These comparisons are not always perfectly controlled because the compared methods may use different:

- APIs
- language models
- retrievers
- corpora
- evaluation subsets

Nevertheless, IRCoT is reported as highly competitive, including achieving strong results on MuSiQue and near state-of-the-art performance on HotpotQA and 2WikiMultihopQA at the time.

---

## 7. Ablation

### 7.1 Number of paragraphs per retrieval step

The authors study:

```text
K ∈ {2, 4, 6, 8}
```

where `K` is the number of paragraphs retrieved at each step.

This tests the trade-off between retrieving more evidence at each reasoning step and maintaining an efficient iterative retrieval process.

---

### 7.2 Number of distractor paragraphs

The few-shot reasoning demonstrations are tested with:

```text
M ∈ {1, 2, 3}
```

random distractor paragraphs.

This evaluates whether the model can reason and retrieve correctly when the context contains irrelevant information.

---

### 7.3 Direct vs CoT reader

The final QA reader is evaluated with both:

- Direct prompting
- Chain-of-Thought prompting

The effect depends on the underlying LM.

For example:

- Flan-T5-XXL performs better with Direct prompting.
- GPT-3 performs better with CoT prompting.

Despite these differences, the overall retrieval comparison remains favorable to IRCoT:

```text
IRCoT > OneR > NoR
```

---

### 7.4 OOD demonstrations

Cross-dataset demonstrations are used to test whether the method depends strongly on demonstrations from the same dataset.

The results indicate that IRCoT can generalize its reasoning-guided retrieval behavior across datasets.

---

## 8. Limitation

### 8.1 Limitations mentioned by the paper

#### 1. Dependence on Chain-of-Thought capability

IRCoT relies on the LM being capable of producing useful intermediate reasoning.

At the time of the paper, strong zero-shot/few-shot CoT behavior was primarily associated with relatively large language models.

The authors expect this limitation to become less significant as smaller models improve.

#### 2. Long context

The context grows because the model needs to account for:

- the original question,
- previous reasoning,
- retrieved paragraphs,
- few-shot demonstrations.

This creates a substantial context-length requirement.

The GPT-3 experiments use an approximately 8K context, while the Flan-T5 experiments use around 6K tokens due to GPU memory constraints.

#### 3. Computational cost

IRCoT requires repeated cycles of:

```text
LM generation → retrieval → LM generation → retrieval → ...
```

Therefore, the iterative nature of the method introduces additional inference computation compared with a single retrieval step.

#### 4. Dependence on deprecated model

Some experiments rely on OpenAI's `code-davinci-002`, which has since been deprecated.

This makes exact reproduction of those experiments more difficult.

#### 5. Retrieval does not eliminate hallucination

Although IRCoT reduces factual errors, retrieval and reasoning do not completely eliminate hallucination or factual errors.

#### 6. Future direction: adaptive retrieval timing

The paper identifies an important future direction: models should dynamically determine **when to retrieve additional information and when to continue reasoning using information that is already available**.

This is closely related to the motivation behind ART, although IRCoT itself does not formulate retrieval timing as a learned decision problem.

---

### 8.2 My observations

#### 1. Context can grow significantly

In addition to the limitations explicitly discussed by the paper, the iterative structure naturally causes context accumulation:

```text
Few-shot demonstrations
        +
Question
        +
Reasoning₁
        +
Retrieved documents₁
        +
Reasoning₂
        +
Retrieved documents₂
        +
...
```

Therefore, both the number of demonstrations and the accumulated reasoning/documents can increase the context substantially.

---

#### 2. Repeated reasoning can be computationally expensive

Each retrieval changes the available context.

Therefore, the model repeatedly processes an evolving context:

```text
Context₁ → Generate
Context₂ → Generate
Context₃ → Generate
...
```

This does not naturally provide the same level of KV-cache reuse that is possible during ordinary continuous autoregressive generation, because retrieval introduces new context and requires another reasoning/generation step.

Thus, the iterative architecture can have considerable inference overhead.

---

#### 3. Retrieval is coupled to reasoning steps

A more important observation for ART is that IRCoT effectively couples retrieval with the reasoning process.

The paper explicitly generates a reasoning sentence, keeps the first sentence, and uses it as the query for the next retrieval step.

Therefore, the procedure is approximately:

```text
Reasoning sentence
       ↓
Retrieve
       ↓
Reasoning sentence
       ↓
Retrieve
       ↓
...
```

However, **not every reasoning sentence necessarily requires new external information**.

For example:

```text
S₁: Identify a relevant entity.
    → Retrieval may be useful.

S₂: Use the retrieved information to identify another entity.
    → Retrieval may be useful.

S₃: Combine the already retrieved facts to derive the answer.
    → New retrieval may be unnecessary.
```

Some reasoning steps may simply:

- combine previously retrieved facts,
- perform logical inference,
- reformulate an existing fact,
- move toward the final answer.

Therefore, retrieving after every reasoning step can potentially cause unnecessary retrieval.

This motivates a distinction:

```text
"What should I retrieve?"
        ↓
IRCoT

"Do I need to retrieve at all?"
        ↓
ART
```

A retrieval budget can limit the **maximum amount of retrieval**, but it does not necessarily learn **whether retrieval is needed at each reasoning step**.

---

## 9. Relate to ART

IRCoT and ART share the idea that retrieval should interact with the model's reasoning process rather than being treated as a completely separate preprocessing stage.

However, their main objectives are different.

| Aspect               | IRCoT                             | ART                                    |
| -------------------- | --------------------------------- | -------------------------------------- |
| Main question        | What should be retrieved next?    | When should retrieval happen?          |
| Retrieval signal     | Generated CoT sentence            | Learned retrieval decision             |
| Query                | Question → CoT sentence           | Retriever query + current model state  |
| Retrieval timing     | Procedurally coupled to reasoning | Adaptively decided                     |
| Decision supervision | No retrieval-decision labels      | No retrieval-decision labels           |
| Reasoning/retrieval  | Interleaved                       | Interleaved                            |
| LLM training         | Few-shot prompting                | LLM frozen; decision component trained |
| Final generation     | Separate QA reader                | Continuous reasoning/generation        |
| Main mechanism       | Reasoning-guided retrieval        | Attention-based retrieval timing       |

### Key conceptual distinction

IRCoT mainly addresses:

> **What information should be retrieved next based on the current reasoning state?**

ART addresses:

> **Does the model need external information at this point in reasoning?**

This distinction is important because a reasoning sentence can provide a useful retrieval query without necessarily implying that retrieval is required.

Therefore, ART can be viewed as addressing a different axis of the reasoning–retrieval interaction.

---

## 10. What ART can learn

### 1. Reasoning sentence can be used as a retrieval query

IRCoT demonstrates that a reasoning sentence generated during reasoning can be used directly as the query for subsequent retrieval.

ART can experiment with using the reasoning sentence as the query for a retriever such as **E5**.

However, this should be **measured experimentally** rather than assumed to be effective.

An important experiment is therefore to compare:

```text
Question → E5 retrieval
```

against:

```text
Reasoning sentence → E5 retrieval
```

and evaluate whether reasoning-based queries provide better retrieval quality.

---

### 2. A lightweight prompt can format the reasoning as CoT

IRCoT uses multiple few-shot demonstrations to guide the LM toward the desired reasoning format.

ART can explore a lighter alternative:

```text
Simple prompt
     ↓
CoT-style reasoning
     ↓
ART decision
```

Instead of using many demonstrations, a lightweight prompt can instruct the model to produce reasoning in a consistent CoT format.

This may reduce the context overhead introduced by few-shot demonstrations while still providing a structured reasoning signal for ART.

---

### 3. Optional: separate retrieval and final generation

The current ART design is based on **interleaved retrieval and generation**:

```text
Reason
  ↓
Retrieve
  ↓
Continue reasoning
  ↓
Retrieve
  ↓
Continue reasoning
  ↓
...
```

As an optional experiment, after ART finishes its retrieval process, all retrieved documents can be collected and combined with the original question.

The model can then generate the final answer once:

```text
ART retrieval process
        ↓
Collect all retrieved documents
        ↓
Question + all documents
        ↓
Generate final answer once
```

This is conceptually similar to the way IRCoT separates:

```text
Iterative retrieval
        ↓
Final QA reader
```

This should be treated as an **optional variant**, not the core ART architecture.

It can nevertheless provide a useful experiment to determine whether ART's benefit comes primarily from:

1. learning **when to retrieve**, or
2. continuously interleaving retrieval with generation.

---

## Summary

IRCoT demonstrates that reasoning and retrieval can be tightly coupled:

```text
Reason → Retrieve → Reason → Retrieve → ...
```

Its key contribution is using intermediate reasoning to guide the next retrieval query.

For ART, the most important distinction is that **retrieval need not necessarily follow every reasoning step**.

This motivates the decomposition:

```text
WHAT to retrieve?
    ↓
Reasoning-guided query generation
    ↓
IRCoT

WHEN to retrieve?
    ↓
Adaptive retrieval decision
    ↓
ART
```

IRCoT therefore provides both a strong conceptual foundation for ART and a useful source of experimental ideas around reasoning-based queries, CoT prompting, and alternative retrieval/generation architectures.
