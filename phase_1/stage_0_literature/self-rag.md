# Self-RAG — Learning to Retrieve, Generate and Critique through Self-Reflection

**Paper:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi

**Title:** _Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection_

**Venue:** ICLR 2024 — Oral

**Paper:** https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf

**Code:** https://github.com/akariasai/self-rag

---

## 1. Problem

Large Language Models can answer many questions using **parametric knowledge** stored in their parameters, but they still suffer from several problems:

- factual errors;
- lack of specialized knowledge or external information;
- hallucination;
- outputs that are not supported by retrieved evidence;
- retrieving too many passages or retrieving when retrieval is unnecessary;
- retrieved passages may be irrelevant or may not actually support the generated answer.

A traditional RAG pipeline typically follows:

```text
Question
   ↓
Retrieve K documents
   ↓
Retrieved documents
   ↓
Generate answer
```

The problem is that this pipeline usually **retrieves a fixed number of passages**, regardless of whether the query actually requires external information.

For example:

```text
Question A
   ↓
Retrieve
   ↓
Generate
```

Question A may be completely answerable using parametric knowledge.

Similarly:

```text
Question B
   ↓
Retrieve
   ↓
Generate
```

The retrieved passages may be irrelevant or insufficient to support the answer.

Therefore, the problem is not only:

> **What should be retrieved?**

but also:

> **When should the model retrieve, and how can it determine whether the retrieved information actually helps?**

Self-RAG addresses this problem by allowing the LM to **retrieve on demand**, while also evaluating retrieved passages and its own generation using **reflection tokens**.

Instead of:

```text
Question → Retrieve → Generate
```

Self-RAG uses:

```text
Question
   ↓
Generate / Reason
   ↓
Decide whether to retrieve
   ↓
Retrieve if necessary
   ↓
Evaluate evidence
   ↓
Generate
   ↓
Evaluate generation
   ↓
Continue / Retrieve again
```

---

## 2. Core Idea

The core idea of Self-RAG is to **train an LM to generate reflection tokens**, which are used to control retrieval and evaluate the generation process.

Self-RAG uses four groups of reflection tokens:

```text
Retrieve:
    {Yes, No, Continue}

ISREL:
    {Relevant, Irrelevant}

ISSUP:
    {Fully supported,
     Partially supported,
     No support}

ISUSE:
    {5, 4, 3, 2, 1}
```

The paper defines them as:

| Reflection | Meaning                                                  |
| ---------- | -------------------------------------------------------- |
| Retrieve   | Whether external retrieval is needed                     |
| ISREL      | Whether the retrieved passage is relevant to the input   |
| ISSUP      | Whether the output is supported by the retrieved passage |
| ISUSE      | How useful the generated output is                       |

These reflection tokens are added to the LM vocabulary and learned through **next-token prediction**.

### Retrieval decision

For retrieval timing, the most important reflection tokens are:

```text
Retrieve = Yes
Retrieve = No
Retrieve = Continue
```

The paper makes retrieval decisions at the **segment level**.

In the experiments, one sentence is treated as one segment, although the framework can support other segment units.

Therefore, retrieval can occur multiple times during generation:

```text
Segment 1 → No
Segment 2 → Yes → Retrieve
Segment 3 → No
Segment 4 → Yes → Retrieve
...
```

### Retrieval score

Self-RAG can use the probability of the retrieval reflection token to obtain a retrieval confidence score.

Conceptually:

```math
s(\text{Retrieve}) =
\frac{P(\text{Yes})}
{P(\text{Yes}) + P(\text{No})}
```

and retrieval is triggered when:

```math
s(\text{Retrieve}) > \delta
```

Thus, retrieval does not necessarily rely only on:

```text
argmax(Yes, No)
```

but can instead be controlled using a threshold. The paper describes this adaptive retrieval mechanism in Section 3.3 and Appendix A.4.

### Retrieval is not the whole Self-RAG system

Self-RAG does not stop at:

```text
Should we retrieve?
```

After retrieval, it further evaluates:

```text
Retrieved document
       ↓
    ISREL?
       ↓
Relevant / Irrelevant
       ↓
    ISSUP?
       ↓
Supported / Partially / No support
       ↓
    ISUSE?
       ↓
Useful?
```

These reflection scores are then used in **segment-level beam search** to select better continuations.

---

## 3. Model, Retriever

### Language models

Self-RAG uses two models during its training pipeline:

```text
Critic C
   ↓
Predict reflection tokens

Generator M
   ↓
Generate text + reflection tokens
```

The Critic is fine-tuned to predict reflection tokens.

The trained Critic is then used offline to create training data for the Generator.

The final Generator is fine-tuned to **generate both text and reflection tokens**, so a separate Critic is not required during inference.

The paper uses models from the Llama 2 family, including Self-RAG 7B and 13B models.

The official repository also describes the pipeline as:

```text
1. Critic Data Creation
2. Critic Training
3. Generator Data Creation
4. Generator Training
```

with Llama-2-7B used in the official training setup.

### Retriever

The default retriever used by Self-RAG is:

**Contriever-MS MARCO**

It is a **dense retriever**.

The pipeline is:

```text
Query
  ↓
Contriever-MS MARCO
  ↓
Dense embedding
  ↓
Similarity search
  ↓
Top-K passages
```

The official implementation uses:

```text
facebook/contriever-msmarco
```

and supports retrieval over Wikipedia passage embeddings.

The paper primarily uses English Wikipedia as the retrieval corpus in its main experimental settings.

Different tasks may use different corpus versions or configurations.

### Important distinction

The retriever is **not jointly trained end-to-end with the Generator in Self-RAG**.

It is used as an external retrieval component.

This is important because the quality of the off-the-shelf retriever can become a bottleneck for the overall system.

---

## 4. How to Make Query

Query construction in Self-RAG is different from IRCoT.

### Initial retrieval

During generator-data construction, when the Critic determines that retrieval is useful, the system performs retrieval based on the available input and generation context.

At inference time, Algorithm 1 describes retrieval based on:

```text
Original input
+
preceding generation
```

More formally, retrieval at timestep `t` is conditioned on the input `x` and preceding generated tokens `y^{<t}`.

Therefore, the retrieval query is **not simply the current reasoning sentence**.

### Query construction during generation

Conceptually:

```text
Input
+
previous generation
+
current generation state
        ↓
Critic
        ↓
Retrieve?
```

If retrieval is required:

```text
Input + generation context
        ↓
Retriever
        ↓
Retrieved passages
```

### Compared with IRCoT

IRCoT:

```text
Question
   ↓
Retrieve

Reasoning sentence₁
   ↓
Retrieve

Reasoning sentence₂
   ↓
Retrieve
```

The generated reasoning sentence is directly used as the query for the next retrieval step.

Self-RAG:

```text
Question
+
Current generation context
        ↓
Retrieve
```

Therefore, Self-RAG preserves more task-level and generation-level context when constructing the retrieval query.

### Important distinction

The difference can be viewed as:

```text
IRCoT:

"What am I reasoning about right now?"
        ↓
Current CoT sentence
        ↓
Retriever
```

while:

```text
Self-RAG:

"What is the original task,
and what am I generating right now?"
        ↓
Question + generation context
        ↓
Retriever
```

This suggests that a retrieval query may benefit from both **task-level context** and the **current generation state**, rather than relying only on an isolated reasoning sentence.

---

## 5. Training

Self-RAG has a considerably more complex training pipeline than IRCoT.

Overall:

```text
Seed data
   ↓
GPT-4
   ↓
Reflection labels
   ↓
Train Critic
   ↓
Critic predicts reflection
   ↓
Retriever
   ↓
Create Generator training data
   ↓
Train Generator
```

### Step 1 — Generate reflection labels

Manually annotating reflection tokens for every segment and multiple retrieved passages would be expensive.

Therefore, Self-RAG uses GPT-4 to generate supervised reflection labels.

The reflection groups are generated separately:

```text
Retrieve
ISREL
ISSUP
ISUSE
```

The paper reports approximately **4K–20K supervised instances for each reflection type**, which are then combined into training data for the Critic.

The official repository provides approximately **150K training instances**.

### Step 2 — Train Critic

The Critic is initialized from a pretrained LM.

It is trained using a standard conditional language-modeling objective:

```math
\max_C
\mathbb{E}_{(x,y,r)\sim D_{critic}}
\log p_C(r|x,y)
```

where:

```text
x = input
y = output
r = reflection token
```

The Critic therefore learns to predict:

```text
Input + Output
       ↓
Reflection token
```

For example:

```text
Question + Answer
       ↓
Retrieve = Yes
```

or:

```text
Question + Passage + Answer
       ↓
ISREL = Relevant
ISSUP = Fully supported
```

### Step 3 — Create Generator training data

After the Critic has been trained:

```text
Original training pair
        ↓
Critic
        ↓
Retrieve decision
        ↓
Retriever
        ↓
Retrieved passages
        ↓
Critic evaluates passages
        ↓
Reflection tokens
        ↓
D_gen
```

For each segment:

#### If Retrieve = No

```text
Retrieve = No
       ↓
Do not retrieve new passages
       ↓
Continue generation
```

#### If Retrieve = Yes

```text
Retrieve = Yes
       ↓
Retrieve K passages
       ↓
ISREL
       ↓
Relevant passages
       ↓
ISSUP
       ↓
Supported passage/output
       ↓
ISUSE
```

The paper describes this process as creating an augmented corpus that simulates the inference-time procedure.

### Step 4 — Train Generator

The Generator is trained on `D_gen` using a standard next-token prediction objective:

```math
\max_M
\mathbb{E}_{(x,y,r)\sim D_{gen}}
\log p_M(y,r|x)
```

The important point is:

```text
Generator learns:
    output tokens
    +
    reflection tokens
```

Retrieved passages are provided as context but **are masked from the Generator's LM loss**.

Therefore, at inference time, the Generator can operate without a separate Critic:

```text
Input
 ↓
Generator
 ↓
Retrieve token
 ↓
Retriever
 ↓
Reflection tokens
 ↓
Generation
 ↓
Reflection tokens
...
```

### Important point for ART

Self-RAG **has direct supervision for the retrieval decision**:

```text
GPT-4
  ↓
Retrieve = Yes / No / Continue
  ↓
Critic
  ↓
Generator
```

In contrast, ART uses:

```text
LM loss + retrieval cost
        ↓
latent decision d
```

ART does not require ground-truth retrieval-decision labels.

This represents a fundamental difference in the learning paradigm.

---

## 6. Evaluation

Self-RAG is evaluated across several types of tasks:

```text
Open-domain QA
Fact checking
Long-form generation
Citation-related tasks
Closed-set QA
```

The main datasets include:

- PopQA
- TriviaQA
- ASQA
- PubHealth
- ARC-Challenge
- FEVER
- FactScore
- ALCE
- biomedical/biographical generation tasks

The paper evaluates Self-RAG on multiple downstream tasks, including zero-shot settings.

### 6.1 Short-form QA / closed-set tasks

The tasks include:

```text
PopQA
TriviaQA
PubHealth
ARC-Challenge
```

Metrics depend on the task:

- Exact Match / F1 for QA;
- Accuracy for closed-set tasks.

The official implementation provides several retrieval modes:

```text
adaptive_retrieval
no_retrieval
always_retrieve
```

which allow comparison of different retrieval strategies.

### 6.2 Long-form generation

Self-RAG is evaluated on:

```text
ASQA
FactScore
```

as well as citation- and factuality-related benchmarks.

The evaluation considers:

- citation precision;
- answer quality;
- factuality;
- FactScore;
- human evaluation.

The official repository uses ALCE for ASQA evaluation and FactScore for biographical generation.

### 6.3 Adaptive retrieval

Self-RAG allows retrieval frequency to be controlled using a threshold:

```math
s(\text{Retrieve}) =
\frac{P(\text{Yes})}
{P(\text{Yes})+P(\text{No})}
```

and:

```math
s(\text{Retrieve}) > \delta
```

→ retrieve.

The official implementation uses a default:

```text
threshold = 0.2
```

in its evaluation scripts.

### 6.4 Segment-level beam search

When retrieval is triggered:

```text
Retrieve K passages
        ↓
Generate K candidate continuations
        ↓
ISREL
ISSUP
ISUSE
        ↓
Score each candidate
        ↓
Beam search
```

The segment score combines:

```text
Generation probability
+
weighted reflection scores
```

The corresponding weights include:

```text
w_rel
w_sup
w_use
```

The official implementation uses defaults such as:

```text
w_rel = 1.0
w_sup = 1.0
w_use = 0.5
beam_width = 2
max_depth = 6
```

### 6.5 Parametric vs non-parametric memory

An important analysis in the paper examines whether correct answers are actually contained in the retrieved evidence.

Conceptually:

```text
Correct answer
      ↓
Does retrieved passage contain answer?
      │
   ┌──┴──┐
  Yes    No
   ↓      ↓
Evidence  Parametric
grounded  memory may contribute
```

The paper reports that only around **2% of correct predictions** in the relevant analysis had the gold answer absent from the provided passages, a substantially lower rate than for the instruction-tuned baselines considered.

This suggests that Self-RAG tends to produce outputs that are more strongly **grounded in retrieved evidence**.

---

## 7. Ablation

### 7.1 Training data scale

The paper trains Self-RAG 7B variants with different amounts of training data, including:

```text
5K
10K
20K
50K
150K
```

In general:

```text
Training data ↑
       ↓
Performance ↑
```

The improvement is particularly visible on some tasks such as PopQA and ASQA citation precision.

However, the improvement does not continue to increase strongly in every setting; some results suggest saturation as the training data grows from 50K to 150K.

### 7.2 Reflection prediction accuracy

The Critic is evaluated by comparing its predictions against GPT-4-generated labels.

Reported results are:

| Reflection | Accuracy |
| ---------- | -------: |
| Retrieve   |    93.8% |
| ISSUP      |    93.5% |
| ISREL      |    80.2% |
| ISUSE      |    73.5% |

Retrieve has the highest accuracy.

ISUSE is more difficult, partly because it involves finer-grained utility judgments.

### 7.3 Human evaluation

Human annotators evaluate:

```text
Output
 ↓
Plausible?
 ↓
Supported?
 ↓
ISREL correct?
 ↓
ISSUP correct?
```

The results show that Self-RAG generally produces plausible outputs and is able to ground them in evidence when the retrieved passages are relevant.

### 7.4 Retrieval frequency / inference control

Self-RAG also allows its behavior to be modified through inference-time controls.

For example:

```text
Higher retrieval threshold
        ↓
Less retrieval
```

while:

```text
Lower retrieval threshold
        ↓
More retrieval
```

The reflection-score weights can also be adjusted to prioritize:

```text
Relevance
Support
Overall utility
```

This demonstrates that Self-RAG does not simply learn a fixed retrieval policy; it also provides mechanisms for **controlling retrieval behavior at inference time**.

---

## 8. Limitation

### 8.1 Limitations mentioned by the paper

#### 1. Complex training pipeline

Self-RAG requires:

```text
Seed data
   ↓
GPT-4 labeling
   ↓
Critic training
   ↓
Retriever
   ↓
Generator data creation
   ↓
Generator training
```

Therefore, constructing the training data is relatively complex.

The paper uses GPT-4 to generate reflection supervision and then distills this supervision into the Critic, reducing the need to directly rely on GPT-4 in subsequent stages.

#### 2. Dependence on reflection supervision

Self-RAG's retrieval behavior is learned from reflection labels.

Therefore:

```text
GPT-4 labels
      ↓
Critic
      ↓
Generator behavior
```

The quality of the reflection supervision can therefore affect the resulting model behavior.

#### 3. Off-the-shelf retriever

Self-RAG uses Contriever-MS MARCO.

The retriever is not jointly optimized with the Generator for the final generation objective.

Therefore, retrieval quality can become a bottleneck.

#### 4. Retrieved evidence is not always useful

Retrieved passages can be:

```text
Relevant
Irrelevant
Fully supporting
Partially supporting
No support
```

Therefore, Self-RAG requires additional mechanisms:

```text
ISREL
ISSUP
ISUSE
```

to evaluate evidence and generated output.

This makes the system more capable, but also more complex.

#### 5. Retrieval still has computational cost

Adaptive retrieval can reduce unnecessary retrieval, but Self-RAG **does not explicitly optimize retrieval cost through a loss term** such as:

```math
\text{LM loss}+\lambda\cdot\text{retrieval cost}
```

as in the ART formulation.

Therefore, Self-RAG's main objective remains focused on **generation quality, factuality, attribution, and controllability**, rather than explicit quality–retrieval-cost optimization.

---

### 8.2 My observations

#### 1. Self-RAG solves several problems simultaneously

Self-RAG does not only learn:

```text
WHEN?
```

It simultaneously addresses:

```text
WHEN?
 ↓
IS THE DOCUMENT RELEVANT?
 ↓
IS THE OUTPUT SUPPORTED?
 ↓
IS THE OUTPUT USEFUL?
```

Therefore, its architecture is relatively broad if the research question is specifically about retrieval timing.

---

#### 2. Retrieval decision and retrieval quality are different problems

Self-RAG highlights the distinction between:

```text
Problem A:
Do we need retrieval?
        ↓
WHEN TO RETRIEVE

Problem B:
Is the retrieved evidence useful?
        ↓
RETRIEVAL QUALITY
```

These two problems should not be conflated.

In particular:

```text
Retrieve = Yes
```

does not imply:

```text
Retrieved document = Useful
```

---

#### 3. Retrieval frequency is not the same as retrieval necessity

Self-RAG allows retrieval frequency to be controlled through a threshold.

However:

```text
Threshold
   ↓
Retrieval frequency
```

does not necessarily imply:

```text
Threshold
   ↓
Optimal retrieval cost
```

This distinction is important for ART.

---

#### 4. Self-RAG can retrieve multiple times during generation

This is particularly relevant to retrieval timing.

The generation process can look like:

```text
Generate
   ↓
Retrieve
   ↓
Generate
   ↓
No Retrieve
   ↓
Generate
   ↓
Retrieve
   ↓
Generate
```

Therefore, retrieval timing is effectively a **sequential decision problem**.

---

## 9. Relate to ART

Self-RAG and ART have a clear area of overlap:

```text
Traditional RAG:

Question
   ↓
Retrieve
   ↓
Generate
```

whereas:

```text
Self-RAG / ART:

Generate
   ↓
Should retrieve?
   ↓
Retrieve if necessary
   ↓
Continue generation
```

However, the research objectives of the two approaches are substantially different.

| Aspect                   | Self-RAG                           | ART                                     |
| ------------------------ | ---------------------------------- | --------------------------------------- |
| Main question            | When to retrieve + how to critique | **When to retrieve**                    |
| Decision                 | Retrieve token                     | Continuous `d`                          |
| Decision values          | Yes / No / Continue                | `d ∈ [0,1]`                             |
| Decision supervision     | **GPT-4 reflection labels**        | **None**                                |
| Learning signal          | Supervised next-token prediction   | **LM loss + retrieval cost**            |
| Critic                   | Yes                                | Decision module                         |
| Retriever                | Contriever-MS MARCO                | Retriever chosen by ART experiments     |
| Retrieval quality        | ISREL + ISSUP + ISUSE              | Not in Phase 1                          |
| Query                    | Input + generation context         | Current model state + query formulation |
| LLM                      | Fine-tuned                         | **Frozen**                              |
| Retrieval cost objective | Not an explicit main objective     | **Explicit**                            |
| RL                       | No                                 | No                                      |
| Retrieval timing         | Adaptive                           | **Adaptive**                            |
| Scope                    | Retrieval + generation + critique  | **Retrieval timing**                    |

### Key conceptual distinction

Self-RAG asks:

> **Can an LM learn to retrieve on demand and critique retrieved evidence and its own generation using supervised reflection tokens?**

ART asks a narrower question:

> **Can retrieval timing be learned directly from the downstream generation objective and retrieval cost without explicit retrieval-decision supervision or reinforcement learning?**

Therefore, ART **should not be positioned simply as a "simplified version of Self-RAG."**

A more accurate positioning is:

```text
Self-RAG

GPT-4 reflection supervision
          ↓
Retrieve token
          ↓
Retrieval timing
          ↓
ISREL / ISSUP / ISUSE
          ↓
Controlled generation
```

whereas:

```text
ART

LM loss + retrieval cost
          ↓
latent decision d
          ↓
Retrieval timing
```

ART deliberately **isolates retrieval timing** as its primary research question.

---

## 10. What ART can learn

### 1. Retrieval timing should be treated as a sequential decision

Self-RAG demonstrates that retrieval does not have to be a single decision for the entire question.

It can look like:

```text
Segment 1 → No
Segment 2 → Yes → Retrieve
Segment 3 → No
Segment 4 → Yes → Retrieve
...
```

This supports the idea of making multiple retrieval decisions throughout generation:

```text
0/32
1/32
2/32
...
32/32
```

rather than:

```text
Question
   ↓
One retrieval decision
```

ART can therefore investigate:

> **At which point during generation does external information actually become worth retrieving?**

---

### 2. Retrieval queries should contain sufficient context

Self-RAG suggests that retrieval does not necessarily need to rely on an isolated reasoning sentence.

A query can incorporate:

```text
Original question
+
Current generation segment
+
Previous context
```

Therefore, ART can experimentally compare:

```text
Question
   ↓
Retriever
```

vs.

```text
Current reasoning segment
   ↓
Retriever
```

vs.

```text
Question
+
Current reasoning segment
   ↓
Retriever
```

and evaluate retrieval quality empirically.

This also combines insights from Self-RAG and IRCoT:

```text
IRCoT:
Reasoning sentence → Query

Self-RAG:
Question + generation context → Query
```

ART does not necessarily need to commit to one formulation immediately; query construction can instead be treated as an experimental variable.

---

### 3. Retrieve ≠ automatically useful

This is perhaps the most important insight from Self-RAG for ART.

Self-RAG demonstrates that a retrieved passage can be:

```text
Relevant
Irrelevant
Fully supporting
Partially supporting
No support
```

Therefore:

```math
\text{Retrieve} \neq \text{Automatically useful}
```

This supports the interpretation:

```math
\text{ShouldRetrieve}
\approx
\text{Expected benefit of evidence}
-
\text{Retrieval cost}
```

rather than:

```math
\text{ShouldRetrieve}
=
\text{Model is uncertain}
```

This provides strong motivation for ART.

If the evidence does not improve generation enough to compensate for the retrieval cost, retrieval should not occur.

---

### 4. Retrieval-quality prediction does not need to be included in Phase 1

Self-RAG has the following architecture:

```text
WHEN?
 ↓
Retrieve
 ↓
ISREL?
 ↓
Relevant?
 ↓
ISSUP?
 ↓
Supported?
 ↓
ISUSE?
 ↓
Useful?
```

ART Phase 1 only needs:

```text
WHEN?
 ↓
Retrieve / No Retrieve
```

Therefore, ART should not immediately introduce:

```text
ISREL
ISSUP
ISUSE
```

Doing so would shift the research question from:

> **When should we retrieve?**

to:

> **When should we retrieve, what should we retrieve, and how should we evaluate the evidence?**

This would significantly increase the scope of Phase 1.

---

### 5. Self-RAG's retrieval threshold is relevant to ART

Self-RAG uses:

```math
s(\text{Retrieve}) =
\frac{P(\text{Yes})}
{P(\text{Yes})+P(\text{No})}
```

and a threshold to control retrieval.

ART similarly uses:

```math
d = \sigma(z)
```

and:

```math
d \geq \tau \Rightarrow \text{Retrieve}
```

Therefore, there is a similarity in the **decision mechanism**:

```text
Self-RAG:

reflection probability
        ↓
continuous score
        ↓
threshold
        ↓
retrieve


ART:

decision d
        ↓
threshold
        ↓
retrieve
```

However, the **learning signals are fundamentally different**:

```text
Self-RAG:
GPT-4 labels
    ↓
supervised reflection prediction


ART:
LM loss + retrieval cost
    ↓
learn d
```

Therefore, the threshold itself is not the main novelty of ART.

The novelty lies in **how the decision score is learned**.

---

### 6. Joint retriever training is a reasonable future extension

Self-RAG uses an off-the-shelf retriever.

This opens a potential future direction:

```text
Decision module
       ↕
Retriever
       ↓
Evidence
       ↓
Generation
       ↓
LM loss
```

A future system could investigate:

```text
Question
   ↓
Decision
   ↓
Retriever
   ↓
Evidence
   ↓
Generation
   ↓
LM loss
   ↓
Update retrieval-related components
```

However, this **should not be required for ART Phase 1**.

Introducing joint retriever training immediately would shift the research question from:

> **When should we retrieve?**

to:

> **When and what should we retrieve, and how should the retriever adapt?**

That is a substantially larger research problem.

---

## Summary

Self-RAG demonstrates that retrieval can be integrated directly into the generation process:

```text
Generate
   ↓
Should retrieve?
   ↓
Yes / No / Continue
   ↓
Retrieve
   ↓
Evaluate evidence
   ↓
Generate
   ↓
Evaluate generation
```

The three most important insights for ART are:

### 1. Retrieval timing is a decision made during generation

```text
Generate
   ↓
Should retrieve?
   ↓
Yes / No / Continue
   ↓
Generate further
```

Retrieval does not necessarily need to happen only once at the beginning of the query.

---

### 2. Retrieval queries need sufficient context

Self-RAG shows that retrieval can depend on:

```text
Question
+
Current generation context
```

rather than only an isolated reasoning sentence.

This complements IRCoT:

```text
IRCoT:
Reasoning → Retrieval query

Self-RAG:
Question + generation context → Retrieval query
```

---

### 3. Retrieval does not guarantee useful evidence

```text
Retrieve
   ↓
Relevant?
   ↓
Supported?
   ↓
Useful?
```

Therefore:

```math
\boxed{\text{Retrieve} \neq \text{Automatically useful}}
```

This provides an important motivation for ART.

The core difference is:

```text
                     SELF-RAG
                         │
              GPT-4 reflection supervision
                         ↓
                  Retrieve token
                         ↓
                 Retrieval timing
                         ↓
              Evidence / output critique


                       ART
                         │
               LM loss + retrieval cost
                         ↓
                  latent decision d
                         ↓
                 Retrieval timing
```

Therefore, ART **should not be positioned as "Self-RAG but simpler."**

A more accurate positioning is:

> **Self-RAG demonstrates that adaptive retrieval timing is a useful capability. ART investigates whether retrieval timing itself can be learned directly from the downstream generation objective and retrieval cost, without explicit retrieval-decision supervision or reinforcement learning.**
