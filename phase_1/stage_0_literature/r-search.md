# R-Search — Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning

**Paper:** Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha, Ma Bowen, Zhichun Wang, Shijie Jia, Limin Liu, Xin Wang

**Title:** _R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning_

**Venue:** Findings of ACL 2026

**Paper:** https://aclanthology.org/2026.findings-acl.1896/

**Code:** https://github.com/QingFei1/R-Search

**Hugging Face:** https://huggingface.co/qingfei1/R-Search_datasets

---

## 1. Problem

Large Language Models have strong parametric knowledge and increasingly powerful reasoning abilities, but complex knowledge-intensive tasks can still require external information.

A standard RAG pipeline usually follows:

```text
Question
   ↓
Retrieve K documents
   ↓
Retrieved documents
   ↓
Generate answer
```

This has an important limitation:

> **The reasoning process and retrieval process are largely separated.**

For multi-hop questions, the model may need to:

```text
Reason
   ↓
Search
   ↓
Observe evidence
   ↓
Reason again
   ↓
Search again
   ↓
Combine evidence
   ↓
Answer
```

Therefore, the problem is not simply:

> **What documents should be retrieved?**

but also:

> **When should the model search, what should it search for, and how should retrieved information be integrated into the reasoning process?**

R-Search focuses on this broader **reasoning–search interaction** problem.

Instead of:

```text
Question → Retrieve → Generate
```

R-Search allows the LLM to autonomously construct an interleaved trajectory:

```text
Question
   ↓
Reason
   ↓
Search?
   ↓
Observation
   ↓
Reason
   ↓
Search?
   ↓
Observation
   ↓
Reason
   ↓
Synthesize evidence
   ↓
Answer
```

The key objective is to learn a useful **reasoning–search trajectory** rather than simply performing a fixed retrieval step.

The paper formulates the interaction as a sequential decision process and uses reinforcement learning to optimize the resulting trajectory.

---

# 2. Core Idea

The core idea of R-Search is:

> **Use multi-reward reinforcement learning to teach an LLM how to interleave reasoning and search, while globally synthesizing useful evidence from the retrieved information.**

R-Search has three important components:

```text
1. Dynamic search triggering
            ↓
2. Multi-step reasoning + search
            ↓
3. Global evidence synthesis
            ↓
4. Multi-reward RL optimization
```

---

## Dynamic search triggering

During generation, the model can decide to search by generating:

```text
<search>query</search>
```

The search query is sent to an external retrieval system.

The retrieved passages are returned as:

```text
<observation>
retrieved passages
</observation>
```

The model then continues reasoning using the retrieved information.

Therefore:

```text
Reason
   ↓
<search>query</search>
   ↓
Search engine
   ↓
<observation>documents</observation>
   ↓
Reason
   ↓
<search>query</search>
   ↓
<observation>documents</observation>
   ↓
...
```

Search is therefore **not restricted to one fixed position**.

It can happen multiple times during reasoning.

---

## Token-level search control

Unlike a method that makes a retrieval decision only once per question, R-Search allows the policy model to trigger search during generation by producing the corresponding search tags.

Conceptually:

```text
Generated reasoning tokens
        ↓
   Search trigger?
        │
   ┌────┴────┐
   No        Yes
   │          │
Continue   <search>...</search>
               ↓
          Observation
               ↓
          Continue
```

Therefore, search timing becomes part of the generated trajectory itself.

---

## Evidence distillation

R-Search does not simply append all retrieved passages to the final answer.

After the reasoning–search process, the model performs an additional global synthesis step.

It generates:

```text
<original_evidence>
...
</original_evidence>
```

The purpose is to consolidate useful information from potentially many observations.

The complete trajectory is approximately:

```text
Question
   ↓
Reason
   ↓
<search>query</search>
   ↓
<observation>...</observation>
   ↓
Reason
   ↓
<search>query</search>
   ↓
<observation>...</observation>
   ↓
Reason / rethink
   ↓
<original_evidence>...</original_evidence>
   ↓
<answer>...</answer>
```

This evidence-generation stage is important because multiple searches can produce a large amount of heterogeneous information.

Instead of forcing the final answer to directly consume every observation, R-Search asks the policy model to **globally identify and synthesize the useful evidence**.

The paper's ablation shows that removing the evidence mechanism causes a noticeable performance drop, especially on complex multi-hop datasets.

---

# 3. Model, Retriever

## Language model

R-Search trains open-source instruction-tuned models using RL.

The main policy models are:

```text
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
```

This is important:

> **R-Search uses the Instruct variants, not the base Qwen2.5 models.**

The released checkpoints include:

| Model            | Backbone            | RL algorithm |
| ---------------- | ------------------- | ------------ |
| R-Search-3b-grpo | Qwen2.5-3B-Instruct | GRPO         |
| R-Search-3b-ppo  | Qwen2.5-3B-Instruct | PPO          |
| R-Search-7b-grpo | Qwen2.5-7B-Instruct | GRPO         |
| R-Search-7b-ppo  | Qwen2.5-7B-Instruct | PPO          |

The official Hugging Face model cards explicitly identify these backbones.

Therefore, if ART uses Qwen2.5-3B-Instruct, the backbone family and scale are directly comparable to the 3B R-Search models.

---

## Evidence evaluator

R-Search uses a separate cross-family model for the evidence reward:

```text
Llama-3.2-3B-Instruct
```

The reason is important.

If the same policy model evaluated its own evidence, the evaluation could be biased toward its own generation behavior.

Instead:

```text
R-Search policy
       ↓
original_evidence
       ↓
Llama-3.2-3B-Instruct
       ↓
independent answer
       ↓
F1 against gold answer
       ↓
Evidence reward
```

The paper describes this as a cross-model evidence evaluation mechanism.

---

# 4. Retriever

R-Search uses an external retrieval system rather than training the retriever jointly with the policy model.

The default dense retriever is:

**E5-base-v2**

The official repository provides both:

```text
Dense retrieval:
E5-base-v2
```

and:

```text
Sparse retrieval:
BM25
```

The repository supports retrieval corpora including:

```text
2WikiMultihopQA
HotpotQA
MuSiQue
Wiki-18
```

For example:

```text
Query
  ↓
E5-base-v2
  ↓
Similarity search
  ↓
Top-K passages
  ↓
<observation>
```

The GitHub repository explicitly describes E5-base-v2 as the default dense retriever and also provides a BM25 implementation.

---

## Training vs evaluation retrieval

R-Search uses:

```text
Training retrieval top-k = 3
```

and:

```text
Evaluation retrieval top-k = 5
```

The authors state that evaluation uses top-k = 5 across methods for comparison, while training uses top-k = 3 following Search-R1.

---

# 5. How to Make Query

R-Search is different from IRCoT in how retrieval queries are produced.

## IRCoT

IRCoT explicitly uses the generated reasoning step to guide retrieval:

```text
Question
   ↓
Reasoning sentence 1
   ↓
Retrieve using sentence 1
   ↓
Reasoning sentence 2
   ↓
Retrieve using sentence 2
```

The reasoning step itself acts as the search query.

---

## R-Search

R-Search allows the model to explicitly generate a search query:

```text
Reasoning
   ↓
<search>
query generated by the policy
</search>
```

The query is therefore an explicit action produced by the model.

Conceptually:

```text
Current reasoning state
        ↓
Policy model
        ↓
Generate search query
        ↓
External retriever
        ↓
Observation
```

This gives the model control over both:

```text
WHEN to search
```

and:

```text
WHAT to search for
```

This is broader than ART Phase 1, whose primary question is specifically **when to retrieve**.

---

# 6. Training

R-Search uses reinforcement learning rather than supervised retrieval-decision labels.

The overall training pipeline is approximately:

```text
2WikiMQA training questions
        ↓
Policy model
        ↓
Reasoning
        ↓
Search actions
        ↓
Retriever
        ↓
Observations
        ↓
Reasoning
        ↓
Original evidence
        ↓
Final answer
        ↓
Multiple rewards
        ↓
GRPO / PPO
        ↓
Update policy
```

The released models are trained only on the **2WikiMQA training set**.

---

## Step 1 — Generate a reasoning–search trajectory

For each question, the policy model generates a trajectory.

For example:

```text
Question
   ↓
Reasoning
   ↓
<search>query 1</search>
   ↓
<observation>documents 1</observation>
   ↓
Reasoning
   ↓
<search>query 2</search>
   ↓
<observation>documents 2</observation>
   ↓
Reasoning
   ↓
<original_evidence>...</original_evidence>
   ↓
<answer>...</answer>
```

The model therefore learns both:

```text
Reasoning tokens
```

and:

```text
Search actions
```

as part of one policy.

---

# 7. Multi-Reward Reinforcement Learning

This is the central component of R-Search.

Instead of relying only on final answer correctness, R-Search uses multiple reward signals.

The total reward is conceptually:

```math
r = r_answer + r_evidence + r_format
```

The three major components are:

```text
Answer Reward
Evidence Reward
Format Reward
```

---

## 7.1 Answer Reward

The answer reward evaluates the final answer against the gold answer.

Conceptually:

```math
r_{answer}
=
F1(answer_{pred}, answer_{gold})
```

Therefore:

```text
Reasoning
   ↓
Search
   ↓
Evidence
   ↓
Answer
   ↓
F1
   ↓
Answer reward
```

This provides a direct task-level signal.

---

# 7.2 Evidence Reward

The evidence reward is one of the distinctive components of R-Search.

The model first generates:

```text
<original_evidence>...</original_evidence>
```

Then a separate cross-family model evaluates whether this evidence contains sufficient information to answer the question.

The process is:

```text
Question
   +
Generated evidence
   ↓
Llama-3.2-3B-Instruct
   ↓
Independent answer
   ↓
F1 against gold answer
   ↓
Evidence reward
```

Formally:

```math
r_{evidence}
=
F1(answer_{cross-model}, answer_{gold})
```

The idea is that good evidence should enable an independent model to produce the correct answer.

This provides an intermediate signal about the **quality of the knowledge acquired during search**, rather than evaluating only the final answer.

---

# 7.3 Format Reward

R-Search also rewards correct structural formatting.

The policy is expected to correctly use tags such as:

```text
<search>...</search>

<observation>...</observation>

<original_evidence>...</original_evidence>

<answer>...</answer>
```

The format reward encourages the model to follow the expected reasoning–search protocol.

Conceptually:

```text
Correct search format
        +
Correct evidence format
        +
Correct answer format
        ↓
Format reward
```

Therefore, the final reward combines:

```text
Task success
      +
Evidence quality
      +
Trajectory / format validity
```

---

# 8. RL Algorithms

R-Search evaluates two reinforcement learning algorithms:

```text
PPO
GRPO
```

---

## PPO

PPO uses a value model / critic to estimate the advantage of generated actions.

Conceptually:

```text
Question
   ↓
Sample trajectory
   ↓
Reward
   ↓
Value / advantage estimation
   ↓
PPO update
   ↓
Policy
```

---

## GRPO

R-Search also uses Group Relative Policy Optimization.

For each question, multiple responses are sampled:

```text
Question
   │
   ├── Response 1 → Reward 1
   ├── Response 2 → Reward 2
   ├── Response 3 → Reward 3
   ├── Response 4 → Reward 4
   └── Response 5 → Reward 5
```

The rewards are normalized relative to the group.

Conceptually:

```text
5 trajectories
      ↓
5 rewards
      ↓
Group mean / std
      ↓
Relative advantages
      ↓
GRPO update
```

The R-Search implementation samples **5 responses per input prompt** during rollout.

The paper reports that GRPO generally performs better than PPO, particularly for the larger 7B model and more complex multi-hop tasks.

---

# 9. Training Configuration

The main RL training setup is:

```text
Training dataset:
2WikiMQA

Policy:
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct

GPU:
8 × A100 80GB

Maximum training steps:
195

Batch size:
256

Learning rate:
1e-6

Warm-up ratio:
0.95

Training retrieval top-k:
3

Evaluation retrieval top-k:
5

Rollout samples:
5 responses / input
```

These settings are reported in the implementation details of the paper.

---

# 10. Important Masking Mechanism

The R-Search trajectory contains both:

```text
Tokens generated by the policy
```

and:

```text
Tokens returned by the retrieval environment
```

For example:

```text
Policy:
"Who was X?"
<search>Who was X?</search>

Environment:
<observation>
Passage A...
Passage B...
</observation>

Policy:
"Therefore..."
```

The retrieved observation is **not generated by the policy**.

Therefore, retrieved document tokens are masked from the policy loss.

Conceptually:

```text
<search>query</search>
        ↓
   train policy

<observation>...</observation>
        ↓
   DO NOT train policy

<original_evidence>...</original_evidence>
        ↓
   train policy

<answer>...</answer>
        ↓
   train policy
```

The distinction is important:

> **External observation is treated as environment-provided information, whereas generated evidence is part of the policy's behavior.**

The paper's training description uses masking for retrieved observations while retaining generated evidence in the optimization process.

---

# 11. Evaluation

R-Search evaluates on seven QA datasets:

### Multi-hop QA

```text
HotpotQA
2WikiMQA
MuSiQue
Bamboogle
```

### Single-hop / general QA

```text
NQ
TriviaQA
PopQA
```

The evaluation compares R-Search with vanilla and advanced RAG methods and also with Search-R1.

---

## Main evaluation question

The key question is:

> **Can RL-trained reasoning–search interaction improve knowledge-intensive reasoning compared with conventional RAG pipelines?**

The experiments show improvements across the evaluated datasets, with particularly notable gains on complex multi-hop tasks.

---

# 12. R-Search vs Search-R1

R-Search explicitly compares against Search-R1.

Both methods use RL to integrate reasoning and search, but R-Search introduces additional evidence-oriented mechanisms.

Conceptually:

```text
Search-R1

Reason
  ↓
Search
  ↓
Observation
  ↓
Reason
  ↓
Answer
```

while:

```text
R-Search

Reason
  ↓
Search
  ↓
Observation
  ↓
Reason
  ↓
Search
  ↓
Observation
  ↓
Global evidence synthesis
  ↓
Answer
```

R-Search therefore adds an explicit **global evidence distillation stage** and corresponding evidence reward.

The paper reports that R-Search outperforms Search-R1 in most of its comparisons, including a reported 5.6-point advantage on MuSiQue.

---

# 13. Ablation

## 13.1 Removing evidence

The paper evaluates:

```text
R-Search
```

against:

```text
R-Search w/o Evidence
```

The latter removes the evidence component from the rollout and disables the associated evidence-related rewards.

For Qwen2.5-3B-Instruct, the reported F1 results include:

| Dataset   | w/o Evidence | R-Search |
| --------- | -----------: | -------: |
| HotpotQA  |         53.4 |     54.4 |
| 2WikiMQA  |         66.6 |     72.6 |
| MuSiQue   |         33.3 |     34.8 |
| Bamboogle |         42.3 |     49.8 |
| NQ        |         44.6 |     46.0 |
| TriviaQA  |         62.3 |     64.0 |
| PopQA     |         41.6 |     44.9 |

The difference is particularly visible on several multi-hop datasets.

This supports the paper's argument that globally synthesizing evidence is useful when reasoning requires combining information obtained from multiple searches.

---

## 13.2 PPO vs GRPO

The paper compares PPO and GRPO.

For Qwen2.5-3B-Instruct:

| Method | HotpotQA | 2WikiMQA | MuSiQue | Bamboogle |   NQ | TriviaQA | PopQA |
| ------ | -------: | -------: | ------: | --------: | ---: | -------: | ----: |
| PPO    |     52.4 |     60.3 |    33.5 |      50.0 | 43.7 |     63.1 |  44.2 |
| GRPO   |     54.4 |     72.6 |    34.8 |      49.8 | 46.0 |     64.0 |  44.9 |

For Qwen2.5-7B-Instruct:

| Method | HotpotQA | 2WikiMQA | MuSiQue | Bamboogle |   NQ | TriviaQA | PopQA |
| ------ | -------: | -------: | ------: | --------: | ---: | -------: | ----: |
| PPO    |     58.7 |     68.9 |    37.3 |      53.7 | 47.3 |     70.4 |  45.5 |
| GRPO   |     64.4 |     77.7 |    41.6 |      57.6 | 49.1 |     71.7 |  48.1 |

The paper notes that GRPO generally performs better, especially for the larger model and more difficult multi-hop tasks.

---

# 14. R-Search-as-a-Tool

An additional component of the work is **R-Search-as-a-Tool (RSTool)**.

The motivation is that the reasoning–search process can be computationally expensive.

Instead of requiring every downstream model to perform the entire reasoning-search trajectory, R-Search can produce distilled evidence that another model can consume.

Conceptually:

```text
Question
   ↓
R-Search
   ↓
Reasoning + Search
   ↓
Original Evidence
   ↓
Downstream LLM
   ↓
Final Answer
```

Therefore, R-Search can act as an external knowledge-search tool rather than necessarily being the final answer generator.

This makes the learned search-and-evidence process more modular.

---

# 15. Limitations

## 15.1 RL training complexity

R-Search requires:

```text
Policy model
   +
Retriever
   +
Rollout
   +
Reward computation
   +
Evidence verifier
   +
GRPO / PPO
```

This is considerably more complicated than standard supervised fine-tuning.

The training loop therefore involves repeated generation, retrieval, evidence evaluation, and policy optimization.

---

## 15.2 Multiple reward components

R-Search requires designing:

```text
Answer reward
Evidence reward
Format reward
```

Each reward captures a different aspect of desired behavior.

This introduces reward-design complexity.

In particular, evidence reward requires an additional model:

```text
Llama-3.2-3B-Instruct
```

to independently evaluate the generated evidence.

---

## 15.3 Search trajectory exploration

Because R-Search learns reasoning–search interaction through RL, the model must explore different trajectories.

For example:

```text
Trajectory A:
Reason → Search → Search → Answer

Trajectory B:
Reason → Search → Reason → Answer

Trajectory C:
Reason → Search → Search → Search → Answer
```

The RL process must determine which trajectories lead to better outcomes.

This introduces the usual challenges associated with trajectory-level policy optimization, including exploration, credit assignment, and reward sparsity.

The paper itself discusses the difficulty of sparse/delayed rewards when comparing PPO and GRPO.

---

## 15.4 Training data diversity

The released RL models are trained only on the **2WikiMQA training set**.

The evaluation, however, spans seven datasets.

Therefore, the training distribution is considerably narrower than the evaluation distribution.

The official repository also explicitly states that only the 2WikiMultihopQA training set is required for training.

---

## 15.5 R-Search solves a broader problem than retrieval timing

R-Search does not isolate:

```text
WHEN TO RETRIEVE
```

Instead, it jointly learns:

```text
WHEN TO SEARCH
      +
WHAT TO SEARCH
      +
HOW TO REASON AFTER SEARCH
      +
HOW TO SYNTHESIZE EVIDENCE
      +
HOW TO ANSWER
```

Therefore, its research scope is substantially broader than a method whose primary objective is specifically adaptive retrieval timing.

---

# 16. My observations

## 16.1 Retrieval timing becomes a policy decision

R-Search strongly supports the idea that search timing should not necessarily be fixed.

Instead:

```text
Reason
   ↓
Search?
   ↓
Reason
   ↓
Search?
   ↓
...
```

This means retrieval/search timing can be viewed as part of the model's sequential decision process.

This is directly relevant to ART.

---

## 16.2 But R-Search does not isolate the timing problem

A crucial distinction is:

```text
R-Search:

WHEN?
+
WHAT QUERY?
+
WHAT EVIDENCE?
+
HOW TO REASON?
+
HOW TO ANSWER?
```

whereas ART focuses on:

```text
WHEN?
```

Therefore, if ART improves retrieval timing, it becomes possible to ask a more isolated research question:

> **Can the timing decision itself be learned without requiring the model to learn the entire reasoning–search policy through RL?**

---

## 16.3 R-Search treats search as an action

R-Search effectively makes search part of the policy's action space:

```text
Reasoning token
      ↓
Search action
      ↓
Observation
      ↓
Reasoning
```

This is fundamentally different from ART's formulation.

ART attempts to make the retrieval decision a **differentiable latent variable**:

```text
Hidden state
    ↓
Decision d
    ↓
Attention modulation
    ↓
LM prediction
    ↓
LM loss
    ↓
Gradient
    ↓
Decision d
```

Therefore:

```text
R-Search:
learn decision through trajectory reward

ART:
learn decision through downstream LM objective
```

This is one of the most important conceptual differences between the two approaches.

---

# 17. Relation to ART

R-Search and ART overlap in the high-level goal of adaptive retrieval/search.

Both allow retrieval to happen dynamically during generation:

```text
Traditional RAG:

Question
   ↓
Retrieve
   ↓
Generate
```

versus:

```text
R-Search / ART:

Generate
   ↓
Should retrieve?
   ↓
Retrieve if necessary
   ↓
Continue generation
```

However, their learning paradigms are fundamentally different.

| Aspect               | R-Search                               | ART                                                    |
| -------------------- | -------------------------------------- | ------------------------------------------------------ |
| Main question        | Reasoning–search integration           | **When to retrieve**                                   |
| Search timing        | Learned                                | **Learned**                                            |
| Search query         | Generated by policy                    | Separate retrieval mechanism                           |
| Search action        | Explicit generated action              | **Latent decision**                                    |
| Backbone             | Qwen2.5-3B/7B-Instruct                 | **Frozen LLM**                                         |
| Decision supervision | None                                   | **None**                                               |
| Learning signal      | **RL rewards**                         | **LM loss + retrieval cost**                           |
| RL                   | **Yes**                                | **No**                                                 |
| Answer reward        | **Yes**                                | LM loss                                                |
| Evidence reward      | **Yes**                                | Not in Phase 1                                         |
| Format reward        | **Yes**                                | Not required                                           |
| Evidence synthesis   | **Yes**                                | Not Phase 1                                            |
| Policy update        | RL                                     | **Gradient through differentiable decision mechanism** |
| Retriever            | External                               | Experimental choice                                    |
| Scope                | Reasoning + search + evidence + answer | **Retrieval timing**                                   |

---

# 18. Key conceptual distinction

The difference can be summarized as:

```text
                    R-SEARCH

              Reasoning trajectory
                       ↓
                Search action
                       ↓
                  Observation
                       ↓
                Reasoning again
                       ↓
                 Search again
                       ↓
              Evidence synthesis
                       ↓
                     Answer
                       ↓
              Multi-type rewards
                       ↓
                  RL update
```

while:

```text
                       ART

                Reasoning / state
                       ↓
                latent decision d
                       ↓
              retrieval modulation
                       ↓
                  LM prediction
                       ↓
                   LM loss
                       +
                retrieval cost
                       ↓
                 gradient update
```

Therefore, the fundamental distinction is not simply:

> **R-Search uses RL, ART does not.**

The deeper distinction is:

> **R-Search learns a complete reasoning–search policy through trajectory-level rewards, whereas ART investigates whether the retrieval-timing decision can be learned locally through the downstream language-modeling objective.**

---

# 19. What R-Search suggests for ART

## 1. Retrieval timing is a legitimate learning problem

R-Search provides evidence that dynamic search timing can be useful in complex reasoning.

Therefore, ART's focus on:

```text
WHEN TO RETRIEVE
```

is not an artificial subproblem.

It is one component of a larger reasoning–search interaction problem.

---

## 2. Search and reasoning are naturally coupled

R-Search shows:

```text
Reasoning
   ↕
Search
```

rather than:

```text
Reasoning
   ↓
Search
   ↓
Reasoning
```

being a strictly one-way pipeline.

The usefulness of retrieval depends on the model's current reasoning state.

This motivates ART's use of the model's hidden state to determine retrieval timing.

---

## 3. Search decisions are not necessarily directly supervised

R-Search does not require manually annotated:

```text
Retrieve = Yes
Retrieve = No
```

labels.

Instead, the model discovers useful search behavior through RL rewards.

This is important because it establishes that retrieval decisions can be treated as **latent policy decisions** rather than necessarily requiring explicit labels.

However, ART asks whether an even more local learning signal can be sufficient:

```text
R-Search:

trajectory
   ↓
reward
   ↓
RL
```

versus:

```text
ART:

decision
   ↓
LM objective
   ↓
gradient
```

---

## 4. Evidence synthesis is a separate problem

R-Search demonstrates the value of globally synthesizing retrieved information.

However, this does not mean ART Phase 1 must solve evidence synthesis.

A clean decomposition is:

```text
Phase 1:

WHEN?
 ↓
Retrieve / No Retrieve


Future:

WHEN?
 ↓
WHAT?
 ↓
HOW TO USE?
 ↓
FINAL ANSWER
```

This keeps the first research question experimentally isolated.

---

## 5. R-Search is a useful RL baseline

For ART, R-Search is especially relevant because it represents a modern RL-based reasoning–search approach using a similarly sized backbone.

A fair comparison can use:

```text
ART:
Qwen2.5-3B-Instruct
```

against:

```text
R-Search:
Qwen2.5-3B-Instruct
```

while clearly reporting that R-Search's policy is RL-trained and ART trains only its retrieval-decision mechanism.

The released R-Search 3B checkpoints are explicitly based on Qwen2.5-3B-Instruct.

---

# 20. R-Search vs ART — learning paradigm

The most concise comparison is:

```text
                   SUPERVISED
                       │
                       │
              Explicit decision labels
                       │
                       ▼
                   Self-RAG
                       │
                       │
              Reflection supervision
                       │
                       ▼
                ───────────────
                       │
                       │
                       ▼
                     ART
                       │
             latent decision
                       │
             LM objective
                       │
                       ▼
                ───────────────
                       │
                       │
                       ▼
                  R-SEARCH
                       │
              trajectory exploration
                       │
                  RL reward
                       │
                       ▼
                  Policy learning
```

This diagram should **not** be interpreted as a formal ML taxonomy.

Instead, it highlights a difference in **how the retrieval decision receives its learning signal**.

Self-RAG:

```text
GPT-4
 ↓
Retrieve label
 ↓
Supervised training
```

R-Search:

```text
Trajectory
 ↓
Answer / evidence / format rewards
 ↓
RL
```

ART:

```text
Retrieval decision
 ↓
Effect on generation
 ↓
LM loss + retrieval cost
 ↓
Gradient
```

---

# 21. Main takeaway

R-Search demonstrates that:

> **LLMs can learn dynamic reasoning–search interaction through reinforcement learning and multi-stage rewards.**

Its contribution is broader than retrieval timing:

```text
WHEN TO SEARCH
+
WHAT TO SEARCH
+
HOW TO REASON
+
HOW TO SYNTHESIZE EVIDENCE
+
HOW TO ANSWER
```

ART deliberately isolates one part of this problem:

> **Can retrieval timing itself be learned directly from the downstream generation objective and retrieval cost, without explicit retrieval-decision supervision or reinforcement learning?**

Therefore, ART should **not** be positioned as:

> “R-Search without RL.”

A more accurate positioning is:

> **R-Search learns reasoning–search trajectories through trajectory-level reinforcement learning, whereas ART investigates whether retrieval timing can be learned locally from the language-modeling objective, while leaving the underlying LLM frozen.**

This distinction makes the comparison scientifically meaningful:

```text
R-Search
Trajectory-level optimization
        ↓
RL
        ↓
Reasoning + Search policy


ART
Local decision optimization
        ↓
LM objective + retrieval cost
        ↓
Retrieval timing
```

The central experimental question for ART is therefore not whether it can replace RL for every tool-use problem, but whether **retrieval timing contains enough local learning signal in the downstream LM objective that explicit decision supervision and trajectory-level RL can be avoided.**
