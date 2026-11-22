# H1 — Attention Modulation

## Hypothesis

Attention-score modulation does not destroy the language-generation
capability of the base language model.

Furthermore, varying the retrieval decision should provide a controllable
and ordered change in the amount of information the model receives from the
retrieved document. This change should lead to a corresponding change in
answer accuracy.

---

## H1.1 — Language Capability Preservation

The first part of the hypothesis is that modifying attention scores should
not fundamentally impair the language-generation capability of the model.

The model should remain capable of generating coherent and meaningful
language when the attention scores are modulated at different retrieval
decision levels.

In other words:

```text
Attention modulation
        ↓
different document influence
        ↓
language generation remains functional

The modulation mechanism should control information access rather than
destroy the underlying language modeling capability.
```

## H1.2 — Ordered Information Control

The second part of the hypothesis is that the retrieval decision can
continuously control the amount of information propagated from the retrieved
document.

As the retrieval decision increases, the contribution of the retrieved
document to the model's attention should increase accordingly.

Conceptually:

```text
Low retrieval decision
        ↓
Less document information
        ↓
Higher retrieval decision
        ↓
More document information
```

The expected relationship is ordered rather than necessarily perfectly
linear.

## H1.3 — Downstream Answer Behavior

If the retrieval decision successfully controls the amount of information
available from the retrieved document, this should be reflected in the
model's downstream QA behavior.

As more relevant document information becomes accessible, answer accuracy
should generally improve.

Conceptually:

```text
Retrieval decision
        ↓
Document information accessible
        ↓
Model's effective evidence
        ↓
Answer accuracy
```

Therefore, the expected relationship is:

```text
Higher retrieval decision
        ↓
Greater document information access
        ↓
Better answer accuracy
```

## Experimental Requirement

The hypothesis should be evaluated by keeping the following factors fixed:

- model parameters
- question
- retrieved document
- document location
- decoding strategy
- generation budget

while varying only the retrieval decision / attention modulation strength.

The experiment should evaluate multiple retrieval-decision levels to
determine whether the resulting model behavior follows the expected ordered
relationship.

## Summary

H1 tests whether attention-score modulation can function as a controllable
document-information mechanism while preserving the language-generation
capability of the underlying language model.

The hypothesis consists of two primary claims:

- Language preservation: attention modulation does not destroy language
  generation.
- Ordered information control: increasing retrieval decision produces an
  ordered increase in accessible document information, which should result
  in a corresponding improvement in answer accuracy.
