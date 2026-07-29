# Agentic Synthetic Dataset Generation for Prompt Injection Detection

> **Project Type:** Personal Portfolio Project
>
> **Goal:** Build an agentic workflow capable of generating high-quality synthetic e-commerce customer support prompts containing embedded prompt injections, while ensuring diversity, balanced class distributions, and high dataset quality.

---

# 1. Project Goal

The objective of this project is to automatically generate a synthetic dataset that can be used to train and evaluate Prompt Injection Detection systems.

Instead of generating prompts completely from scratch, the system transforms existing customer support prompts into realistic malicious variants while preserving their original customer intent.

The generated dataset should satisfy several requirements:

- Preserve the original customer support context
- Introduce realistic prompt injection attacks
- Balance attack categories
- Balance original customer intents
- Minimize duplicated generations
- Produce structured outputs suitable for ML training

This project intentionally separates deterministic operations from LLM reasoning, following modern agent engineering practices.

---

# 2. High-Level Architecture

The system follows an iterative generation pipeline.

Each iteration generates **exactly one** synthetic record.

```
                 START
                    │
                    ▼
         Pick Source Prompt
                    │
                    ▼
       Pick Target Injection Class
                    │
                    ▼
      Retrieve Similar Examples
                    │
                    ▼
          Prompt Generator
                    │
                    ▼
        Prompt Validator ◄──────────────────┐
                    │                       │
         ┌──────────┴──────────┐            │
         │                     │            │
     Accepted               Rejected         │
         │                     │            │
         ▼                     ▼            │
   Save Record          Repair / Regenerate  │
         │              (should_retry,       │
         │               retries += 1,       │
         │               regenerated_prompt) │
         │                     │             │
         │                     └─────────────┘
         ▼
   Dataset Complete?
         │
    ┌────┴─────┐
    │          │
   Yes         No
    │          │
    ▼          │
   END ◄───────┘
```

The Prompt Validator performs the similarity and quality checks in a single pass and produces one overall assessment. When it rejects the current candidate, `should_retry` is set to `True` and the graph routes to the Repair Agent, which stores the fixed candidate in `regenerated_prompt` and increments `retries`. The graph then loops back to the Prompt Validator so the regenerated prompt is validated again.

---

# 3. Design Philosophy

The project intentionally separates:

- deterministic logic
- retrieval
- reasoning
- validation

Only tasks requiring reasoning are delegated to LLMs.

Everything else is implemented as deterministic Python nodes.

Advantages:

- reproducibility
- lower cost
- easier debugging
- easier testing
- modular architecture

---

# 4. Global State

The graph maintains a shared state that is updated after each node execution.

## State Field Semantics

| Field | Type | Meaning |
|---|---|---|
| `target_size` | `int` | Total number of synthetic records the run must produce. |
| `generated_count` | `int` | Number of records actually generated so far; may end up lower than `target_size` if some generations fail. |
| `remaining_input_indices` | `list[int]` | Indices of source prompts not yet used, so each source prompt is consumed at most once. |
| `input_distribution` | `dict[str, DistributionBucket]` | Per source-intent target/actual counters, used to keep original intents balanced. |
| `target_distribution` | `dict[str, DistributionBucket]` | Per attack-class target/actual counters, used to keep injected attack classes balanced. |
| `DistributionBucket.target` | `int` | Desired number of samples for that class, derived from the configured distribution. |
| `DistributionBucket.actual` | `int` | Number of samples picked so far for that class. |
| `source` | `InputDatasetModel \| None` | The source prompt (and its original intent) selected for the current iteration. |
| `target` | `InputAttackModel \| None` | The attack class (and its retrieved examples) selected for the current iteration. |
| `generated_prompt` | `OutputModel \| None` | The malicious prompt produced by the generator for the current iteration. |
| `regenerated_prompt` | `OutputModel \| None` | The prompt produced by the Repair Agent when repairing a rejected generation, if any. |
| `should_retry` | `bool` | Set by the Prompt Validator when the current candidate fails similarity or quality validation and must go through the Repair Agent. |
| `validation_output` | `QualityAssessmentModel \| None` | Overall assessment (`accepted`, `reason`, `quality_score`) produced by the Prompt Validator for the current candidate. |
| `retries` | `int` | Number of times the Repair Agent has regenerated/repaired the current record. |

Each agent updates a subset of fields within the main `DatasetState`.
First of all the starting state looks like this: 

```python
starting_state = DatasetState(
    target_size=5000,
    generated_count=0,
    remaining_input_indices=[0, 1, 2, 3, 4, 5, ..., 26999],

    # one bucket per source intent, target counts derived from "balanced" distribution
    input_distribution={
        "cancel_order": DistributionBucket(target=192, actual=0),
        "change_order": DistributionBucket(target=192, actual=0),
        "track_order": DistributionBucket(target=192, actual=0),
        # ... one entry per class in config/dataset.yml -> source_dataset.class_labels
    },

    # one bucket per attack class, target counts derived from configured percentages
    target_distribution={
        "Malware/Hacking": DistributionBucket(target=2000, actual=0),
        "Economic harm": DistributionBucket(target=1000, actual=0),
        "Fraud/Deception": DistributionBucket(target=500, actual=0),
        "Sexual/Adult content": DistributionBucket(target=500, actual=0),
        "Privacy": DistributionBucket(target=1000, actual=0),
    },

    source=None,
    target=None,
    generated_prompt=None,
    regenerated_prompt=None,
    should_retry=False,
    validation_output=None,
    retries=0,
)
```

---

# 5. Agent Responsibilities

## 5.1 Input Picker Agent

### Type

Deterministic Python node

### Responsibilities

- randomly sample an unused source prompt
- update remaining indices
- maintain balanced sampling over original intents

### Updated states
- `source`: the samples prompt and current intent is added to the state
- `remaining_input_indices`: the sampled index is removed from the list
- `input_distribution`: the distribution is either created as a `DistributionBucket` with the property `actual=0` or updated by incrementing the `actual` 

---

## 5.2 Attack Class Picker

### Type

Deterministic Python node

### Responsibilities

Choose the next attack class and a few target examples of that class to inject in the prompt, while maintaining a balanced distribution.

Example classes

- Malware / Hacking
- Fraud / Deception
- Privacy
- Economic Harm
- Adult Content

### Input
```
DatasetState
```

### Updated states
- `target`: the selected attack class and its sampled target examples are added to the state
- `target_distribution`: the distribution is either created as a `DistributionBucket` with the property `actual=0` or updated by incrementing the `actual`

---

## 5.3 Prompt Generator

### Type

Large Language Model

### Responsibilities

Transform the original prompt into a new malicious prompt.

- build the actual prompt using the information sampled at the preceding steps (`source` and `target`)
- generate the new prompt

Requirements:

- preserve customer intent
- preserve e-commerce context
- inject the malicious behavior
- avoid copying examples
- produce natural language


### Updated states
- `generated_prompt`: the newly generated malicious prompt is added to the state

---

## 5.4 Prompt Validator

### Type

Embedding Tool + Small LLM

### Responsibilities

Validate the current candidate (`regenerated_prompt` if the Repair Agent has already produced one, otherwise `generated_prompt`) in a single pass, producing one overall assessment:

- Similarity check: compare the candidate against previously generated prompts, the original dataset, and retrieved examples using cosine similarity, to catch near-duplicates
- Quality check: verify that the attack is actually present, customer support context is preserved, the prompt sounds natural, it isn't an obvious copy of the examples, and it follows instructions

Metrics

```
Cosine Similarity
```

Recommended models

- Sentence Transformers
- BGE
- E5

Output

```python
QualityAssessmentModel(
    accepted=False,
    reason="...",
    quality_score=0.91,
)
```

### Updated states
- `validation_output`: the single `QualityAssessmentModel` combining the similarity and quality outcome is stored in the state
- `should_retry`: set to `True` when `validation_output.accepted` is `False`, `False` otherwise

---

## 5.5 Repair Agent (Optional)

### Type

Large Language Model

### Responsibilities

Runs only when `should_retry` is `True`. Instead of regenerating from scratch, repair the generated prompt.

Example prompt

> Modify only the minimum amount necessary to satisfy the validation feedback.

This generally reduces token usage.

### Updated states
- `regenerated_prompt`: set to the repaired prompt
- `retries`: incremented by one
- `should_retry`: reset to `False` before looping back to the Prompt Validator, so the regenerated prompt goes through the same validation steps again

---

## 5.6 Save Node

### Type

Deterministic Python node

### Responsibilities

- append accepted record (`regenerated_prompt` if set, otherwise `generated_prompt`)
- update statistics
- update embedding index
- persist dataset

### Updated states
- `generated_count`: incremented after the accepted record is persisted
- `source`, `target`, `generated_prompt`, `regenerated_prompt`, `should_retry`, `validation_output`, `retries`: reset so the next iteration starts from a clean slate

---

# 6. Tool Layer

The project benefits from integrating tools that perform deterministic computations.

---


## Tool 2 — Similarity Search

Purpose

Detect duplicate generations.

Pipeline

```
Embedding

↓

Vector Search

↓

Maximum similarity
```

Output

```python
{
    "max_similarity": 0.82
}
```

---

## Tool 3 — Embedding Generator

Purpose

Generate vector representations for

- source prompts
- generated prompts
- malicious examples

Possible models

- all-MiniLM
- BGE
- E5

---

## Tool 4 — Dataset Statistics

Purpose

Return current statistics.

Example

```python
{
    "Shipping": 124,
    "Refund": 119,
    "Returns": 118
}
```

Used for balanced sampling.

---

## Tool 5 — Novelty Score

Compute

```
Novelty =
distance(original prompt)
+
distance(generated dataset)
+
distance(examples)
```

Useful for quality evaluation.

---

# 7. Dataset Generation Loop

```
while generated < target_size

    pick input

    pick attack class

    retrieve examples

    generate

    validate (similarity + quality)

    save
```

Each iteration generates exactly one record.

---

# 8. Balancing Strategy

The system should avoid over-representing classes.

Maintain two distributions.

## Original Intent Distribution

Example

```
Shipping

Returns

Payment

Refund

Account
```

---

## Attack Distribution

Example

```
Malware

Privacy

Fraud

Economic Harm

Adult Content
```

Sampling probability

```
P(class)

∝

1 / (count + 1)
```

This naturally converges toward balanced datasets.

---

# 9. Duplicate Prevention

The project uses multiple layers.

## Layer 1

Different source prompts.

---

## Layer 2

Balanced attack classes.

---

## Layer 3

Embedding similarity search.

---

## Layer 4

LLM quality verification.

---

## Layer 5

Hash exact duplicates.

---

# 10. Data Model

```python
class SyntheticRecord:

    text: str

    original_intent: str

    attack_class: str

    attack_present: bool

    attack_strength: str

    similarity_to_source: float

    similarity_to_dataset: float

    quality_score: float
```

---

# 11. Possible Extensions

## Multi-turn Conversations

Instead of generating a single prompt, generate realistic conversations.

---

## Attack Severity

Introduce levels

- Weak
- Medium
- Strong

---

## Attack Position

Inject attacks at

- beginning
- middle
- end
- interleaved

---

## Retrieval-Augmented Generation

Replace random examples with semantic retrieval.

---

## Automatic Evaluation

Evaluate generated prompts using another LLM.

---

## Human Review Mode

Flag uncertain generations for manual inspection.

---

# 12. Suggested Technology Stack

## Orchestration

- LangGraph

---

## LLM Framework

- LangChain

---

## Models

Generator

- GPT-4.1 / GPT-5 / Claude / Llama 3.1

Checker

- Smaller GPT model
- Llama 3 8B
- Mistral

---

## Embeddings

- Sentence Transformers
- BGE
- E5

---

## Vector Database

- FAISS (recommended)
- Chroma
- LanceDB
- Qdrant

---

## Dataset

- HuggingFace Datasets

---

## Validation

- Pydantic

---

## Storage

- JSONL
- Parquet

---

## Experiment Tracking (Optional)

- MLflow
- Weights & Biases

---

# 13. Future Improvements

- Parallel workers with shared embedding index
- Human-in-the-loop validation
- Active learning
- Automatic prompt difficulty estimation
- Synthetic benchmark generation
- Reinforcement through validator feedback
- Multiple generator models
- Ensemble quality checking

---

# 14. Why This Is an Agentic Workflow

Although the workflow is deterministic, it is agentic because each component has a specialized responsibility and collaborates through a shared state to achieve a long-term objective (building a high-quality dataset). Rather than relying on a single monolithic LLM call, the system decomposes the task into retrieval, reasoning, validation, and persistence stages. This separation improves modularity, observability, and extensibility.

The "agents" in this design are better viewed as specialized workers coordinated by LangGraph, with only the reasoning-intensive tasks delegated to language models. This hybrid architecture combines the reliability of deterministic software with the flexibility of LLMs and closely resembles production-grade LLM pipelines.