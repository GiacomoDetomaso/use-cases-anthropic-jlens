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
        Similarity Validation
                    │
         ┌──────────┴──────────┐
         │                     │
     Similar               Too Similar
         │                     │
         ▼                     ▼
   Quality Checker      Repair / Regenerate
         │
         ▼
      Save Record
         │
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

Example:

```python
class DatasetState:

    target_size: int

    generated_count: int

    remaining_input_indices: list[int]

    input_distribution: dict[str, int]

    attack_distribution: dict[str, int]

    current_prompt: InputRecord | None

    target_attack: str | None

    retrieved_examples: list[str]

    generated_prompt: QueryRecordModel | None

    retries: int

    generated_hashes: set[str]
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

### Input

```
DatasetState
```

### Output

```
current_prompt
```

---

## 5.2 Attack Class Picker

### Type

Deterministic Python node

### Responsibilities

Choose the next attack class while maintaining a balanced distribution.

Example classes

- Malware / Hacking
- Fraud / Deception
- Privacy
- Economic Harm
- Adult Content

Possible strategy:

Inverse frequency sampling

```
weight = 1 / (count + 1)
```

---

## 5.3 Example Retrieval Agent

### Type

Retrieval node

### Responsibilities

Retrieve examples belonging to the selected attack class.

Possible retrieval strategies:

- random
- semantic search
- nearest neighbors
- hardest examples

Output:

```
Top-K examples
```

---

## 5.4 Prompt Generator

### Type

Large Language Model

### Responsibilities

Transform the original prompt into a new malicious prompt.

Requirements:

- preserve customer intent
- preserve e-commerce context
- inject the malicious behavior
- avoid copying examples
- produce natural language

Input:

```
Original prompt

Original intent

Target attack class

Retrieved examples
```

Output

```
Generated prompt
```

---

## 5.5 Similarity Validator

### Type

Embedding Tool

Responsibilities

Compare the generated prompt against

- previously generated prompts
- original dataset
- retrieved examples

Metrics

```
Cosine Similarity
```

Decision

```
similarity < threshold

↓

accept

otherwise

↓

regenerate
```

Recommended models

- Sentence Transformers
- BGE
- E5

---

## 5.6 Quality Checker

### Type

Small LLM

Responsibilities

Verify:

- attack is actually present
- customer support context preserved
- prompt sounds natural
- no obvious copy from examples
- follows instructions

Output

```python
{
    "accepted": bool,
    "reason": "...",
    "quality_score": 0.91
}
```

---

## 5.7 Repair Agent (Optional)

Instead of regenerating from scratch, repair the generated prompt.

Example prompt

> Modify only the minimum amount necessary to satisfy the validation feedback.

This generally reduces token usage.

---

## 5.8 Save Node

Deterministic node.

Responsibilities

- append accepted record
- update statistics
- update embedding index
- persist dataset

---

# 6. Tool Layer

The project benefits from integrating tools that perform deterministic computations.

---

## Tool 1 — Semantic Retrieval

Purpose

Retrieve examples most similar to the source prompt.

Implementation

- FAISS
- Chroma
- Qdrant

Input

```
query
category
k
```

Output

```
Top K examples
```

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

    similarity check

    quality check

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