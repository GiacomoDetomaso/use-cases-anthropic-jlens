# Agentic Synthetic Dataset Generation

## Purpose

The project creates synthetic e-commerce customer-support prompts that retain a source customer's intent while embedding an injection from a selected attack class.

It is a LangGraph state machine. Deterministic nodes select, balance, discard, and persist records. One cached LangChain chat model performs generation, validation, and repair through separate prompts and Pydantic schemas.

The current implementation has no embedding model, vector store, similarity search, or separate validation model.

## Graph

```mermaid
flowchart TD
    start([START]) --> pickInput[Pick source prompt]
    pickInput --> pickTarget[Pick attack class and examples]
    pickTarget --> generator[Generate candidate]
    generator -->|candidate generated| validator[Validate candidate]
    generator -->|generation failed, work remains| pickInput
    generator -->|generation failed, target reached| end([END])
    validator -->|accepted| save[Save accepted record]
    validator -->|rejected, retries remain| repair[Repair candidate]
    validator -->|rejected, retries exhausted| discard[Discard candidate]
    repair --> validator
    save -->|target reached| end
    save -->|work remains| pickInput
    discard -->|target reached| end
    discard -->|work remains| pickInput
```

Each iteration selects one source prompt and one target attack class. An accepted candidate is persisted. A rejected candidate is repaired and revalidated until `workflow.max_repair_attempts` is reached. Exhausted candidates are discarded, their class counters are rolled back, and the achievable target size decreases by one.

## Inputs And Balancing

`dataset_agent.dataset_source` caches two configured CSV datasets as pandas DataFrames.

- `source_dataset` provides customer-support prompts and original intents.
- `target_transformation_examples_dataset` provides attack classes and reference examples.

`PickerInputDatasetNode` consumes an unused source-row index, stores an `InputDatasetModel`, and increments the source-intent counter. `PickerTargetDatasetNode` chooses an attack class, samples three examples, stores an `InputAttackModel`, and increments the attack-class counter.

Both nodes use `DistributionBalancer`. It initializes class targets using the configured `balanced`, `random`, or percentage distribution, then chooses among classes with the largest remaining target deficit. The current source distribution is balanced; attack-class targets use configured percentages.

## State

`DatasetState` is the graph state schema.

| Field | Meaning |
|---|---|
| `target_size` | Current achievable accepted-record target; failures can lower it. |
| `index_to_generate` | Number of accepted records saved so far. |
| `remaining_input_indices` | Source-row indices available for selection. |
| `input_distribution` / `target_distribution` | Per-class `DistributionBucket(target, actual)` counters. |
| `source` / `target` | Current source prompt and selected attack class with examples. |
| `generated_prompt` / `regenerated_prompt` | Initial and latest repaired `OutputModel` candidates. |
| `validation_output` | Latest `QualityAssessmentModel`. |
| `should_retry` / `retries` | Rejection state and repair attempt count for the current record. |
| `last_checkpoint_index` | Checkpoint marker used by the save node. |

`get_graph_initial_state()` starts with `index_to_generate=0`, the configured target size, and every source-row index available.

## Nodes

### Pickers

`dataset_agent/nodes/pickers.py` contains `PickerInputDatasetNode` and `PickerTargetDatasetNode`. Selection has seeded random tie-breaking and example sampling; distribution counters control the run-wide class mix.

### Generator

`dataset_agent/nodes/generator.py` formats `dataset_generator_prompt.yml` using the current source and target, then requests `OutputModel` through structured output. Failed inference rolls back source and target counters and lowers `target_size`. `generation_router` continues to validation, selects another record, or ends when the reduced target is reached.

### Validator

`dataset_agent/nodes/validator.py` evaluates `regenerated_prompt` when present, otherwise `generated_prompt`. It formats `prompt_validator.yml` and requests `QualityAssessmentModel` through structured output.

The model classifies these categories independently as `very_low`, `low`, `medium`, `high`, or `very_high`:

- `intent_context_preservation`
- `attack_class_alignment`
- `originality`
- `naturalness_and_coherence`

The Pydantic model computes `overall_level` as the mode of the four levels, with conservative lower-level tie-breaking. It computes `accepted` as `True` only for `high` and `very_high`. `feedback` is one actionable sentence capped at 160 characters.

On validation inference failure, the node creates a `very_low` assessment containing the failure feedback. `validation_router` sends accepted records to save, rejects to repair while retries remain, and exhausted rejects to discard.

### Repair

`dataset_agent/nodes/repair.py` formats `prompt_repair.yml` with the candidate and validator feedback. It requests a new `OutputModel`, stores it in `regenerated_prompt`, increments `retries`, and clears `should_retry` before returning to validation. A failed repair still consumes a retry; subsequent validation evaluates the initial candidate.

### Discard

`dataset_agent/nodes/discard.py` handles candidates rejected after all repair attempts. It rolls back the active distribution buckets, lowers `target_size` without dropping below `index_to_generate`, and clears transient generation, validation, and retry state.

### Save

`dataset_agent/nodes/save.py` creates `SyntheticRecord` from the source, target, and repaired candidate when present, otherwise the initial candidate. It buffers the record in `JsonLDatasetWriter`, increments `index_to_generate`, and resets per-record state. Checkpoint slices are serialized to `output/dataset.jsonl` according to `workflow.save_checks` or on the final accepted record. `save_router` ends when `index_to_generate == target_size`.

## LLM And Inference

`dataset_agent/core/llm/ai_model_client_builder.py` creates one cached chat model from `ai_models.yml`. All LLM nodes reuse it.

| Node | Prompt | Schema |
|---|---|---|
| Generator | `dataset_generator_prompt.yml` | `OutputModel` |
| Validator | `prompt_validator.yml` | `QualityAssessmentModel` |
| Repair | `prompt_repair.yml` | `OutputModel` |

`text_generator.py` provides shared sync and async structured-invocation helpers. It retries schema and output-length failures up to `workflow.generation_schema_fix_retries`; a context-length error becomes a failed generation.

`workflow.yml` sets inference mode. In `vllm` mode, `inference_environment()` starts an OpenAI-compatible server, optionally warms it up, and overrides the LangChain base URL. In `no_inference_engine` mode, LangChain calls the configured provider directly.

With vLLM `invoke_mode: async`, the graph registers async generator, validator, and repair functions and `main.py` uses `ainvoke`. Otherwise it uses synchronous nodes and `invoke`. Prefix caching, chunked prefill, and one cached client reduce repeated inference overhead.

## Configuration And Execution

`settings.py` validates and loads the following files.

| File | Responsibility |
|---|---|
| `config/dataset.yml` | Datasets, labels, distributions, and output target. |
| `config/ai_models.yml` | Shared model provider, identifier, and parameters. |
| `config/dataset_generator_prompt.yml` | Generator prompt templates. |
| `config/prompt_validator.yml` | Validator rubric and prompt templates. |
| `config/prompt_repair.yml` | Repair prompt templates. |
| `config/workflow.yml` | Retry limits, checkpoint cadence, and inference settings. |

`main.py` builds the graph, creates initial state, and runs within `inference_environment()`. That context starts and stops vLLM when configured and clears the cached model after its server lifecycle ends.

The workflow is agentic because specialized workers mutate shared state toward a dataset-level target. Deterministic workers own selection, balancing, discard, and persistence; the shared LLM owns the three reasoning tasks under explicit structured contracts.