from loguru import logger

from use_cases_anthropic_jlens.dataset_agent.models.dataset_generation_state_model import (
    DatasetState,
    DistributionState,
)

node_logger = logger.bind(node="discard")


def _rollback(distribution: DistributionState, class_name: str | None) -> DistributionState:
    if class_name is None or class_name not in distribution:
        return distribution

    bucket = distribution[class_name]
    if bucket.actual == 0:
        return distribution

    updated_distribution = distribution.copy()
    updated_distribution[class_name] = bucket.model_copy(update={"actual": bucket.actual - 1})
    return updated_distribution


def discard_node(state: DatasetState) -> DatasetState:
    source_class = state.source.original_intent if state.source else None
    target_class = state.target.target_intent if state.target else None
    feedback = state.validation_output.feedback if state.validation_output else None

    node_logger.warning(
        "Record {}/{} discarded after {} repair attempts: source={!r}, target={!r}, feedback={!r}",
        state.index_to_generate + 1,
        state.target_size,
        state.retries,
        source_class,
        target_class,
        feedback,
    )

    return state.model_copy(
        update={
            "target_size": max(state.index_to_generate, state.target_size - 1),
            "input_distribution": _rollback(state.input_distribution, source_class),
            "target_distribution": _rollback(state.target_distribution, target_class),
            "source": None,
            "target": None,
            "generated_prompt": None,
            "regenerated_prompt": None,
            "validation_output": None,
            "should_retry": False,
            "retries": 0,
        }
    )
