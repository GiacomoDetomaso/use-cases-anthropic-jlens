from dataset_agent.models.dataset_generation_state_model import (
    DatasetState,
    DistributionState,
    OutputModel,
)

from dataset_agent.core.llm.ai_model_client_builder import get_chat_model
from dataset_agent.core.llm.text_generator import (
    FailedGenerationException,
    generate_sync,
    generate_async,
)

from loguru import logger


node_logger = logger.bind(node="generate")


def _rollback_class_selection(
    distribution: DistributionState,
    class_name: str | None,
    distribution_name: str,
) -> DistributionState:
    if class_name is None:
        return distribution

    bucket = distribution[class_name]

    if bucket.actual == 0:
        node_logger.warning(
            "Cannot release {} class {!r}: its allocation is already zero",
            distribution_name,
            class_name,
        )
        return distribution

    updated_distribution = distribution.copy()
    updated_distribution[class_name] = bucket.model_copy(
        update={"actual": bucket.actual - 1}
    )
    return updated_distribution


def _failed_generation_update(state: DatasetState) -> dict:
    source_class = state.source.original_intent if state.source else None
    target_class = state.target.target_intent if state.target else None

    return {
        "generated_prompt": None,
        "target_size": max(state.index_to_generate, state.target_size - 1),
        "input_distribution": _rollback_class_selection(
            state.input_distribution, source_class, "input"
        ),
        "target_distribution": _rollback_class_selection(
            state.target_distribution, target_class, "target"
        ),
    }


def generator_node_sync(state: DatasetState) -> DatasetState:
    node_logger.info(
        "Generating record {}/{}: source={!r}, target={!r}",
        state.index_to_generate + 1,
        state.target_size,
        state.source.original_intent if state.source else None,
        state.target.target_intent if state.target else None,
    )
    try:
        generated_prompt = generate_sync(
            chat_model=get_chat_model(),
            source=state.source,
            target=state.target,
            output_schema=OutputModel
        )
    except FailedGenerationException:
        node_logger.warning(
            "Generation failed for record {}/{}; releasing selected classes",
            state.index_to_generate + 1,
            state.target_size,
        )
        return state.model_copy(update=_failed_generation_update(state))

    node_logger.success(
        "Generated candidate for record {}/{}",
        state.index_to_generate + 1,
        state.target_size,
    )
    return state.model_copy(update={"generated_prompt": generated_prompt})


async def generator_node_async(state: DatasetState) -> DatasetState:
    node_logger.info(
        "Generating record {}/{}: source={!r}, target={!r}",
        state.index_to_generate + 1,
        state.target_size,
        state.source.original_intent if state.source else None,
        state.target.target_intent if state.target else None,
    )
    try:
        generated_prompt = await generate_async(
            chat_model=get_chat_model(),
            source=state.source,
            target=state.target,
            output_schema=OutputModel,
        )
    except FailedGenerationException:
        node_logger.warning(
            "Generation failed for record {}/{}; releasing selected classes",
            state.index_to_generate + 1,
            state.target_size,
        )
        return state.model_copy(update=_failed_generation_update(state))

    node_logger.success(
        "Generated candidate for record {}/{}",
        state.index_to_generate + 1,
        state.target_size,
    )
    return state.model_copy(update={"generated_prompt": generated_prompt})

def generation_router(state: DatasetState) -> str:
    if state.generated_prompt is None:
        if state.index_to_generate >= state.target_size:
            return "completed"
        return "retry"

    return "success"
