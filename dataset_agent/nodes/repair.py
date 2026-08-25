from dataset_agent.core.llm.ai_model_client_builder import get_chat_model
from dataset_agent.core.llm.text_generator import (
    FailedGenerationException,
    generate_messages_async,
    generate_messages_sync,
)
from dataset_agent.models.dataset_generation_io_models import OutputModel
from dataset_agent.models.dataset_generation_state_model import DatasetState
from dataset_agent.nodes._prompt_messages import build_messages
from settings import settings
from loguru import logger


def _feedback(state: DatasetState) -> str:
    if state.validation_output is None:
        return "Improve overall quality."
    return state.validation_output.feedback


def repair_node_sync(state: DatasetState) -> DatasetState:
    attempt = state.retries + 1
    logger.info("Starting repair {} for record {}", attempt, state.index_to_generate)
    try:
        repaired_prompt = generate_messages_sync(
            chat_model=get_chat_model(),
            base_messages=build_messages(
                state,
                settings.repair_prompts,
                validation_feedback=_feedback(state),
            ),
            output_schema=OutputModel,
        )
    except (FailedGenerationException, ValueError) as error:
        repaired_prompt = None
        logger.warning(
            "Repair {} failed for record {}: {}",
            attempt,
            state.index_to_generate,
            error,
        )
    else:
        logger.info("Repair {} completed for record {}", attempt, state.index_to_generate)

    return state.model_copy(update={
        "regenerated_prompt": repaired_prompt,
        "retries": state.retries + 1,
        "should_retry": False,
    })


async def repair_node_async(state: DatasetState) -> DatasetState:
    attempt = state.retries + 1
    logger.info("Starting repair {} for record {}", attempt, state.index_to_generate)
    try:
        repaired_prompt = await generate_messages_async(
            chat_model=get_chat_model(),
            base_messages=build_messages(
                state,
                settings.repair_prompts,
                validation_feedback=_feedback(state),
            ),
            output_schema=OutputModel,
        )
    except (FailedGenerationException, ValueError) as error:
        repaired_prompt = None
        logger.warning(
            "Repair {} failed for record {}: {}",
            attempt,
            state.index_to_generate,
            error,
        )
    else:
        logger.info("Repair {} completed for record {}", attempt, state.index_to_generate)

    return state.model_copy(update={
        "regenerated_prompt": repaired_prompt,
        "retries": state.retries + 1,
        "should_retry": False,
    })