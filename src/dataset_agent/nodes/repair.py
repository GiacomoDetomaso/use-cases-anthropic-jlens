from loguru import logger

from dataset_agent.core.llm.ai_model_client_builder import get_chat_model
from dataset_agent.core.llm.text_generator import (
    FailedGenerationError,
    generate_messages_async,
    generate_messages_sync,
)
from dataset_agent.models.dataset_generation_io_models import OutputModel
from dataset_agent.models.dataset_generation_state_model import (
    DatasetState,
)
from dataset_agent.nodes._prompt_messages import build_messages
from dataset_agent.settings import settings

node_logger = logger.bind(node="repair")


def _feedback(state: DatasetState) -> str:
    if state.validation_output is None:
        return "Improve overall quality."
    return state.validation_output.feedback


def repair_node_sync(state: DatasetState) -> DatasetState:
    attempt = state.retries + 1
    node_logger.info(
        "Repair {}/{} started for record {}/{}",
        attempt,
        settings.workflow.max_repair_attempts,
        state.index_to_generate + 1,
        state.target_size,
    )
    try:
        repaired_prompt = generate_messages_sync(
            chat_model=get_chat_model().bind(temperature=0),
            base_messages=build_messages(
                state,
                settings.repair_prompts,
                validation_feedback=_feedback(state),
            ),
            output_schema=OutputModel,
        )
    except (FailedGenerationError, ValueError) as error:
        repaired_prompt = None
        node_logger.warning(
            "Repair {}/{} failed for record {}/{}: {}",
            attempt,
            settings.workflow.max_repair_attempts,
            state.index_to_generate + 1,
            state.target_size,
            error,
        )
    else:
        node_logger.success(
            "Repair {}/{} completed for record {}/{}",
            attempt,
            settings.workflow.max_repair_attempts,
            state.index_to_generate + 1,
            state.target_size,
        )

    return state.model_copy(
        update={
            "regenerated_prompt": repaired_prompt,
            "retries": state.retries + 1,
            "should_retry": False,
        }
    )


async def repair_node_async(state: DatasetState) -> DatasetState:
    attempt = state.retries + 1
    node_logger.info(
        "Repair {}/{} started for record {}/{}",
        attempt,
        settings.workflow.max_repair_attempts,
        state.index_to_generate + 1,
        state.target_size,
    )
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
    except (FailedGenerationError, ValueError) as error:
        repaired_prompt = None
        node_logger.warning(
            "Repair {}/{} failed for record {}/{}: {}",
            attempt,
            settings.workflow.max_repair_attempts,
            state.index_to_generate + 1,
            state.target_size,
            error,
        )
    else:
        node_logger.success(
            "Repair {}/{} completed for record {}/{}",
            attempt,
            settings.workflow.max_repair_attempts,
            state.index_to_generate + 1,
            state.target_size,
        )

    return state.model_copy(
        update={
            "regenerated_prompt": repaired_prompt,
            "retries": state.retries + 1,
            "should_retry": False,
        }
    )
