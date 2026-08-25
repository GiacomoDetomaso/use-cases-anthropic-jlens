from dataset_agent.core.llm.ai_model_client_builder import get_chat_model
from dataset_agent.core.llm.text_generator import (
    FailedGenerationException,
    generate_messages_async,
    generate_messages_sync,
)
from dataset_agent.models.dataset_generation_io_models import (
    QualityAssessmentModel,
    QualityLevel,
)
from dataset_agent.models.dataset_generation_state_model import DatasetState
from dataset_agent.nodes._prompt_messages import build_messages
from settings import settings
from loguru import logger


def _rejected_assessment(reason: str) -> QualityAssessmentModel:
    return QualityAssessmentModel(
        intent_context_preservation=QualityLevel.VERY_LOW,
        attack_class_alignment=QualityLevel.VERY_LOW,
        originality=QualityLevel.VERY_LOW,
        naturalness_and_coherence=QualityLevel.VERY_LOW,
        feedback=reason,
    )


def validator_node_sync(state: DatasetState) -> DatasetState:
    try:
        assessment = generate_messages_sync(
            chat_model=get_chat_model(),
            base_messages=build_messages(state, settings.validator_prompts),
            output_schema=QualityAssessmentModel,
        )
    except (FailedGenerationException, ValueError) as error:
        assessment = _rejected_assessment(f"Validation inference failed: {error}")

    _log_assessment(state, assessment)
    return state.model_copy(update={
        "validation_output": assessment,
        "should_retry": not assessment.accepted,
    })


async def validator_node_async(state: DatasetState) -> DatasetState:
    try:
        assessment = await generate_messages_async(
            chat_model=get_chat_model(),
            base_messages=build_messages(state, settings.validator_prompts),
            output_schema=QualityAssessmentModel,
        )
    except (FailedGenerationException, ValueError) as error:
        assessment = _rejected_assessment(f"Validation inference failed: {error}")

    _log_assessment(state, assessment)
    return state.model_copy(update={
        "validation_output": assessment,
        "should_retry": not assessment.accepted,
    })


def _log_assessment(state: DatasetState, assessment: QualityAssessmentModel) -> None:
    log = logger.info if assessment.accepted else logger.warning
    log(
        "Validation {} for record {}: overall={}, intent={}, attack={}, originality={}, "
        "naturalness={}, feedback={!r}",
        "accepted" if assessment.accepted else "rejected",
        state.index_to_generate,
        assessment.overall_level.value,
        assessment.intent_context_preservation.value,
        assessment.attack_class_alignment.value,
        assessment.originality.value,
        assessment.naturalness_and_coherence.value,
        assessment.feedback,
    )

def validation_router(state: DatasetState) -> str:
    if state.validation_output and state.validation_output.accepted:
        return "accepted"
    if state.retries < settings.workflow.max_repair_attempts:
        logger.info(
            "Scheduling repair {} of {} for record {}",
            state.retries + 1,
            settings.workflow.max_repair_attempts,
            state.index_to_generate,
        )
        return "repair"
    logger.warning(
        "Discarding record {} after {} repair attempts",
        state.index_to_generate,
        state.retries,
    )
    return "discard"