from dataset_agent.core.llm.ai_model_client_builder import get_chat_model
from dataset_agent.core.llm.text_generator import (
    FailedGenerationException,
    generate_messages_async,
    generate_messages_sync,
)
from dataset_agent.models.dataset_generation_io_models import (
    QualityAssessmentModel,
)
from dataset_agent.models.dataset_generation_state_model import DatasetState
from dataset_agent.nodes._prompt_messages import build_messages
from settings import settings
from loguru import logger


node_logger = logger.bind(node="validate")


def _rejected_assessment(reason: str) -> QualityAssessmentModel:
    return QualityAssessmentModel(
        original_intent_preserved=False,
        attack_present=False,
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
    log = node_logger.success if assessment.accepted else node_logger.warning
    candidate = state.regenerated_prompt or state.generated_prompt
    candidate_kind = "repaired" if state.regenerated_prompt is not None else "generated"
    log(
        "Record {}/{} {}: intent_preserved={}, attack_matches_pattern={}, "
        "feedback={!r}, {}_prompt={!r}",
        state.index_to_generate + 1,
        state.target_size,
        "accepted" if assessment.accepted else "rejected",
        assessment.original_intent_preserved,
        assessment.attack_present,
        assessment.feedback,
        candidate_kind,
        candidate.text if candidate is not None else None,
    )

def validation_router(state: DatasetState) -> str:
    if state.validation_output and state.validation_output.accepted:
        return "accepted"
    if state.retries < settings.workflow.max_repair_attempts:
        node_logger.info(
            "Scheduling repair {}/{} for record {}/{}",
            state.retries + 1,
            settings.workflow.max_repair_attempts,
            state.index_to_generate + 1,
            state.target_size,
        )
        return "repair"
    node_logger.warning(
        "Discarding record {}/{} after {} repair attempts",
        state.index_to_generate + 1,
        state.target_size,
        state.retries,
    )
    return "discard"