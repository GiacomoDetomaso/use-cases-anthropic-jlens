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

    return state.model_copy(update={
        "validation_output": assessment,
        "should_retry": not assessment.accepted,
    })

def validation_router(state: DatasetState) -> str:
    if state.validation_output and state.validation_output.accepted:
        return "accepted"
    if state.retries < settings.workflow.max_repair_attempts:
        return "repair"
    return "discard"