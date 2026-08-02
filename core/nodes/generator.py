from core.models.dataset_generation_state_model import DatasetState, OutputModel

from core.utils.llm.ai_model_client_builder import get_generation_chat_model
from core.utils.llm.text_generator import generate, FailedGenerationException


def generator_node(state: DatasetState) -> DatasetState:
    try:
        generated_prompt = generate(
            chat_model=get_generation_chat_model(),
            source=state.source,
            target=state.target,
            output_schema=OutputModel
        )
    except FailedGenerationException:
        generated_prompt = None

    return state.model_copy(update={
        "generated_prompt": generated_prompt
    })

def generation_router(state: DatasetState) -> str:
    if state.generated_prompt is None:
        return "retry"

    return "success"
