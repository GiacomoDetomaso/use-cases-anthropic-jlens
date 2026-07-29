from core.models.dataset_generation_state_model import DatasetState
from core.utils.generator.ai_model_client_builder import build_chat_model
from core.utils.generator.text_generator import TextGenerator
from settings import AIModelSettings, settings


class GeneratorNode:
    def __init__(self, ai_model_settings: AIModelSettings | None = None):
        ai_model_settings = ai_model_settings or settings.ai_models.generation
        self.text_generator = TextGenerator(build_chat_model(ai_model_settings))

    def __call__(self, state: DatasetState) -> DatasetState:
        if state.source is None or state.target is None:
            raise ValueError("GeneratorNode requires both 'source' and 'target' to be set in state")

        generated_prompt = self.text_generator.generate(state.source, state.target)

        return state.model_copy(update={"generated_prompt": generated_prompt})
