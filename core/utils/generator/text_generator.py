"""Invokes the generation chat model to produce a new malicious prompt."""
from langchain_core.language_models import BaseChatModel

from core.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
    OutputModel,
)
from core.utils.generator.prompt_builder import build_generation_messages


class TextGenerator:
    def __init__(self, chat_model: BaseChatModel):
        self._structured_chat_model = chat_model.with_structured_output(OutputModel)

    def generate(self, source: InputDatasetModel, target: InputAttackModel) -> OutputModel:
        messages = build_generation_messages(source, target)
        return self._structured_chat_model.invoke(messages)
