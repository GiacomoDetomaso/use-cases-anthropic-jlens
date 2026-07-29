"""Renders the configured generator prompt template into LangChain chat messages."""
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from core.models.dataset_generation_io_models import InputAttackModel, InputDatasetModel
from settings import settings


def build_generation_messages(source: InputDatasetModel, target: InputAttackModel) -> list[BaseMessage]:
    prompt_settings = settings.prompts

    user_content = prompt_settings.user.format(
        original_prompt=source.original_prompt,
        original_intent=source.original_intent,
        target_intent=target.target_intent,
        target_examples=target.target_examples,
    )

    return [
        SystemMessage(content=prompt_settings.system),
        HumanMessage(content=user_content),
    ]
