"""Invokes the generation chat model to produce a new malicious prompt."""
from unittest.mock import Base

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, ValidationError

from core.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
)

from settings import settings

_INTERNAL_RETRIES = settings.workflow.generation_schema_fix_retries

class FailedGenerationException(Exception):
    def __init__(self, *args, retries: int=_INTERNAL_RETRIES):
        super().__init__(*args)
        self.retries = retries


def _build_generation_messages(source: InputDatasetModel, target: InputAttackModel) -> list[BaseMessage]:
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


def generate(
    chat_model: BaseChatModel,
    source: InputDatasetModel,
    target: InputAttackModel,
    output_schema: type[BaseModel]
) -> type[BaseModel]:
    llm = chat_model.with_structured_output(output_schema)

    messages = _build_generation_messages(source, target)

    retries = 0

    while retries < _INTERNAL_RETRIES:
        try:
            response = llm.invoke(messages)

            return response
        except ValidationError as e:
            messages.append(AIMessage(response))
            messages.append(HumanMessage("The provided response does not satisfy the output schema. Read the error and fix it."))
            messages.append(HumanMessage(e.json(indent=4)))

            retries += 1

    raise FailedGenerationException(retries=retries)
