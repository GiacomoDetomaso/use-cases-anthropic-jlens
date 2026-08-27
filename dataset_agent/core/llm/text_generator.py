"""Invokes the generation chat model to produce a new malicious prompt."""
from openai import BadRequestError, LengthFinishReasonError

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, ValidationError

from dataset_agent.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
)

from settings import settings

from loguru import logger

_INTERNAL_RETRIES = settings.workflow.generation_schema_fix_retries
node_logger = logger.bind(node="model")

class FailedGenerationException(Exception):
    def __init__(self, *args, retries: int=_INTERNAL_RETRIES):
        super().__init__(*args)
        self.retries = retries


def _is_context_length_error(error: BadRequestError) -> bool:
    error_message = str(error).lower()
    return (
        "maximum context length" in error_message
        or "context length" in error_message and "input_tokens" in error_message
    )


def _build_generation_messages(source: InputDatasetModel, target: InputAttackModel) -> list[BaseMessage]:
    prompt_settings = settings.prompts

    user_content = prompt_settings.user.format(
        original_prompt=source.original_prompt,
        original_intent=source.original_intent,
        target_intent=target.target_intent,
        target_description=target.target_description,
        target_examples=target.target_examples,
    )

    return [
        SystemMessage(content=prompt_settings.system),
        HumanMessage(content=user_content),
    ]


def _prepare_generation(
    chat_model: BaseChatModel,
    source: InputDatasetModel,
    target: InputAttackModel,
    output_schema: type[BaseModel],
):
    return (
        chat_model.with_structured_output(output_schema, include_raw=True),
        _build_generation_messages(source, target),
    )


def _parse_generation_response(response) -> BaseModel:
    if response["parsing_error"]:
        raise response["parsing_error"]

    node_logger.debug("Structured model response parsed")
    return response["parsed"]


def _handle_generation_error(
    error: Exception,
    retries: int,
) -> tuple[int, str | None]:
    if isinstance(error, LengthFinishReasonError):
        retries += 1
        node_logger.warning(
            "Output token limit reached; retrying model invocation ({}/{})",
            retries,
            _INTERNAL_RETRIES,
        )
        return retries, (
            "Generation exceeded the output token limit. "
            "Return a complete schema-valid response with significantly "
            "more concise fields."
        )

    if isinstance(error, BadRequestError) and _is_context_length_error(error):
        node_logger.warning(
            "Input and requested output exceed the model context window; ending invocation"
        )
        raise FailedGenerationException(retries=retries) from error

    if isinstance(error, ValidationError):
        retries += 1
        node_logger.warning(
            "Structured response failed validation; retrying model invocation ({}/{})",
            retries,
            _INTERNAL_RETRIES,
        )
        return retries, (
            "Generate the answer again. Return only a complete answer "
            "matching the required output schema. Validation errors:\n"
            f"{error.json(indent=2)}"
        )

    raise error


def _build_input_messages(
    base_messages: list[BaseMessage],
    retry_instruction: str | None,
) -> list[BaseMessage]:
    messages = list(base_messages)

    if retry_instruction is not None:
        messages.append(HumanMessage(content=retry_instruction))

    return messages


def generate_messages_sync(
    chat_model: BaseChatModel,
    base_messages: list[BaseMessage],
    output_schema: type[BaseModel],
) -> BaseModel:
    llm = chat_model.with_structured_output(output_schema, include_raw=True)

    retries = 0
    retry_instruction: str | None = None

    while retries < _INTERNAL_RETRIES:
        try:
            node_logger.debug(
                "Invoking structured model (attempt {}/{})",
                retries + 1,
                _INTERNAL_RETRIES,
            )

            response = llm.invoke(
                _build_input_messages(base_messages, retry_instruction)
            )

            return _parse_generation_response(response)
        except Exception as error:
            retries, retry_instruction = _handle_generation_error(error, retries)

    node_logger.error(
        "Model invocation failed after {}/{} attempts",
        retries,
        _INTERNAL_RETRIES,
    )

    raise FailedGenerationException(retries=retries)


async def generate_messages_async(
    chat_model: BaseChatModel,
    base_messages: list[BaseMessage],
    output_schema: type[BaseModel],
) -> BaseModel:
    llm = chat_model.with_structured_output(output_schema, include_raw=True)

    retries = 0
    retry_instruction: str | None = None

    while retries < _INTERNAL_RETRIES:
        try:
            node_logger.debug(
                "Invoking structured model asynchronously (attempt {}/{})",
                retries + 1,
                _INTERNAL_RETRIES,
            )

            response = await llm.ainvoke(
                _build_input_messages(base_messages, retry_instruction)
            )
            
            return _parse_generation_response(response)
        except Exception as error:
            retries, retry_instruction = _handle_generation_error(error, retries)

    node_logger.error(
        "Asynchronous model invocation failed after {}/{} attempts",
        retries,
        _INTERNAL_RETRIES,
    )

    raise FailedGenerationException(retries=retries)


def generate_sync(
    chat_model: BaseChatModel,
    source: InputDatasetModel,
    target: InputAttackModel,
    output_schema: type[BaseModel],
) -> BaseModel:
    return generate_messages_sync(
        chat_model=chat_model,
        base_messages=_build_generation_messages(source, target),
        output_schema=output_schema,
    )


async def generate_async(
    chat_model: BaseChatModel,
    source: InputDatasetModel,
    target: InputAttackModel,
    output_schema: type[BaseModel],
) -> BaseModel:
    return await generate_messages_async(
        chat_model=chat_model,
        base_messages=_build_generation_messages(source, target),
        output_schema=output_schema,
    )
