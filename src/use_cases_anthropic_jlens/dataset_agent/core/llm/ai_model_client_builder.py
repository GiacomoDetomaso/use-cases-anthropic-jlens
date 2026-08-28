"""Builds LangChain chat model clients from `AIModelSettings` configuration."""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from use_cases_anthropic_jlens.settings import AIModelSettings, settings


def _build_chat_model(ai_model_settings: AIModelSettings) -> BaseChatModel:
    """Instantiate the LangChain chat model wrapper matching `ai_model_settings.provider`."""
    inference_mode = settings.workflow.inference.mode

    gen_param_dict = {**ai_model_settings.generation_params}

    # Update the base_url param with the correct dynamically built one
    if inference_mode == "vllm" and settings.workflow.inference.vllm:
        gen_param_dict["base_url"] = settings.workflow.inference.vllm.get_base_url()

    return init_chat_model(
        model=ai_model_settings.model_name,
        model_provider=ai_model_settings.provider,
        **gen_param_dict,
    )


# TODO build the embedding model


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Lazy singleton accessor for the generation model."""
    return _build_chat_model(settings.ai_models.generation)
