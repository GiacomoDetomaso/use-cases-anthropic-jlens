"""Builds LangChain chat model clients from `AIModelSettings` configuration."""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.settings import AIModelSettings, settings


def _build_chat_model(ai_model_settings: AIModelSettings) -> BaseChatModel:
    """Instantiate the LangChain chat model wrapper matching `ai_model_settings.provider`."""
    gen_param_dict = {**ai_model_settings.generation_params}
    gen_param_dict["base_url"] = settings.workflow.inference.get_base_url()

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
