"""Builds LangChain chat model clients from `AIModelSettings` configuration."""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from settings import AIModelSettings


def build_chat_model(ai_model_settings: AIModelSettings) -> BaseChatModel:
    """Instantiate the LangChain chat model wrapper matching `ai_model_settings.provider`."""
    return init_chat_model(
        model=ai_model_settings.model_name,
        model_provider=ai_model_settings.provider,
        **ai_model_settings.generation_params,
    )
