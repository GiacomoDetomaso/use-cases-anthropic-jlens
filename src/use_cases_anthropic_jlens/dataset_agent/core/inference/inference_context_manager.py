import contextlib

from loguru import logger

from use_cases_anthropic_jlens.dataset_agent.core.inference.vllm import (
    start_vllm_server,
    stop_vllm_server,
)
from use_cases_anthropic_jlens.dataset_agent.core.llm.ai_model_client_builder import (
    get_chat_model,
)
from use_cases_anthropic_jlens.settings import settings


@contextlib.contextmanager
def inference_environment():
    """Context manager handling model server lifecycle based on active settings."""
    mode = settings.workflow.inference.mode

    if mode == "vllm":
        if settings.workflow.inference.vllm:
            process = start_vllm_server(warmup=settings.workflow.inference.vllm.warmup_step)
        else:
            raise ValueError("VLLM config cannot be None in mode VLLm")
        try:
            yield process
        finally:
            logger.debug("🛑 Terminating vLLM server process...")
            stop_vllm_server(process)

            # Clear cached models if server restarts
            get_chat_model.cache_clear()
    else:
        # Direct LangChain mode (No server process needed)
        yield None
