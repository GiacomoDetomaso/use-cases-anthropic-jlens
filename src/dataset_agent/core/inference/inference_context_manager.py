import contextlib

from loguru import logger

from app.settings import settings
from dataset_agent.core.inference.vllm import (
    start_vllm_server,
    stop_vllm_server,
)
from dataset_agent.core.llm.ai_model_client_builder import (
    get_chat_model,
)


@contextlib.contextmanager
def inference_environment():
    """Start and stop the configured vLLM server for one generation run."""
    process = start_vllm_server(warmup=settings.workflow.inference.warmup_step)
    try:
        yield process
    finally:
        logger.debug("Stopping vLLM server")
        stop_vllm_server(process)
        get_chat_model.cache_clear()
