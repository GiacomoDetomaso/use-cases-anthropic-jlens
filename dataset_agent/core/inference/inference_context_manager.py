import contextlib

from settings import settings
from dataset_agent.core.inference.vllm import start_vllm_server

from dataset_agent.core.llm.ai_model_client_builder import (
    get_chat_model, 
)

from loguru import logger

@contextlib.contextmanager
def inference_environment():
    """Context manager handling model server lifecycle based on active settings."""
    mode = settings.workflow.inference.mode

    if mode == "vllm":
        process = start_vllm_server(settings.workflow.inference.vllm.warmup_step)
        try:
            yield process
        finally:
            logger.debug("🛑 Terminating vLLM server process...")
            process.terminate()
            process.wait()

            # Clear cached models if server restarts
            get_chat_model.cache_clear()
    else:
        # Direct LangChain mode (No server process needed)
        yield None
