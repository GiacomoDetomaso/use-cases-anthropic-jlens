import contextlib

from settings import settings
from core.utils.inference.vllm import start_vllm_server

from llm.ai_model_client_builder import get_generation_chat_model, get_validation_chat_model

@contextlib.contextmanager
def inference_environment():
    """Context manager handling model server lifecycle based on active settings."""
    mode = settings.workflow.inference.mode

    if mode == "vllm":
        process = start_vllm_server()
        try:
            yield process
        finally:
            print("🛑 Terminating vLLM server process...")
            process.terminate()
            process.wait()

            # Clear cached models if server restarts
            get_generation_chat_model.cache_clear()
            get_validation_chat_model.cache_clear()
    else:
        # Direct LangChain mode (No server process needed)
        yield None
