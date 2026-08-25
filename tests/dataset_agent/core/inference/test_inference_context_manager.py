from unittest.mock import MagicMock

from dataset_agent.core.inference import inference_context_manager
from settings import settings


def test_vllm_context_yields_server_and_cleans_up(monkeypatch, vllm_configuration):
    process = MagicMock()
    start_server = MagicMock(return_value=process)
    chat_model_accessor = MagicMock()

    monkeypatch.setattr(inference_context_manager, "start_vllm_server", start_server)
    monkeypatch.setattr(inference_context_manager, "get_chat_model", chat_model_accessor)

    with inference_context_manager.inference_environment() as yielded_process:
        assert yielded_process is process

    start_server.assert_called_once_with(warmup=vllm_configuration.warmup_step)
    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with()
    chat_model_accessor.cache_clear.assert_called_once_with()


def test_no_inference_engine_context_yields_none_without_starting_vllm(monkeypatch):
    start_server = MagicMock()
    monkeypatch.setattr(settings.workflow.inference, "mode", "no_inference_engine")
    monkeypatch.setattr(inference_context_manager, "start_vllm_server", start_server)

    with inference_context_manager.inference_environment() as yielded_process:
        assert yielded_process is None

    start_server.assert_not_called()