import pytest


@pytest.fixture
def vllm_configuration(monkeypatch):
    """Configure the shared settings object for tests that require vLLM mode."""
    from app.settings import settings

    inference = settings.workflow.inference
    vllm_settings = inference.vllm
    assert vllm_settings is not None

    monkeypatch.setattr(inference, "mode", "vllm")
    monkeypatch.setattr(vllm_settings, "warmup_step", False)
    return vllm_settings
