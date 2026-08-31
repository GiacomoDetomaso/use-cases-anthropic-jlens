import pytest


@pytest.fixture
def vllm_configuration(monkeypatch):
    """Configure the shared settings object for tests that require vLLM mode."""
    from app.settings import settings

    monkeypatch.setattr(settings.workflow.inference, "warmup_step", False)
    return settings.workflow.inference
