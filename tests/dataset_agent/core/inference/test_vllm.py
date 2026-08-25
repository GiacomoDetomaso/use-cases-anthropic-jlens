from unittest.mock import MagicMock

import pytest
import requests

from dataset_agent.core.inference import vllm


def test_start_vllm_server_starts_process_and_waits_for_healthcheck(
    monkeypatch,
    vllm_configuration,
):
    process = MagicMock()
    process.stdout = MagicMock()
    command = ["vllm", "serve", "test-model"]
    popen = MagicMock(return_value=process)
    thread = MagicMock()
    health_response = MagicMock(status_code=200)

    monkeypatch.setattr(
        type(vllm_configuration),
        "get_vllm_cmd",
        MagicMock(return_value=command),
    )
    monkeypatch.setattr(vllm.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm.threading, "Thread", thread)
    monkeypatch.setattr(vllm.requests, "get", MagicMock(return_value=health_response))

    result = vllm.start_vllm_server(warmup=False)

    assert result is process
    popen.assert_called_once_with(
        command,
        stdout=vllm.subprocess.PIPE,
        stderr=vllm.subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    thread.assert_called_once_with(
        target=vllm._forward_logs,
        args=(process.stdout,),
        daemon=True,
    )
    thread.return_value.start.assert_called_once_with()


def test_start_vllm_server_raises_when_process_dies_before_healthcheck(
    monkeypatch,
    vllm_configuration,
):
    process = MagicMock()
    process.stdout = MagicMock()
    process.poll.return_value = 1

    monkeypatch.setattr(vllm.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(vllm.threading, "Thread", MagicMock())
    monkeypatch.setattr(
        vllm.requests,
        "get",
        MagicMock(side_effect=requests.exceptions.ConnectionError),
    )
    monkeypatch.setattr(vllm.time, "sleep", MagicMock())

    with pytest.raises(RuntimeError, match="failed to start"):
        vllm.start_vllm_server(warmup=False)

    process.poll.assert_called_once_with()