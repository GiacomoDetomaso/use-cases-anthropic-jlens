import subprocess
import threading
import time
from typing import Any, IO
from weakref import WeakKeyDictionary

import requests
from loguru import logger

from use_cases_anthropic_jlens.settings import settings

_shutdown_events: WeakKeyDictionary[subprocess.Popen[str], threading.Event] = WeakKeyDictionary()


def _warmup_server(url: str, model_name: str) -> None:
    logger.info("⏳ Warming up vLLM server...")

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Warmup request"}],
        "max_tokens": 10,
    }

    # Retry until server is ready and warmed up
    for _ in range(30):
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                print("✅ Warmup complete! JIT kernels compiled.")
                return
        except requests.exceptions.RequestException:
            time.sleep(2)


def _forward_logs(pipe: IO[Any], shutting_down: threading.Event) -> None:
    for line in iter(pipe.readline, ""):
        line = line.rstrip()

        if not line:
            continue

        if shutting_down.is_set():
            logger.debug("[vllm] {}", line)
        elif " ERROR " in line:
            logger.error("[vllm] {}", line)
        elif " WARNING " in line:
            logger.warning("[vllm] {}", line)
        else:
            logger.info("[vllm] {}", line)

    pipe.close()


def start_vllm_server(warmup: bool) -> subprocess.Popen[str]:
    """Reads YAML config and launches a background vLLM OpenAI-compatible server."""
    model_name = settings.ai_models.generation.model_name

    # Construct CLI command dynamically
    vllm_settings = settings.workflow.inference.vllm

    if vllm_settings is None:
        raise ValueError("VLLM cannot be none if modality is VLLM")
    cmd = vllm_settings.get_vllm_cmd(model_name=model_name)

    logger.info(f"🚀 Launching vLLM server for model: {model_name}...")
    logger.info(f"Using the following commad: \n\n{cmd}")

    # Spawn background process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    shutdown_event = threading.Event()
    _shutdown_events[process] = shutdown_event
    assert process.stdout is not None

    threading.Thread(
        target=_forward_logs,
        args=(process.stdout, shutdown_event),
        daemon=True,
    ).start()

    # Wait for server readiness
    health_url = vllm_settings.get_health_url()

    while True:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                logger.info("✅ vLLM server is up and running!")

                if warmup:
                    _warmup_server(
                        url=f"{vllm_settings.get_base_url()}/chat/completions",
                        model_name=model_name,
                    )

                break
        except requests.exceptions.ConnectionError:
            logger.warning("Attempting to connect")

        # Check if process died prematurely (e.g. OOM or wrong path)
        if process.poll() is not None:
            raise RuntimeError("vLLM server process failed to start. Check terminal output/logs.")

        time.sleep(vllm_settings.health_sleep_seconds)

    return process


def stop_vllm_server(process: subprocess.Popen[str], timeout_seconds: float = 10) -> None:
    """Stop a vLLM server while suppressing expected shutdown output.

    Parameters
    ----------
    process : subprocess.Popen
        vLLM API server process returned by :func:`start_vllm_server`.
    timeout_seconds : float, default=10
        Maximum graceful shutdown wait before the process is killed.
    """
    shutdown_event = _shutdown_events.pop(process, None)
    if shutdown_event is not None:
        shutdown_event.set()

    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM did not exit within {} seconds; killing it", timeout_seconds)
        process.kill()
        process.wait()
