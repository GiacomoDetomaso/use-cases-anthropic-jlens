import subprocess
import time
import requests
import threading

from settings import settings

from loguru import logger


def _warmup_server(url: str, model_name: str):
    logger.info("⏳ Warming up vLLM server...")

    payload = {
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


def _forward_logs(pipe):
    for line in iter(pipe.readline, ""):
        line = line.rstrip()

        if not line:
            continue

        if " ERROR " in line:
            logger.error("[vllm] {}", line)
        elif " WARNING " in line:
            logger.warning("[vllm] {}", line)
        else:
            logger.info("[vllm] {}", line)

    pipe.close()

def start_vllm_server(warmup: bool) -> subprocess.Popen:
    """Reads YAML config and launches a background vLLM OpenAI-compatible server."""
    model_name = settings.ai_models.generation.model_name

    # Construct CLI command dynamically
    cmd = settings.workflow.inference.vllm.get_vllm_cmd(model_name=model_name)

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

    threading.Thread(
        target=_forward_logs,
        args=(process.stdout,),
        daemon=True,
    ).start()

    # Wait for server readiness
    health_url = settings.workflow.inference.vllm.get_health_url()
    
    while True:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:

                logger.info("✅ vLLM server is up and running!")

                if warmup:
                    _warmup_server(
                        url=f"{settings.workflow.inference.vllm.get_base_url()}/chat/completions",
                        model_name=model_name
                    )

                break
        except requests.exceptions.ConnectionError as e:
            logger.warning("Attempting to connect")
        
        # Check if process died prematurely (e.g. OOM or wrong path)
        if process.poll() is not None:
            raise RuntimeError("vLLM server process failed to start. Check terminal output/logs.")
            
        time.sleep(settings.workflow.inference.vllm.health_sleep_seconds)
        
    return process

