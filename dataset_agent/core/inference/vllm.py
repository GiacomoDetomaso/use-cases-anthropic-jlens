import subprocess
import time
import requests
import threading

from settings import settings

from loguru import logger

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

def start_vllm_server() -> subprocess.Popen:
    """Reads YAML config and launches a background vLLM OpenAI-compatible server."""
    model_name = settings.ai_models.generation.model_name

    health_iteration = 0

    # Construct CLI command dynamically
    cmd = settings.workflow.inference.vllm.get_vllm_cmd(model_name=model_name)

    logger.info(f"🚀 Launching vLLM server for model: {model_name}...")
    
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
                health_iteration += 1

                if health_iteration == 1:
                    logger.info("✅ vLLM server is up and ready to accept requests!")
                elif health_iteration > 1:
                    logger.info("✅ vLLM server is up and running!")

                break
        except requests.exceptions.ConnectionError as e:
            logger.warning("Attempting to connect")
        
        # Check if process died prematurely (e.g. OOM or wrong path)
        if process.poll() is not None:
            raise RuntimeError("vLLM server process failed to start. Check terminal output/logs.")
            
        time.sleep(settings.workflow.inference.vllm.health_sleep_seconds)
        
    return process
