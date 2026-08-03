import subprocess
import time
import requests

from settings import settings

from loguru import logger

def start_vllm_server() -> subprocess.Popen:
    """Reads YAML config and launches a background vLLM OpenAI-compatible server."""
    model_name = settings.ai_models.generation.model_name

    # Construct CLI command dynamically
    cmd = settings.workflow.inference.vllm.get_vllm_cmd(model_name=model_name)

    logger.info(f"🚀 Launching vLLM server for model: {model_name}...")
    
    # Spawn background process
    process = subprocess.Popen(cmd)

    # Wait for server readiness
    health_url = settings.workflow.inference.vllm.get_health_url()
    
    while True:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                logger.debug("✅ vLLM server is up and ready to accept requests!")
                break
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            pass
        
        # Check if process died prematurely (e.g. OOM or wrong path)
        if process.poll() is not None:
            raise RuntimeError("vLLM server process failed to start. Check terminal output/logs.")
            
        time.sleep(settings.workflow.inference.vllm.health_sleep_seconds)
        
    return process
