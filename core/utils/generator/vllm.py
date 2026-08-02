import subprocess
import time
import requests
import contextlib

from settings import settings

def _start_vllm_server() -> subprocess.Popen:
    """Reads YAML config and launches a background vLLM OpenAI-compatible server."""
    model_name = settings.ai_models.generation.model_name

    # Construct CLI command dynamically
    cmd = settings.workflow.inference.vllm.get_vllm_cmd(model_name=model_name)

    print(f"🚀 Launching vLLM server for model: {model_name}...")
    
    # Spawn background process
    process = subprocess.Popen(cmd)

    # Wait for server readiness
    health_url = settings.workflow.inference.vllm.get_health_url()
    
    while True:
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                print("✅ vLLM server is up and ready to accept requests!")
                break
        except requests.exceptions.ConnectionError:
            pass
        
        # Check if process died prematurely (e.g. OOM or wrong path)
        if process.poll() is not None:
            raise RuntimeError("vLLM server process failed to start. Check terminal output/logs.")
            
        time.sleep(3)
        
    return process


@contextlib.contextmanager
def vllm_server_context(config_path: str, port: int = 8000):
    """Context manager to auto-start and cleanly shut down vLLM."""
    process = _start_vllm_server(config_path, port)

    try:
        yield process
    finally:
        print("🛑 Terminating vLLM server process...")
        process.terminate()
        process.wait()
