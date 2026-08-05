import asyncio

from dataset_agent.graph import build_and_compile_graph, get_graph_initial_state
from dataset_agent.core.inference.inference_context_manager import inference_environment

from settings import settings

from logger import setup_logger

from loguru import logger

def generate_dataset() -> None:
    initial_state = get_graph_initial_state()
    graph = build_and_compile_graph()

    # Wrap graph execution inside the environment
    with inference_environment():
        graph.invoke(initial_state)


async def generate_dataset_async() -> None:
    initial_state = get_graph_initial_state()
    graph = build_and_compile_graph()
    
    # Wrap graph execution inside the environment
    with inference_environment():
        await graph.ainvoke(initial_state)

def main() -> None:
    setup_logger()
    inference_mode = settings.workflow.inference.mode
    invoke_mode = settings.workflow.inference.vllm.invoke_mode

    logger.info(f"Running inference mode {inference_mode}")

    if inference_mode == "vllm" and invoke_mode == "async":
        logger.info(f"Execute vllm inference in mode: {invoke_mode}")
        asyncio.run(generate_dataset_async())
    else:
        generate_dataset()

if __name__ == "__main__":
    main()
