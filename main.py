import asyncio

from settings import settings
from logger import setup_logger
from loguru import logger


async def main() -> None:
    setup_logger()

    from dataset_agent.graph_invoker import (
        generate_dataset,
        generate_dataset_async,
        generate_datasets_with_workers
    )
    
    inference_mode = settings.workflow.inference.mode
    invoke_mode = settings.workflow.inference.vllm.invoke_mode

    logger.info(f"Running inference mode {inference_mode}")

    if settings.workflow.workers > 1:
        logger.info(
            "Executing {} independent workers with forced asynchronous inference",
            settings.workflow.workers,
        )
        await generate_datasets_with_workers()
        return

    if inference_mode == "vllm" and invoke_mode == "async":
        logger.info(f"Execute vllm inference in mode: {invoke_mode}")
        await generate_dataset_async()
    else:
        generate_dataset()

if __name__ == "__main__":
    asyncio.run(main())
