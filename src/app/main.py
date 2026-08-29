import asyncio

from loguru import logger

from app.logger import setup_logger
from app.settings import settings


def main() -> None:
    """Run dataset generation from the command line."""
    asyncio.run(run())


async def run() -> None:
    setup_logger()

    from dataset_agent.graph_invoker import (
        generate_dataset,
        generate_dataset_async,
        generate_datasets_with_workers,
    )

    inference_mode = settings.workflow.inference.mode
    vllm_settings = settings.workflow.inference.vllm

    logger.info(f"Running inference mode {inference_mode}")

    if settings.workflow.workers > 1:
        logger.info(
            "Executing {} independent workers with forced asynchronous inference",
            settings.workflow.workers,
        )
        await generate_datasets_with_workers()
        return

    if inference_mode == "vllm" and vllm_settings and vllm_settings.invoke_mode == "async":
        logger.info(f"Execute vllm inference in mode: {vllm_settings.invoke_mode}")
        await generate_dataset_async()
    else:
        generate_dataset()


if __name__ == "__main__":
    main()
