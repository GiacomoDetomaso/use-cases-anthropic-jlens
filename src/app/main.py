from loguru import logger

from app.logger import setup_logger
from app.settings import settings


async def run() -> None:
    setup_logger()

    from dataset_agent.graph_invoker import (
        generate_dataset,
        generate_dataset_async,
        generate_datasets_with_workers,
    )

    if settings.workflow.workers > 1:
        logger.info(
            "Executing {} independent workers with forced asynchronous inference",
            settings.workflow.workers,
        )
        await generate_datasets_with_workers()
        return

    if settings.workflow.inference.invoke_mode == "async":
        logger.info("Execute vLLM inference in async mode")
        await generate_dataset_async()
    else:
        generate_dataset()
