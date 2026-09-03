import pandas as pd
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


def build_training_dataset() -> pd.DataFrame:
    """Build and return the training feature dataset.

    This synchronous entry point is intended for notebook environments such as
    Kaggle, where it can be invoked after project configuration is available.
    """
    setup_logger()

    from training.dataset.build_dataset import build

    logger.info("Starting training dataset build")
    dataset = build()
    logger.info("Training dataset build completed with {} rows", len(dataset))
    return dataset
