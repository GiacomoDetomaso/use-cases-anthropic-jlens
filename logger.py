from pathlib import Path
import sys

from loguru import logger


_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[node]: <10}</cyan> | <magenta>{extra[worker]: <10}</magenta> | "
    "<level>{message}</level>\n{exception}"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[node]: <10} | "
    "{extra[worker]: <10} | "
    "{message}\n{exception}"
)


def setup_logger() -> None:
    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)

    logger.remove()
    logger.configure(extra={"node": "app", "worker": "main"})

    logger.add(
        sys.stdout,
        format=_CONSOLE_FORMAT,
        level="INFO",
        colorize=True,
    )
    logger.add(
        log_directory / "app_{time:YYYY-MM-DD}.log",
        format=_FILE_FORMAT,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        level="DEBUG",
        enqueue=True,
    )
