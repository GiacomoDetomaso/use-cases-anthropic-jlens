from loguru import logger
import sys

# Remove default handler
logger.remove()

# Add custom console sink
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}",
    level="INFO"
)

logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="10 MB",     # Create new file when size hits 10 MB (or use "00:00" for daily)
    retention="14 days",  # Automatically delete log files older than 14 days
    compression="zip",    # Compress archived log files
    level="DEBUG"
)

logger.add(
    "logs/async_app.log",
    level="INFO",
    enqueue=True  # Makes logging non-blocking and safe across async tasks
)