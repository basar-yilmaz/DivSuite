import logging
import sys
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Define color mapping for different log levels
LOG_COLORS = {
    logging.DEBUG: Fore.BLUE,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
}


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors based on log level.
    """

    def format(self, record):
        log_color = LOG_COLORS.get(record.levelno, "")
        record.msg = f"{log_color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Returns a logger with a colorized console handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create and set a formatter with colors
        formatter = ColoredFormatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
