# logger.py

import logging
import sys

# ANSI escape codes for colors
COLOR_CODES = {
    'DEBUG': '\033[94m',     # Blue
    'INFO': '\033[92m',      # Green
    'WARNING': '\033[93m',   # Yellow
    'ERROR': '\033[91m',     # Red
    'CRITICAL': '\033[95m',  # Magenta
    'RESET': '\033[0m'       # Reset to default
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLOR_CODES.get(record.levelname, '')
        reset = COLOR_CODES['RESET']
        # Format includes: timestamp, level, filename, and message
        record.msg = f"{log_color}{record.msg}{reset}"
        formatted = super().format(record)
        return formatted

def get_logger(name=__name__):
    logger = logging.getLogger(name)

    # Prevent multiple handlers if already configured
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
