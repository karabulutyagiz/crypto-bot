import logging
import os
from datetime import datetime
from config import TIMEZONE_OFFSET
from datetime import timedelta

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

def get_utc3_time():
    return datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)

class UTC3Formatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = get_utc3_time()
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger
    
    today = get_utc3_time().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"bot_{today}.log")
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = UTC3Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = UTC3Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger("crypto_bot")
