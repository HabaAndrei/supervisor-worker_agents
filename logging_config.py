"""
Simple Logging Configuration for Errors and User Actions

This module provides a simple logging setup focused on:
- Logging errors with details
- Tracking important user steps/actions
- Simple file output with dates
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_file: str = "app.log"):
    """
    Set up simple logging for errors and user actions.

    Args:
        log_file: Name of the log file (will be created in 'logs' directory)
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Full path to log file
    log_path = log_dir / log_file

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove default handlers

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()


def log_error(
    message: str, error: Exception = None, user_id: str = None, details: str = None
):
    """
    Log an error with details.

    Args:
        message: Description of what went wrong
        error: The exception object (optional)
        user_id: ID of the user who encountered the error (optional)
        details: Details
    """
    error_msg = f"ERROR: {message}"

    if user_id:
        error_msg += f" | User: {user_id}"

    if error:
        error_msg += f" | Exception: {str(error)}"

    if details:
        error_msg += f" | Details: {details}"

    logger.error(error_msg, stacklevel=2)


def log_user_action(action: str, user_id: str = None, details: str = None):
    """
    Log an important user action.

    Args:
        action: What the user did (e.g., "login", "upload_file", "delete_project")
        user_id: ID of the user (optional)
        details: Additional details about the action (optional)
    """
    action_msg = f"USER ACTION: {action}"

    if user_id:
        action_msg += f" | User: {user_id}"

    if details:
        action_msg += f" | Details: {details}"

    logger.info(action_msg, stacklevel=2)


def log_important_step(step: str, details: str = None):
    """
    Log an important step in the application.

    Args:
        step: Description of the step
        details: Additional details (optional)
    """
    step_msg = f"STEP: {step}"

    if details:
        step_msg += f" | {details}"

    logger.info(step_msg, stacklevel=2)


def log_message(message: str):
    """
    Log a simple message.

    Args:
        message: The message to log
    """
    logger.info(message, stacklevel=2)