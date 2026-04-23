# modules/debug.py

"""
Debug Module

This module contains helper functions to facilitate debugging.
It sets up logging to output debug messages to both a log file and the console.
"""

import logging
from config import LOG_LEVEL
import warnings

# Ignore specific matplotlib font warnings
warnings.filterwarnings("ignore", message=".*findfont.*", category=UserWarning)

def setup_logging(log_file: str = '../data/logs/model_debug.log') -> None:
    """
    Set up logging configuration.
    
    This function configures logging so that all debug-level (and above)
    messages are written to the specified log file (which is overwritten
    on each run) and also output to the console.
    
    Args:
        log_file (str): Path to the log file (default 'model_debug.log').
    """
    log_level = LOG_LEVEL
    # Configure logging to file
    logging.basicConfig(
        level=log_level, # logging.DEBUG is more verbose
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'  # Overwrite the file on each run
    )
    # Create a console handler with the same log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)

def log_state(state: dict, message: str = "State update:") -> None:
    """
    Log the current state of the simulation.
    
    Args:
        state (dict): The simulation state (e.g., a dictionary of current values).
        message (str): A message prefix for the log entry (default "State update:").
    """
    logging.debug(f"{message} {state}")
