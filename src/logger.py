import logging
from typing import Optional

def get_console_logger(name: Optional[str] = "exchange_rates") -> logging.Logger:

    logger = logging.getLogger(name)
    
    # Create a logger if one doesn't exist
    if not logger.handlers:

        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Set a format for the outputs of the stream handler
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)
        
    return logger