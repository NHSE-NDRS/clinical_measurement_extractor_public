import logging
from datetime import datetime
import os


def setup_logging(
        enable_console:bool,
        enable_file:bool,
        console_log_level:int,
        log_dir: str = "logs"
    ):
        """
        Set up logging with fixed settings:
        - Logs always saved to 'logs/' directory with timestamped file name
        - File logs only include INFO and above
        - Console logging optional, with user-defined verbosity
    
        Args:
            enable_console (bool): Whether to log to the console.
            enable_file (bool): Whether to save logs to a file (INFO and above).
            console_log_level (int): Level for console output (e.g., logging.INFO).
            log_dir (str): name of the logging directory.
        """
        os.makedirs(log_dir, exist_ok=True)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
        # Clear previous handlers (important in notebooks/scripts)
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()
    
        root_logger.setLevel(logging.INFO) # This gets overwritten later on.
    
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    
        if enable_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)  # Only INFO and above
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_log_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
    
        logging.getLogger(__name__).info(
            f"Logging initialized. Log file: {log_file if enable_file else 'Disabled'}"
        )