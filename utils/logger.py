import logging
import os

class Logger:
    def __init__(self, log_filename: str, log_dir: str = None):
        """
        Initializes a logger instance with file and console handlers.

        Args:
            log_filename (str): The name of the log file.
            log_dir (str, optional): The directory where logs should be stored. Defaults to `../logs/`.
        """
        self.log_filename = log_filename

        # Determine log directory (default to '../logs/' if not provided)
        if log_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
            self.log_dir = os.path.abspath(os.path.join(project_root, "..", "logs"))  # Default to 'logs' in root
        else:
            self.log_dir = os.path.abspath(log_dir)  # Ensure absolute path

        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists

        self.log_filepath = os.path.join(self.log_dir, self.log_filename)  # Full path to log file
        self.logger = logging.getLogger(self.log_filename)  # Unique logger per file
        self._setup_logger()

    def _setup_logger(self):
        """ Configures the logger with file and console handlers. """
        self.logger.setLevel(logging.DEBUG)  # Capture all logs

        # Ensure we don't duplicate handlers
        for handler in self.logger.handlers[:]:  # Iterate over a copy of the handlers
            self.logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(self.log_filepath, mode='a')
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Attach handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 🔹 Integrate `dlisio` Logging 🔹
        dlisio_logger = logging.getLogger('dlisio')
        dlisio_logger.setLevel(logging.DEBUG)

        for handler in dlisio_logger.handlers[:]:  # Remove any existing handlers
            dlisio_logger.removeHandler(handler)

        dlisio_logger.addHandler(file_handler)

        # ✅ Force flush on setup (this applies to all handlers)
        self._flush_handlers()

    def _flush_handlers(self):
        """ Flush all log handlers to ensure logs are written immediately. """
        for handler in self.logger.handlers:
            handler.flush()

    def get_logger(self):
        """ Returns the configured logger instance. """
        return self.logger