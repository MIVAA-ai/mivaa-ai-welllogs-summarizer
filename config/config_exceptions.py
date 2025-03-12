class ConfigLoadError(Exception):
    """Custom exception for configuration loading failures."""
    def __init__(self, message):
        super().__init__(message)