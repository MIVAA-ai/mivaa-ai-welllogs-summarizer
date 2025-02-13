from datetime import datetime
from pathlib import Path
import os
from utils.logger import Logger

def get_file_creation_time(filepath, file_logger):
    """
    Get the file creation time in a cross-platform manner.
    :param filepath: Path to the file.
    :return: File creation time as a formatted string.
    """
    filepath = Path(filepath)
    try:
        if os.name == 'nt':  # For Windows
            creation_time = filepath.stat().st_ctime
        else:  # For Unix-like systems
            # Use st_birthtime on systems like macOS; fallback to st_ctime
            creation_time = getattr(filepath.stat(), 'st_birthtime', filepath.stat().st_ctime)

        # Convert timestamp to human-readable format
        return datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        file_logger.error(f"Error fetching file creation time: {e}")
        return "Unknown"