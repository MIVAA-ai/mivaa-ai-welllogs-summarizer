import hashlib
import orjson
from pathlib import Path

def calculate_json_checksum(filepath, algorithm="blake2b"):
    """
    Calculates a stable checksum of a JSON file by normalizing it first.

    Args:
        filepath (str or Path): Path to the JSON file.
        algorithm (str, optional): Hash algorithm (default: 'blake2b').

    Returns:
        dict: Dictionary containing checksum and file size.
    """
    filepath = Path(filepath)
    hash_func = hashlib.new(algorithm)

    with open(filepath, "rb") as f:
        try:
            data = orjson.loads(f.read())  # Load JSON efficiently
        except orjson.JSONDecodeError:
            return {"checksum": None, "file_size": None, "error": "Invalid JSON"}

    # **Normalization step**
    normalized_json = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)

    # Compute the checksum
    hash_func.update(normalized_json)

    return hash_func.hexdigest()