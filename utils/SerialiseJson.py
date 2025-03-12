# import orjson
# import numpy as np
# from pydantic import BaseModel
#
# class JsonSerializable:
#     @staticmethod
#     def to_json(obj):
#         """
#         Converts an object to a JSON string using `orjson`, ensuring all numpy types and Pydantic models are serializable.
#
#         Args:
#             obj (any): The object to convert and serialize.
#
#         Returns:
#             str: The JSON string representation of the object.
#         """
#
#         def convert(item):
#             """Efficiently converts complex data types into JSON-serializable structures."""
#             if isinstance(item, BaseModel):
#                 return item.dict()
#             elif isinstance(item, (np.integer, np.floating, np.generic)):  # Convert numpy scalars
#                 return item.item()
#             elif isinstance(item, np.ndarray):
#                 return item.tolist()  # Convert numpy arrays to lists
#             elif isinstance(item, (list, tuple)):
#                 return [convert(v) for v in item]  # Recursively process lists and tuples
#             elif isinstance(item, dict):
#                 return {k: convert(v) for k, v in item.items()}  # Recursively process dictionaries
#             else:
#                 return item  # Return unchanged if not special
#
#         # Apply the conversion
#         serializable_obj = convert(obj)
#
#         # Use orjson for better performance
#         return orjson.dumps(serializable_obj, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS).decode()


import orjson
import numpy as np
from pydantic import BaseModel

class JsonSerializable:
    @staticmethod
    def to_json(obj):
        """
        Converts an object to a JSON-serializable dictionary using `orjson`,
        ensuring all numpy types and Pydantic models are properly handled.

        Args:
            obj (any): The object to convert and serialize.

        Returns:
            dict: A fully JSON-serializable dictionary.
        """

        def convert(item):
            """Efficiently converts complex data types into JSON-serializable structures."""
            if isinstance(item, BaseModel):
                return item.dict()
            elif isinstance(item, (np.integer, np.floating, np.generic)):  # Convert numpy scalars
                return item.item()
            elif isinstance(item, np.ndarray):
                return item.tolist()  # Convert numpy arrays to lists
            elif isinstance(item, (list, tuple)):
                return [convert(v) for v in item]  # Recursively process lists and tuples
            elif isinstance(item, dict):
                return {str(k): convert(v) for k, v in item.items()}  # Ensure all keys are strings
            else:
                return item  # Return unchanged if not special

        # Convert and return a dictionary for Celery compatibility
        return convert(obj)

    @staticmethod
    def to_json_bytes(obj):
        """
        Converts an object to a JSON string using `orjson`, ensuring all numpy types and Pydantic models are serializable.

        Args:
            obj (any): The object to convert and serialize.

        Returns:
            bytes: Optimized JSON in bytes (for fast file writes).
        """
        return orjson.dumps(JsonSerializable.to_json(obj), option=orjson.OPT_INDENT_2)