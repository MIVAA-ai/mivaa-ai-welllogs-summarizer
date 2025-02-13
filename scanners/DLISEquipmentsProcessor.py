from scanners.DLISProcessorBase import DLISProcessorBase

class DLISEquipmentsProcessor(DLISProcessorBase):
    """
    Processes the equipment data in a DLIS logical file and handles extraction and transformation.
    """
    def __init__(self, logical_file_id, items, logger):
        """
        Initialize the DLISParametersProcessor.

        Args:
            logical_file_id (str): Unique identifier for the logical file.
            items (list): List of parameter objects.
            logger: Logger instance.
        """
        super().__init__(logical_file_id, items, logger)  # Pass logger to base class

    def extract_equipments(self):
        """
        Extracts and processes equipment data into a JSON-like format.

        Returns:
            dict: Processed equipment data.
        """
        attributes = {
            "name": "name",
            "trademark_name": "trademark_name",
            "status": "status",
            "generic_type": "generic_type",
            "serial_number": "serial_number",
            "location": "location",
            "height": "height",
            "length": "length",
            "diameter_min": "diameter_min",
            "diameter_max": "diameter_max",
            "volume": "volume",
            "weight": "weight",
            "hole_size": "hole_size",
            "pressure": "pressure",
            "temperature": "temperature",
            "vertical_depth": "vertical_depth",
            "radial_drift": "radial_drift",
            "angular_drift": "angular_drift"
        }
        units_relevant_columns = [
            "height", "length", "diameter_min", "diameter_max", "volume",
            "weight", "hole_size", "pressure", "temperature", "vertical_depth",
            "radial_drift", "angular_drift"
        ]

        return self.process_items(attributes, units_relevant_columns)
