from scanners.DLISProcessorBase import DLISProcessorBase

class DLISZoneProcessor(DLISProcessorBase):
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

    def extract_zones(self):
        """
        Extracts and processes equipment data into a JSON-like format.

        Returns:
            dict: Processed equipment data.
        """
        attributes = {
            "name": "name",
            "domain": "domain",
            "maximum": "maximum",
            "minimum": "minimum"
        }
        units_relevant_columns = [
            "maximum", "minimum"
        ]

        return self.process_items(attributes, units_relevant_columns)
