from scanners.DLISProcessorBase import DLISProcessorBase

class DLISParametersProcessor(DLISProcessorBase):
    """
    Processes the parameters in a DLIS logical file and handles extraction and transformation.
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

    def extract_parameters(self):
        """
        Extracts and processes parameters into a JSON-like format.

        Returns:
            dict: Processed parameter data.
        """
        attributes = {
            "name": "name",
            "values": "values",
            "long_name": "description",
            "dimension": "dimension",
            "axis": "axis",
            "zones": "zones"
        }
        units_relevant_columns = ["values"]
        related_columns = ["axis", "zones"]

        return self.process_items(attributes, units_relevant_columns, related_columns=related_columns)