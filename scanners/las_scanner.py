# """
# This class will take file object as input and return a standardize output for las files
# """

import lasio
import lasio.examples
from mappings.HeaderMappings import HeaderMapping
from utils.DateUtils import DateUtils
from pathlib import Path
from pydantic import ValidationError
import numpy as np
import traceback

class LasScanner:
    def __init__(self, file, logger, extract_bulk=False):
        self._file = file
        self._logger = logger
        self._extract_bulk = extract_bulk


    def scan(self):
        """
         Scans a LAS file and extracts well log data in JSON format.

         Returns:
             list: Standardized JSON output of well log data.
         """
        self._logger.info(f"Scanning LAS file: {self._file}")

        try:
            las_file = lasio.read(self._file, engine="normal", encoding="utf-8")

            # Get different sections of the LAS file in JSON format
            las_headers = self._extract_header(las_file)
            null_value = las_headers.get("null", None)  # Use None if NULL is not defined
            las_curves_headers = self._extract_curve_headers(las_file)


            #extract bulk data only if the self.extract_bulk is set to True
            if self._extract_bulk:
                las_curves_data = self._extract_bulk_data(las_file, null_value)
            else:
                las_curves_data = []

            las_parameters_data = self._extract_parameter_info(las_file)

            # Combine all sections into a single JSON structure
            combined_output = [
                {
                    "header": las_headers,
                    "parameters": las_parameters_data,
                    "curves": las_curves_headers,
                    "data": las_curves_data
                }
            ]

            self._logger.info(f"Successfully scanned LAS file: {self._file}")
            return combined_output
        except Exception as e:
            self._logger.error(f"Error scanning LAS file {self._file}: {e}")
            self._logger.debug(traceback.format_exc())
            return []

    def _extract_bulk_data(self, las_file, null_value):
        """
        Optimized extraction of bulk data (curve measurements) from a LAS file using NumPy.

        Args:
            las_file (lasio.LASFile): The LAS file object.
            null_value (float): The null value to replace NaNs.

        Returns:
            list: A list of data rows, each being an array of values corresponding to the curves.
        """
        try:
            self._logger.info(f"Extracting bulk data for LAS file: {self._file}")

            # Extract data as a NumPy array for faster processing
            curve_data = np.array([curve.data for curve in las_file.curves])

            # Transpose to align rows with indices and columns with curves
            curve_data = curve_data.T

            # Replace NaN values with the specified null value
            if null_value is not None:
                curve_data = np.where(np.isnan(curve_data), null_value, curve_data)

            self._logger.info(f"Successfully extracted bulk data for LAS file: {self._file}")
            # Convert back to a Python list
            return curve_data.tolist()

        except Exception as e:
            self._logger.error(f"Error during bulk data extraction for LAS file {self._file}: {e}")
            self._logger.debug(traceback.format_exc())
            return []

    #extracting only the headers of the well log file
    def _extract_header(self, las_file):
        self._logger.info(f"Extracting header for LAS file: {self._file}")

        #getting the default mapping for the well logs header
        header_mapping = HeaderMapping.get_default_mapping()

        # Build header information based on mapping
        header = {}

        for key, las_fields in header_mapping.items():
            # Handle fields that map to multiple LAS attributes
            header[key] = None
            for las_field in las_fields:
                value = las_file.well[las_field].value if las_field in las_file.well else None

                if value is not None:
                    header[key] = value
                    break  # Stop at the first non-None value

            # Apply ISO 8601 conversion for date fields
            if key in HeaderMapping.get_date_fields() and header[key] is not None:
                header[key] = DateUtils.to_iso8601(header[key])

            # Add any remaining LAS file headers not included in the mapping
            for las_field in las_file.well.keys():
                if las_field not in {field for fields in header_mapping.values() for field in fields}:
                    header[las_field] = las_file.well[las_field].value

            # Putting the name of well log file
            if key in 'name':
                header[key] = Path(self._file).stem

        self._logger.info(f"Successfully extracted header for LAS file: {self._file}")
        return header

    def _extract_curve_headers(self, las_file):
        """
        Extract curve information from a LAS file and structure it according to the provided JSON schema.

        Args:
            las_file (lasio.LASFile): The LAS file object.

        Returns:
            list: A list of curves in the specified JSON schema format.
        """
        self._logger.info(f"Extracting curve headers for LAS file: {self._file}")
        curves = []

        for curve in las_file.curves:
            # Create a dictionary for each curve based on the schema
            curve_data = {
                "name": curve.mnemonic,  # Curve name or mnemonic
                "description": curve.descr if curve.descr else None,  # Curve description
                "quantity": None,  # LAS files typically don't provide quantity
                "unit": curve.unit if curve.unit else None,  # Unit of measurement
                "valueType": "float",  # Defaulting to float for LAS curve data
                "dimensions": 1,  # LAS curve data is 1D
                "axis": [],  # Axis is typically not defined in LAS
                "maxSize": 20  # Default size
            }
            curves.append(curve_data)

        self._logger.info(f"Successfully extracted curve headers for LAS file: {self._file}")
        return curves

    def _extract_parameter_info(self, las_file):
        """
        Extract the parameter information block from a LAS file and structure it into the desired format.

        Args:
            las_file (lasio.LASFile): The LAS file object.

        Returns:
            dict: Parameter information formatted as specified.
        """
        self._logger.info(f"Extracting parameter information for LAS file: {self._file}")

        # Initialize the structure for parameter information
        parameter_info = {
            "attributes": ["value", "unit", "description"],
            "objects": {}
        }

        # Loop through the parameters in the LAS file
        for param in las_file.params:
            # Extract parameter mnemonic, unit, value, and description
            mnemonic = param.mnemonic
            unit = param.unit if param.unit else None
            value = param.value
            description = param.descr if param.descr else None

            # Add parameter to the objects section
            parameter_info["objects"][mnemonic] = [value, unit, description]

        self._logger.info(f"Successfully extracted parameter information for LAS file: {self._file}")
        return parameter_info