from mappings.HeaderMappings import HeaderMapping
from utils.date_utils import DateUtils
import pandas as pd
from utils.dlis_utils import parse_value, process_dataframe_lists
import traceback

def _get_first_matching_value(fields, origins_df):
    """
    Retrieves the first matching value for a list of fields from the origins DataFrame.

    Args:
        fields (list): A list of field names to search for.
        origins_df (pd.DataFrame): DataFrame containing extracted origins.

    Returns:
        Any: The first non-None value found, or None if no match.
    """
    for field in fields:
        match = origins_df[origins_df["name"] == field]
        if not match.empty:
            return match["value"].iloc[0]
    return None


def _add_unmapped_fields(header, header_mapping, origins_df):
    """
    Adds fields from the origins DataFrame that are not part of the header mapping.

    Args:
        header (dict): The header dictionary being constructed.
        header_mapping (dict): The header mapping used for transformation.
        origins_df (pd.DataFrame): DataFrame containing extracted origins.
    """
    mapped_fields = {field for fields in header_mapping.values() for field in fields}
    for _, row in origins_df.iterrows():
        field_name = row["name"]
        if field_name not in mapped_fields:
            header[field_name] = row["value"]


class DLISOriginsProcessor:
    """
    Processes the origins in a DLIS logical file and handles extraction and transformation.
    """

    def __init__(self, logical_file_id, origins, logger):
        """
        Initialize the DLISOriginsProcessor.

        Args:
            logical_file_id (str): Unique identifier for the logical file.
            origins (list): List of origin objects in the logical file.
        """
        self._logical_file_id = logical_file_id
        self._origins = origins
        self._logger = logger  # Store the logger

    def map_headers(self):
        """
        Maps the extracted origins to the header mapping.

        Args:
            origins_df (pd.DataFrame): DataFrame containing extracted origins.

        Returns:
            dict: A dictionary containing the transformed metadata.
        """
        try:
            self._logger.info(f"Mapping headers for logical file {self._logical_file_id}...")

            origins_df = self._extract_origins()
            header_mapping = HeaderMapping.get_default_mapping()
            header = {}

            # Map fields based on the header mapping
            for key, fields in header_mapping.items():
                header[key] = _get_first_matching_value(fields, origins_df)

                # Convert date fields to ISO 8601 if required
                if key in HeaderMapping.get_date_fields() and header[key]:
                    header[key] = DateUtils.to_iso8601(header[key])

            # Add unmapped fields
            _add_unmapped_fields(header, header_mapping, origins_df)

            # Add the logical file ID
            header["name"] = self._logical_file_id

            self._logger.info(f"Successfully mapped headers for logical file {self._logical_file_id}")

            return header

        except Exception as e:
            self._logger.error(f"Error mapping headers for logical file {self._logical_file_id}: {e}")
            self._logger.debug(traceback.format_exc())  # Logs the stack trace for debugging
            return {}

    def _extract_origins(self):
        """
        Extracts and processes origins into a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing origins data.
        """
        try:
            self._logger.info(f"Extracting origins for logical file {self._logical_file_id}...")

            # Extract and process origins
            first_origin, *remaining_origins = self._origins
            origin_list = self._process_origin_attributes(first_origin)

            for origin in remaining_origins:
                origin_list.extend(self._process_origin_attributes(origin))

            # Convert list to DataFrame and clean
            origins_df = pd.DataFrame(origin_list)
            if not origins_df.empty:
                origins_df = process_dataframe_lists(origins_df, logger=self._logger)

            self._logger.info(f"Successfully extracted origins for logical file {self._logical_file_id}")

            return origins_df

        except Exception as e:
            self._logger.error(f"Error while extracting origins for logical file {self._logical_file_id}: {e}")
            self._logger.debug(traceback.format_exc())  # Logs the stack trace for debugging            return pd.DataFrame()

    def _process_origin_attributes(self, origin):
        """
        Processes attributes of a single origin and returns them as a list of dictionaries.

        Args:
            origin: The origin object.

        Returns:
            list: A list of dictionaries containing processed origin attributes.
        """
        origin_list = []
        for key, value in origin.attributes.items():
            try:
                if origin[key] is not None:
                    parsed_value = parse_value(origin[key])
                    origin_list.append({
                        "name": key,
                        "value": parsed_value,
                        "logical-file-id": self._logical_file_id,
                    })
            except ValueError as ve:
                origin_list.append({
                    "name": key,
                    "value": "Unable to extract",  # Default value when error occurs
                    "logical-file-id": self._logical_file_id,
                })
                self._logger.warning(f"Skipping attribute '{key}' due to error: {ve}")
                self._logger.debug(traceback.format_exc())  # Logs the full traceback
            except Exception as ve:
                origin_list.append({
                    "name": key,
                    "value": "Unable to extract",  # Default value when error occurs
                    "logical-file-id": self._logical_file_id,
                })
                self._logger.error(f"Error processing attribute '{key}' in logical file {self._logical_file_id}: {ve}")
                self._logger.debug(traceback.format_exc())  # Logs the full traceback
            return origin_list