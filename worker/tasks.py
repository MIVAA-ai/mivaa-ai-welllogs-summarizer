from config.config_exceptions import ConfigLoadError
from config.config_loader import get_output_settings
from . import app
from utils.SerialiseJson import JsonSerializable
import os
from pathlib import Path
from utils.file_creation_time import get_file_creation_time
from utils.calculate_checksum_and_size import calculate_json_checksum
from utils.IdentifyWellLogFormat import WellLogFormat
import traceback
from scanners.las_scanner import LasScanner
from scanners.dlis_scanner import DLISScanner
from dlisio import dlis
from utils.logger import Logger
from datetime import datetime
from summarise.CSVSummary import CSVSummary
from summarise.WellLogTextInterpretation import InterpretWellLog

# Convert class name string back to class reference
scanner_classes = {
    WellLogFormat.LAS.value: LasScanner,
    WellLogFormat.DLIS.value: DLISScanner
}



def _extract_curve_names(json_data):
    """
    Extracts unique curve names from the given JSON data.

    Args:
        json_data (list): List of parsed JSON records.

    Returns:
        str: Comma-separated string of unique curve names.
    """
    curve_names = set()  # Use a set to automatically remove duplicates
    for record in json_data:
        curves = record.get("curves", [])
        curve_names.update(curve.get("name", "Unknown") for curve in curves)

    return ", ".join(curve_names) if curve_names else "None"

def _consolidate_headers(json_data):
    """
    Consolidates headers from multiple JSON records, ensuring:
    - Duplicate values for the same key are removed.
    - If a key has multiple different values, they are stored as a semicolon-separated string.

    Args:
        json_data (list): List of parsed JSON records.

    Returns:
        dict: Consolidated header dictionary.
    """
    consolidated_header = {}

    for record in json_data:
        current_header = record.get("header", {})

        for key, value in current_header.items():
            if key in consolidated_header:
                if consolidated_header[key] != value:
                    # Convert existing single value to list if not already
                    if not isinstance(consolidated_header[key], list):
                        consolidated_header[key] = [consolidated_header[key]]
                    if value not in consolidated_header[key]:
                        consolidated_header[key].append(value)
            else:
                consolidated_header[key] = value

    # Convert list values back to string format
    for key, value in consolidated_header.items():
        if isinstance(value, list):
            consolidated_header[key] = "; ".join(map(str, value))

    return consolidated_header

def _update_to_csv(file_logger, result):
    """
    Handle the completion of a task by updating the CSV file.
    This function is chained to run after `convert_las_to_json_task`.
    """
    try:
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            raise ValueError(f"Expected result to be a dict, got {type(result).__name__}")

        # Update the CSV file
        csv_summary = CSVSummary(result=result, file_logger=file_logger)
        csv_summary.update_csv()
        file_logger.info(f"CSV updated with task result: {result}")
        return result
    except Exception as e:
        file_logger.error(f"Error updating CSV: {e}")
        return result

@app.task(bind=True)
def convert_to_json_task(self, filepath, output_folder, file_format, logical_file_id=None):
    """
    Generic function to convert LAS or 222DLIS files to JSONWellLogFormat.

    Args:
        self: Celery task context
        filepath (Path): Path to the input file
        output_folder (Path): Path to save the output JSON file
        file_format (WellLogFormat): File format (LAS or DLIS)
        logical_file_id (optional): Logical file object name for DLIS processing

    Returns:
        dict: Result metadata of processing
    """
    # instantiating a logger for each file
    # Initialize basic result structure
    filepath = Path(filepath).resolve()
    output_folder = Path(output_folder).resolve()

    result = {
        "status": "ERROR",
        "task_id": self.request.id,
        "file_name": filepath.name,
        "input_file_format": file_format,
        "input_file_path": str(filepath),
        "input_file_size": os.path.getsize(filepath) if filepath.exists() else "N/A",
        "input_file_creation_user": "Unknown",
        "message": "An error occurred during processing.",
    }

    log_filename = f'{os.path.basename(str(filepath))}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_logger = Logger(log_filename).get_logger()

    file_logger.info(
        f"Task received for processing: {filepath}, Format: {file_format}, Logical File ID: {logical_file_id}")

    try:
        # Load Output Configuration from app_config.yaml
        output_settings = get_output_settings(file_logger)
        csv_output = output_settings["csv_file"]
        json_output = output_settings["json_file"]
        text_interpretation = output_settings["text_interpretation"]
        extract_bulk = output_settings["extract_bulk"]

        #getting some parameters like file creation time and instantiating logical file
        result['input_file_creation_date'] = get_file_creation_time(filepath=filepath, file_logger=file_logger)
        logical_file = None

        # DLIS-specific metadata
        if file_format == WellLogFormat.DLIS.value and logical_file_id is not None:
            logical_files = dlis.load(filepath)
            for single_logical_file in logical_files:
                try:
                    if str(single_logical_file.fileheader.id) == logical_file_id:
                        logical_file = single_logical_file
                except Exception as e:
                    file_logger.error(f"Error accessing logical file header in {filepath}: {e}")
                    continue  # Skip this logical file but continue processing others


        output_filename_suffix = logical_file_id if logical_file_id else ""
        output_interpreted_txt_filename = f"{filepath.stem}{output_filename_suffix}.txt"
        output_json_filename = f"{filepath.stem}{output_filename_suffix}.json"
        output_interpreted_text_file_path = output_folder / output_interpreted_txt_filename
        output_json_file_path = output_folder / output_json_filename

        #updating the result based on the config
        if json_output:
            result["output_json_file"] = str(output_json_file_path)
            result["output_json_file_checksum"] = "Unknown"
            result["output_json_file_size"] = "Unknown"

        if text_interpretation:
            result["output_ai_interpreted_text_file"] = str(output_interpreted_text_file_path)
            result["output_ai_interpreted_text_file_size"] = "Unknown"

        try:
            scanner_cls = scanner_classes[file_format]  # Retrieve actual class

            file_logger.info(f"Scanning {file_format} file: {filepath}{f' (Logical File: {logical_file_id})' if logical_file else ''}...")

            # Initialize scanner
            scanner = scanner_cls(file=filepath, logger=file_logger, extract_bulk=extract_bulk) if not logical_file else scanner_cls(file_path=filepath,
                                                                                      logical_file=logical_file,
                                                                                      logger=file_logger,
                                                                                      extract_bulk=extract_bulk)
            normalised_json = scanner.scan()


            """updating the results with some more metadata like curve names in the data 
            and any other header information present in the file, so that it can be updated in csv later 
            and also be used for text summarisation using llm"""
            if csv_output or text_interpretation:
                file_logger.info(f"Updating the result headers because text summarisation of output to csv is enabled")
                # Extract Curve Names
                result["Curve Names"] = _extract_curve_names(normalised_json)
                # Consolidate Headers to include the headers in results.json
                consolidated_header = _consolidate_headers(normalised_json)
                # Merge result and dynamic headers
                result.update(consolidated_header)
                file_logger.info(f"Updated the result headers because text summarisation of output to csv is enabled")

            # Serialize JSON data and store it in the json file if the configuration is set
            if json_output:
                file_logger.info(f"Serializing scanned data from {filepath}...")
                json_bytes = JsonSerializable.to_json_bytes(normalised_json)

                # Save JSON to file
                file_logger.info(f"Saving JSON data to {output_json_file_path}...")
                with open(output_json_file_path, "wb") as json_file:
                    json_file.write(json_bytes)

                # Calculate checksum of the output JSON file
                checksum = calculate_json_checksum(output_json_file_path)

                result.update({
                    "output_json_file_checksum": checksum,
                    "output_json_file_size": os.path.getsize(output_json_file_path) if output_json_file_path.exists() else "N/A",
                    "message": f"File processed successfully and converted to json: {filepath}",
                })

                result["status"] = "PARTIALLY SUCCESS"
                file_logger.info(f"File converted to JSON: {result}")

            #this is where you can add the task to summarise the json file
            if text_interpretation:
                interpreted_text = InterpretWellLog(
                    result=result,
                    json_data=normalised_json,
                    logger=file_logger
                )
                interpreted_text.json_to_text()

            #setting the status as successful if everything is executed sucessfully
            result["status"] = "SUCCESS"

            #update the csv only if csv output is enabled
            if csv_output:
                result = _update_to_csv(result=result,
                               file_logger=file_logger)

            return result

        except Exception as e:
            result["status"] = "ERROR"
            result["message"] = f"Error processing {file_format} file: {str(e)}"
            file_logger.error(f"Error processing {file_format} file: {e}")
            file_logger.debug(traceback.format_exc())

            if csv_output:
                result = _update_to_csv(result=result,
                                        file_logger=file_logger)

            return result
    except ConfigLoadError as e:
        result["status"] = "ERROR"
        result["message"] = f"Error processing {file_format} file: {str(e)}"
        file_logger.error(f"Error processing {file_format} file: {e}")
        file_logger.debug(traceback.format_exc())
        return result

    except Exception as e:
        result["status"] = "ERROR"
        result["message"] = f"Error processing {file_format} file: {str(e)}"
        file_logger.error(f"Error processing {file_format} file: {e}")
        file_logger.debug(traceback.format_exc())
        return result