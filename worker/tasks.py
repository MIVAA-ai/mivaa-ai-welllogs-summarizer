from . import app
from utils.SerialiseJson import JsonSerializable
from worker.result_handler import handle_task_completion
from celery import chain
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
    log_filename = f'{os.path.basename(str(filepath))}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_logger = Logger(log_filename).get_logger()

    file_logger.info(f"Task received for processing: {filepath}, Format: {file_format}, Logical File ID: {logical_file_id}")

    filepath = Path(filepath).resolve()
    output_folder = Path(output_folder).resolve()

    creation_time = get_file_creation_time(filepath=filepath, file_logger=file_logger)
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
    output_filename = f"{filepath.stem}{output_filename_suffix}.json"
    output_file_path = output_folder / output_filename

    # Initialize result structure
    result = {
        "status": "ERROR",
        "task_id": self.request.id,
        "file_name": filepath.name,
        "input_file_format": file_format,
        "input_file_path": str(filepath),
        "input_file_size": os.path.getsize(filepath) if filepath.exists() else "N/A",
        "input_file_creation_date": creation_time,
        "input_file_creation_user": "Unknown",
        "output_file": str(output_file_path),
        "output_file_checksum": "Unknown",
        "output_file_size": "Unknown",
        "message": "An error occurred during processing.",
    }

    try:
        scanner_cls = scanner_classes[file_format]  # Retrieve actual class

        file_logger.info(f"Scanning {file_format} file: {filepath}{f' (Logical File: {logical_file_id})' if logical_file else ''}...")

        # Initialize scanner
        scanner = scanner_cls(file=filepath, logger=file_logger) if not logical_file else scanner_cls(file_path=filepath,
                                                                                  logical_file=logical_file,
                                                                                  logger=file_logger)
        normalised_json = scanner.scan()

        # Extract Curve Names
        result["Curve Names"] = _extract_curve_names(normalised_json)

        # Consolidate Headers
        consolidated_header = _consolidate_headers(normalised_json)

        # Merge result and dynamic headers
        result.update(consolidated_header)

        # Serialize JSON data
        file_logger.info(f"Serializing scanned data from {filepath}...")
        json_bytes = JsonSerializable.to_json_bytes(normalised_json)

        # Save JSON to file
        file_logger.info(f"Saving JSON data to {output_file_path}...")
        with open(output_file_path, "wb") as json_file:
            json_file.write(json_bytes)

        # Calculate checksum of the output JSON file
        checksum = calculate_json_checksum(output_file_path)

        result.update({
            "status": "SUCCESS",
            "output_file_checksum": checksum,
            "output_file_size": os.path.getsize(output_file_path) if output_file_path.exists() else "N/A",
            "message": f"File processed successfully: {filepath}",
        })

        file_logger.info(f"Task completed successfully: {result}")

        #this is where you can add the task to summarise the json file


        # Chain handle_task_completion
        chain(handle_task_completion.s(result=JsonSerializable.to_json(result),
                                       log_filename=log_filename,
                                       initial_task_id=self.request.id)).apply_async()
        return result

    except Exception as e:
        result["status"] = "FAILED"
        result["message"] = f"Error processing {file_format} file: {str(e)}"
        file_logger.error(f"Error processing {file_format} file: {e}")
        file_logger.debug(traceback.format_exc())
        # Chain handle_task_completion
        chain(handle_task_completion.s(result=JsonSerializable.to_json(result),
                                       log_filename=log_filename,
                                       initial_task_id=self.request.id)).apply_async()
        return result