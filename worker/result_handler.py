import os
import csv
from .celeryconfig import las_csv_path, dlis_csv_path
from .celeryconfig import las_header_file_path, dlis_header_file_path
from . import app
import json
from utils.logger import Logger
from mappings.WellLogsFormat import WellLogFormat

def load_headers(file_format):
    """
    Load headers from the header file.
    Creates an empty list if the file does not exist.
    """
    if file_format == WellLogFormat.DLIS.value:
        if os.path.exists(dlis_header_file_path):
            with open(dlis_header_file_path, "r") as file:
                return json.load(file)  # Return as a list
    if file_format == WellLogFormat.LAS.value:
        if os.path.exists(las_header_file_path):
            with open(las_header_file_path, "r") as file:
                return json.load(file)  # Return as a list
    return []  # Return an empty list if the file does not exist

def save_headers(headers, file_format):
    """
    Save headers to the header file.
    """
    if file_format == WellLogFormat.DLIS.value:
        with open(dlis_header_file_path, "w") as file:
            json.dump(headers, file)  # Save headers as a list
    if file_format == WellLogFormat.LAS.value:
        with open(las_header_file_path, "w") as file:
            json.dump(headers, file)  # Save headers as a list


def append_row_to_csv(row, global_headers, file_format, file_logger):
    """
    Append a row to the CSV file without rewriting the entire file.
    If new headers are added, the file header is updated.
    """
    # Check if the file exists
    if file_format == WellLogFormat.DLIS.value:
        file_exists = os.path.exists(dlis_csv_path)
        csv_path = dlis_csv_path
    elif file_format == WellLogFormat.LAS.value:
        file_exists = os.path.exists(las_csv_path)
        csv_path = las_csv_path
    else:
        file_exists = None
        csv_path = None

    # If the file exists, ensure headers are updated
    if file_exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            current_headers = reader.fieldnames or []

            # If new headers are found, rewrite the header only
            if set(global_headers) != set(current_headers):
                file_logger.info("Rewriting CSV headers due to new fields.")
                rewrite_csv_headers(global_headers, csv_path=csv_path, file_logger=file_logger)

    # Append the row to the CSV file
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=global_headers)
        writer.writerow(row)
        file_logger.info("Appended new row to CSV.")

def rewrite_csv_headers(global_headers, csv_path, file_logger):
    """
    Rewrite only the headers of the CSV file without rewriting rows.
    """
    # Read existing rows
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

    # Write back rows with updated headers
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=global_headers)
        writer.writeheader()
        writer.writerows(rows)
    file_logger.info("CSV headers updated successfully.")

def update_csv(result, file_logger):
    """
    Update the CSV file dynamically based on the result and json_data.
    :param result: Metadata about the LAS to JSON conversion.
    :param json_data: Full JSON data structure including headers, parameters, curves, and data (optional).
    """
    # Load existing headers
    global_headers = load_headers(file_format=result['input_file_format'])

    # Update global headers while preserving their order
    for header in result.keys():
        if header not in global_headers:
            global_headers.append(header)

    # Save the updated headers
    save_headers(global_headers, file_format=result['input_file_format'])

    # Append the row to the CSV file
    append_row_to_csv(result, global_headers, file_format=result['input_file_format'], file_logger=file_logger)
    file_logger.info(f"CSV updated successfully for {result['file_name']}")

@app.task(bind=True)
def handle_task_completion(self, result, log_filename, initial_task_id=None):
    """
    Handle the completion of a task by updating the CSV file.
    This function is chained to run after `convert_las_to_json_task`.
    """
    file_logger = Logger(log_filename).get_logger()
    try:
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            raise ValueError(f"Expected result to be a dict, got {type(result).__name__}")

        # Combine initial task ID with the current task ID
        combined_task_ids = f"{initial_task_id}, {self.request.id}"
        result["task_id"] = combined_task_ids

        # Update the CSV file
        update_csv(result, file_logger)
        file_logger.info(f"CSV updated with task result: {result}")

        # Return a meaningful status
        return f"CSV updated for file: {result['file_name']}"
    except Exception as e:
        file_logger.error(f"Error updating CSV: {e}")
        return f"Error updating CSV for file: {result.get('file_name', 'Unknown')}"