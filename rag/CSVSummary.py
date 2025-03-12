import os
import csv
from config.celeryconfig import las_csv_path, dlis_csv_path
from config.celeryconfig import las_header_file_path, dlis_header_file_path
import json
from mappings.WellLogsFormat import WellLogFormat

class CSVSummary:
    def __init__(self, result, file_logger):
        self._result = result
        self._file_logger = file_logger

    def _load_headers(self, file_format):
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

    def _save_headers(self, headers, file_format):
        """
        Save headers to the header file.
        """
        if file_format == WellLogFormat.DLIS.value:
            with open(dlis_header_file_path, "w") as file:
                json.dump(headers, file)  # Save headers as a list
        if file_format == WellLogFormat.LAS.value:
            with open(las_header_file_path, "w") as file:
                json.dump(headers, file)  # Save headers as a list

    def _rewrite_csv_headers(self, global_headers, csv_path):
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
        self._file_logger.info("CSV headers updated successfully.")

    def _append_row_to_csv(self, global_headers, file_format):
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
                    self._file_logger.info("Rewriting CSV headers due to new fields.")
                    self._rewrite_csv_headers(global_headers, csv_path=csv_path)

        # Append the row to the CSV file
        with open(csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=global_headers)
            # writer.writeheader()
            writer.writerow(self._result)
            self._file_logger.info("Appended new row to CSV.")

    def update_csv(self):
        """
        Update the CSV file dynamically based on the result and json_data.
        :param result: Metadata about the LAS to JSON conversion.
        :param json_data: Full JSON data structure including headers, parameters, curves, and data (optional).
        """
        # Load existing headers
        global_headers = self._load_headers(file_format=self._result['input_file_format'])

        # Update global headers while preserving their order
        for header in self._result.keys():
            if header not in global_headers:
                global_headers.append(header)

        # Save the updated headers
        self._save_headers(global_headers, file_format=self._result['input_file_format'])

        # Append the row to the CSV file
        self._append_row_to_csv(global_headers=global_headers, file_format=self._result['input_file_format'])
        self._file_logger.info(f"CSV updated successfully for {self._result['file_name']}")