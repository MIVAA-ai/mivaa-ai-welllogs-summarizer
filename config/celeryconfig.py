"""Celery configuration using local filesystem."""

from pathlib import Path
import os
import warnings

# Suppress CPendingDeprecationWarnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Root directory for storing broker and backend data
_root = Path(__file__).parent.parent.resolve().joinpath("jobs")
_backend_folder = _root.joinpath("results")
_backend_folder.mkdir(exist_ok=True, parents=True)

_folders = {
    "data_folder_in": _root.joinpath("in"),
    "data_folder_out": _root.joinpath("in")  # Must be the same as 'data_folder_in'
}

for folder in _folders.values():
    folder.mkdir(exist_ok=True)

# Create 'summarise' folder for the CSV file
_summary_folder = _root.joinpath("summarise")
_summary_folder.mkdir(exist_ok=True, parents=True)

# Path for the scanned_file.csv in the 'summarise' folder
las_csv_path = _summary_folder / "las_scanned_files.csv"
las_header_file_path = _summary_folder / "las_headers.json"  # Persistent header storage
dlis_csv_path = _summary_folder / "dlis_scanned_files.csv"
dlis_header_file_path = _summary_folder / "dlis_headers.json"  # Persistent header storage

# Celery configuration
broker_url = "filesystem://localhost//"
result_backend = f"file:///{os.path.normpath(_backend_folder).replace(os.sep, '/')}"
broker_transport_options = {k: str(v) for k, v in _folders.items()}
task_serializer = "json"
persist_results = True
result_serializer = "json"
accept_content = ["json"]
imports = ("worker.tasks",)

# New setting to avoid deprecation warning
broker_connection_retry_on_startup = True