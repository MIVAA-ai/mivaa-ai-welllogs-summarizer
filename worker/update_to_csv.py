from . import app
from utils.logger import Logger
from summarise.CSVSummary import CSVSummary

@app.task(bind=True)
def update_to_csv(self, result, log_filename, initial_task_id=None):
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
        csv_summary = CSVSummary(result=result, file_logger=file_logger)
        csv_summary.update_csv()
        file_logger.info(f"CSV updated with task result: {result}")

        # Return a meaningful status
        return f"CSV updated for file: {result['file_name']}"
    except Exception as e:
        file_logger.error(f"Error updating CSV: {e}")
        return f"Error updating CSV for file: {result.get('file_name', 'Unknown')}"