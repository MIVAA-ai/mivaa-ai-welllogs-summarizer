import os
import yaml
from config_exceptions import ConfigLoadError

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load YAML configuration
def _load_app_config(logger, config_path=os.path.join(script_dir, "app_config.yaml")):
    """
    Reads and parses the application configuration from a YAML file.

    Args:
        config_path (str): Path to the app_config.yaml file.

    Returns:
        dict: Parsed YAML content as a dictionary.
    """
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Error: Config file not found at {config_path}")
        raise ConfigLoadError(f"Missing configuration file: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error: Failed to parse YAML file. Details: {e}")
        raise ConfigLoadError(f"Invalid YAML format in {config_path}")

# Function to get output settings
def get_output_settings(logger):
    """
    Retrieves the 'output' section from the app configuration.

    Returns:
        dict: Output settings (csv_file, json_file, text_interpretation).
    """
    config = _load_app_config(logger=logger)

    # Ensure the "output" section exists
    output_config = config.get("output", {})

    if output_config is None:
        logger.error("Missing 'output' section in app_config.yaml")
        raise ConfigLoadError("Missing 'output' section in app_config.yaml")

    return {
        "csv_file": output_config.get("csv_file", False),
        "json_file": output_config.get("json_file", False),
        "text_interpretation": output_config.get("text_interpretation", False),
        "extract_bulk": output_config.get("extract_bulk", False),
    }