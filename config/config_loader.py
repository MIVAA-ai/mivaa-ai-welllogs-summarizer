import os
import yaml
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from .config_exceptions import ConfigLoadError
import traceback

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, "credentials.env"))
CONFIG_PATH = os.path.join(script_dir, "config.yaml")


# Function to load YAML configuration
def _load_config(logger, config_path=CONFIG_PATH):
    """
    Reads and parses the unified configuration file.

    Args:
        logger: Logger instance.
        config_path (str): Path to the config.yaml file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        ConfigLoadError: If the config file is missing or invalid.
    """
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise ConfigLoadError(f"Missing configuration file: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file. Details: {e}")
        raise ConfigLoadError(f"Invalid YAML format in {config_path}")


# Function to get Output settings
def get_output_settings(logger):
    """
    Retrieves the 'output' section from the configuration.

    Returns:
        dict: Output settings (csv_file, json_file, text_interpretation, extract_bulk).
    """
    config = _load_config(logger)
    output_config = config.get("output", {})

    if not output_config:
        logger.error("Missing 'output' section in config.yaml")
        raise ConfigLoadError("Missing 'output' section in config.yaml")

    return {
        "csv_file": output_config.get("csv_file", False),
        "json_file": output_config.get("json_file", False),
        "text_interpretation": output_config.get("text_interpretation", False),
        "extract_bulk": output_config.get("extract_bulk", False),
    }


# Function to get the LLM instance
def get_active_llm(logger):
    """
    Initializes the LLM instance based on the config.

    Returns:
        LLM instance.
    """
    config = _load_config(logger)
    llm_config = config.get("llm")

    if not llm_config:
        logger.error("Missing 'llm' section in config.yaml")
        raise ConfigLoadError("Missing 'llm' section in config.yaml")

    provider = llm_config.get("provider")

    if provider == "AzureChatOpenAI":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature=llm_config.get("azure", {}).get("temperature", 0.7),
        )

    elif provider == "ChatOllama":
        return ChatOllama(
            model=llm_config.get("ollama", {}).get("model", "default-model"),
            temperature=llm_config.get("ollama", {}).get("temperature", 0.7),
            num_predict=llm_config.get("ollama", {}).get("num_predict", None),
        )

    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ConfigLoadError(f"Unsupported LLM provider: {provider}")


# Function to get a prompt template by name
def get_prompt_template(logger, prompt_name):
    """
    Retrieves the specified prompt template.

    Returns:
        ChatPromptTemplate: The LangChain prompt template.
    """
    config = _load_config(logger)
    prompts = config.get("prompts", {})

    if prompt_name not in prompts:
        logger.error(f"Prompt '{prompt_name}' not found in config.yaml")
        raise ConfigLoadError(f"Prompt '{prompt_name}' not found in config.yaml")

    prompt_data = prompts[prompt_name]

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_data["system"]),
            ("human", prompt_data["human"]),
        ]
    )
