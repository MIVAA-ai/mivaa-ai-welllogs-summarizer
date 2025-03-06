import os
import yaml
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from mappings.WellLogsFormat import WellLogFormat
from mappings.WellLogsSections import WellLogsSections
from .config_exceptions import ConfigLoadError

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, "credentials.env"))
CONFIG_PATH = os.path.join(script_dir, "config.yaml")


# Function to load YAML configuration
def _load_config(logger, config_path=CONFIG_PATH):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise ConfigLoadError(f"Missing configuration file: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file. Details: {e}")
        raise ConfigLoadError(f"Invalid YAML format in {config_path}")


# Function to get output settings
def get_output_settings(logger):
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
def get_summary_prompt_template(logger, prompt_name):
    config = _load_config(logger)
    summary_prompts = config.get("summary_prompts", {})

    if prompt_name not in summary_prompts:
        logger.error(f"Prompt '{prompt_name}' not found in config.yaml")
        raise ConfigLoadError(f"Prompt '{prompt_name}' not found in config.yaml")

    template_str = summary_prompts[prompt_name]

    if "{context}" not in template_str:
        template_str += "\n\n{context}"

    return PromptTemplate(template=template_str, input_variables=["context"])

# Consolidated function to retrieve summary prompt names
def get_summary_prompt_name(logger, file_format, subsection_name):
    format_prompt_map = {
        WellLogFormat.DLIS.value: "stuff_dlis_paragraph_summary",
        WellLogFormat.LAS.value: "stuff_chat_las_paragraph_summary"
    }

    try:
        prompt_base = format_prompt_map[file_format]
        return prompt_base
    except KeyError:
        logger.error(f"Prompt not found for format: {file_format}, section: {subsection_name}")
        raise ConfigLoadError(f"Prompt not found for format: {file_format}, section: {subsection_name}")
