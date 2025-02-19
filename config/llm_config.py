import os
import yaml
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from config_exceptions import ConfigLoadError

# Load environment variables from .env
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, "credentials.env"))

print(f"Current Working Directory: {os.getcwd()}")

# Ensure required environment variables are loaded
required_env_vars = [
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_TRACING_V2",
]

for var in required_env_vars:
    value = os.getenv(var)
    if value is None:
        print(f"Warning: {var} is not set in environment variables.")
    else:
        os.environ[var] = value


# Load YAML config with error handling
def _load_config(logger, config_path=os.path.join(script_dir, "llm_config.yaml")):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise ConfigLoadError(f"Missing configuration file: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file. Details: {e}")
        raise ConfigLoadError(f"Invalid YAML format in {config_path}")


# LLM Factory function
def get_active_llm(logger):
    config = _load_config(logger)

    # Ensure "llm" exists in the YAML file
    if "llm" not in config:
        logger.error("Missing 'llm' section in llm_config.yaml")
        raise ConfigLoadError("Missing 'llm' section in llm_config.yaml")

    provider = config["llm"].get("provider")

    if provider == "AzureChatOpenAI":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature=config["llm"].get("azure", {}).get("temperature", 0.7),
        )

    elif provider == "ollama":
        return ChatOllama(
            model=config["llm"].get("ollama", {}).get("model", "default-model"),
            temperature=config["llm"].get("ollama", {}).get("temperature", 0.7),
            num_predict=config["llm"].get("ollama", {}).get("num_predict", None),  # Optional
        )

    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ConfigLoadError(f"Unsupported LLM provider: {provider}")