import os
import yaml
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama

# Load environment variables from .env
script_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path= os.path.join(script_dir, "credentials.env"))


print(f"Current Working Directory: {os.getcwd()}")

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')


# Load YAML config
def load_config(config_path=os.path.join(script_dir, "llm_config.yaml")):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# LLM Factory function
def get_active_llm():
    config = load_config()
    provider = config["llm"]["provider"]

    if provider == "AzureChatOpenAI":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature=config["llm"]["azure"]["temperature"],
        )

    elif provider == "ollama":
        return ChatOllama(
            model=config["llm"]["ollama"]["model"],
            temperature=config["llm"]["ollama"]["temperature"],
            num_predict=config["llm"]["ollama"].get("num_predict", None),  # Optional
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")