import yaml
from langchain.prompts import ChatPromptTemplate
from pathlib import Path
import os

config_dir = Path(__file__).resolve().parent.parent
# Load YAML config
def load_prompt_config(config_path=os.path.join(config_dir, "config/prompts.yaml")):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Function to get a prompt template by name
def get_prompt_template(prompt_name):
    config = load_prompt_config()
    if prompt_name not in config["prompts"]:
        raise ValueError(f"Prompt '{prompt_name}' not found in configuration!")

    prompt_data = config["prompts"][prompt_name]

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_data["system"]),
            ("human", prompt_data["human"]),
        ]
    )