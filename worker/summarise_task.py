import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.llm_config import get_active_llm
from utils.prompt_library import get_prompt_template

def _remove_null_values(d):
    """
    Recursively removes keys with None values from a dictionary.

    Args:
        d (dict): The input dictionary.

    Returns:
        dict: A new dictionary without None values.
    """
    if not isinstance(d, dict):
        return d  # Return the value as is if it's not a dictionary

    return {k: _remove_null_values(v) for k, v in d.items() if v is not None}

def create_json_summary(result, data):

    result = _remove_null_values(result)

    print("\n===== Result Summary =====")
    print(json.dumps(result, indent=4, sort_keys=False))  # Indented and readable

    print("\n===== Data Summary =====")
    print(json.dumps(data, indent=4, sort_keys=False))

    llm = get_active_llm()

    paragraph_prompt_template = get_prompt_template("dlis_header_paragraph_summary")

    paragraph_chain = paragraph_prompt_template | llm | StrOutputParser()

    # Define the values for the template
    variables = {
        "user_input": result,
        "section": "headers",
        "file_format": "DLIS"
    }


    # Invoke the chain with variables
    response = paragraph_chain.invoke(variables)

    print(response)