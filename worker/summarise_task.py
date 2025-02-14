import json
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=r'F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\config\credentials.env')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')

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

    llm = AzureChatOpenAI(azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                          api_version= os.getenv('AZURE_OPENAI_VERSION'),
                          api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                          azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
                          temperature=0)

    paragraph_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Your name is mivaa an AI assistant. You need to act like a oil and gas upstream industry domain expert. Who has an expertise in well log interpretation and analysis."),
            ("human", "Can you summarise following details {user_input} from the {section} of the {file_format} well log file in a paragraphs?"),
        ]
    )

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

    get started with ollama