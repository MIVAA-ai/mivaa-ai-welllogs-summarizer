import json
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from config.llm_config import get_active_llm
from mappings.WellLogsFormat import WellLogFormat
from utils.cluster_dataframe import cluster_dataframe
from utils.prompt_library import get_prompt_template


def _cleanup_headers(headers):
    """
    Clean up the headers by removing None values and converting them to strings.

    Args:
        headers (dict): The input headers dictionary.

    Returns:
        dict: A new dictionary with cleaned-up headers.
    """
    # Remove all the keys from the headers that have all the values as null
    if not isinstance(headers, dict):
        return headers  # Return the value as is if it's not a dictionary

    # Remove keys where all values are None
    headers = {k: v for k, v in headers.items() if v is not None}

    # Make DLIS-specific changes to the headers
    if headers.get('input_file_format') == WellLogFormat.DLIS.value:
        # Remove specific keys if they exist
        keys_to_remove = ["status", "message", "FILE-ID", "input_file_creation_user",
                          "input_file_creation_date"]  # Add keys to be removed
        for key in keys_to_remove:
            headers.pop(key, None)  # Safely remove key without raising an exception

    return headers

def _cleanup_data(data_section_df, null_values=[-999.25]):
    """
    Clean up the data by removing None values and converting them to strings.

    Args:
        data (dict): The input data dictionary.

    Returns:
        dict: A new dictionary with cleaned-up data.
    """
    #remove all the columns which have null values in the dataframe
    data_section_df = data_section_df.dropna(axis=1, how='all')

    #check if any cell has a values from null_valus list and replace it with None
    data_section_df = data_section_df.replace(null_values, None)

    #replace all the nan values with None
    data_section_df = data_section_df.where(pd.notnull(data_section_df), None)

    #remove duplicate rows
    data_section_df = data_section_df.drop_duplicates()


    return data_section_df


def _merge_dicts_to_dataframe(data_dicts):
    """
    Converts a list of structured dictionaries with 'attributes' and 'objects' keys into a single Pandas DataFrame.

    Args:
        data_dicts (list): A list of dictionaries, each containing 'attributes' and 'objects' keys.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    all_dfs = []  # List to store individual DataFrames

    for data_dict in data_dicts:
        if 'attributes' not in data_dict or 'objects' not in data_dict:
            raise ValueError("Each dictionary must contain 'attributes' and 'objects' keys.")

        attributes = data_dict['attributes']
        objects = data_dict['objects']

        # Convert objects into a DataFrame
        df = pd.DataFrame.from_dict(objects, orient='index', columns=attributes)

        # Move the dictionary key (previous index) to a new column called 'parameter_name'
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'parameter_name'}, inplace=True)

        all_dfs.append(df)

    # Merge all DataFrames vertically (stacking them)
    merged_df = pd.concat(all_dfs, axis=0, ignore_index=False)

    return merged_df

def json_to_text(result, data):
    try:
        #cleaning up headers for las and dlis data
        result = _cleanup_headers(headers=result)

        #cleaning up data if the file is in dlis format
        if result.get('input_file_format') == WellLogFormat.DLIS.value:

            combined_parameters, combined_channels, combined_frames, combined_equipments, combined_tools = [], [], [], [], []

            for sample in data:
                combined_parameters.append(sample['parameters'])

            data_df = _cleanup_data(data_section_df=_merge_dicts_to_dataframe(combined_parameters))

            data_df_clustered = cluster_dataframe(df=data_df, cluster_columns=['parameter_name','description'])
            print(data_df_clustered.to_csv(rf"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\cleaned_parameter.csv", index=False))
            print(None)


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
    except Exception as e:
        print(e)