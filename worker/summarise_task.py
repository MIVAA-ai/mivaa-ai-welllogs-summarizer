import json
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from config.llm_config import get_active_llm
from mappings.WellLogsFormat import WellLogFormat
from utils.cluster_dataframe import cluster_dataframe
from utils.prompt_library import get_prompt_template
import traceback

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

def _cleanup_data(data_section_df, subsection_name, null_values=[-999.25]):
    try:
        """
        Clean up the data by removing None values and converting them to strings.
    
        Args:
            data (dict): The input data dictionary.
    
        Returns:
            dict: A new dictionary with cleaned-up data.
        """
        if data_section_df.empty:
            print(f"The DataFrame is empty for {subsection_name}. Cleaning up will not be performed.")
            return data_section_df  # Return the empty DataFrame as is

        # Convert lists to comma-separated strings if they contain multiple items
        for column in data_section_df.select_dtypes(include=['object']).columns:
            data_section_df[column] = data_section_df[column].apply(
                lambda x: ",".join(map(str, x)) if isinstance(x, list) and len(x) > 1
                else str(x[0]) if isinstance(x, list) and len(x) == 1
                else x
            )

        #remove all the columns which have null values in the dataframe
        data_section_df = data_section_df.dropna(axis=1, how='all')

        #check if any cell has a values from null_valus list and replace it with None
        data_section_df = data_section_df.replace(null_values, None)

        #replace all the nan values with None
        data_section_df = data_section_df.where(pd.notnull(data_section_df), None)

        #remove duplicate rows
        data_section_df = data_section_df.drop_duplicates()


        return data_section_df
    except Exception as e:
        print(f'Error in cleanup data of {subsection_name}: {str(e)}')
        print(traceback.format_exc())
        return pd.DataFrame()

def _merge_dicts_to_dataframe(data_dicts, subsection_name):
    """
    Converts a list of structured dictionaries with 'attributes' and 'objects' keys into a single Pandas DataFrame.

    Args:
        data_dicts (list): A list of dictionaries, each containing 'attributes' and 'objects' keys.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    all_dfs = []  # List to store individual DataFrames

    for data_dict in data_dicts:
        try:
            if subsection_name == 'curves':
                # Handle curves separately
                if not isinstance(data_dict, list):
                    raise ValueError(f"Expected a list for 'curves' subsection, but got {type(data_dict)}.")

                # Convert the curves list into a DataFrame
                df = pd.DataFrame(data_dict)

                # Handle 'axis' column: store as a string (single value or comma-separated)
                if "axis" in df.columns:
                    df["axis"] = df["axis"].apply(
                        lambda x: str(x[0]) if len(x) == 1 else ",".join(map(str, x)) if x else None)

                all_dfs.append(df)  # Append the DataFrame for curves
            else:
                if 'attributes' not in data_dict or 'objects' not in data_dict or data_dict is None:
                    raise ValueError(f"Unable to convert dictionary to data frame for subsection {subsection_name}. Each dictionary must contain 'attributes' and 'objects' keys. {data_dict}")

                attributes = data_dict['attributes']
                objects = data_dict['objects']

                # Convert objects into a DataFrame
                df = pd.DataFrame.from_dict(objects, orient='index', columns=attributes)

                # Move the dictionary key (previous index) to a new column called 'parameter_name'
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'name'}, inplace=True)
                all_dfs.append(df)  # Append the DataFrame to the list
        except Exception as e:
            print(f'Error in merger_dicts_to_data_frame: {str(e)}')
            print(traceback.format_exc())
            #assign an empty dataframe if there is an error
            all_dfs.append(pd.DataFrame())

    # Merge all DataFrames vertically (stacking them)
    merged_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)


    #cleaning up the dataframe right after merging
    merged_df = _cleanup_data(data_section_df=merged_df, subsection_name=subsection_name)

    return merged_df

def json_to_text(result, data):
    try:
        #cleaning up headers for las and dlis data
        result = _cleanup_headers(headers=result)

        #cleaning up data if the file is in dlis format
        if result.get('input_file_format') == WellLogFormat.DLIS.value:

            combined_parameters, combined_equipments, combined_tools, combined_zones, combined_frames, combined_curves  = [], [], [], [], [], []

            for sample in data:
                combined_parameters.append(sample['parameters'])
                combined_equipments.append(sample['equipments'])
                combined_tools.append(sample['tools'])
                combined_zones.append(sample['zones'])
                combined_frames.append(sample['frame'])
                combined_curves.append(sample['curves'])


            parameter_df = _merge_dicts_to_dataframe(data_dicts=combined_parameters, subsection_name='parameters')
            equipment_df = _merge_dicts_to_dataframe(data_dicts=combined_equipments, subsection_name='equipments')
            tool_df = _merge_dicts_to_dataframe(data_dicts=combined_tools, subsection_name='tools')
            zone_df = _merge_dicts_to_dataframe(data_dicts=combined_zones, subsection_name='zones')
            frame_df = _merge_dicts_to_dataframe(data_dicts=combined_frames, subsection_name='frames')
            curves_df = _merge_dicts_to_dataframe(data_dicts=combined_curves, subsection_name='curves')

            parameter_df = cluster_dataframe(df=parameter_df, subsection_name='parameters', cluster_columns=['name','description'])
            equipment_df = cluster_dataframe(df=equipment_df, subsection_name='equipments', cluster_columns=['name'])
            tool_df = cluster_dataframe(df=tool_df, subsection_name='tools', cluster_columns=['name','description'])
            zone_df = cluster_dataframe(df=zone_df, subsection_name='zones', cluster_columns=['name','description'])
            frame_df = cluster_dataframe(df=frame_df, subsection_name='frames', cluster_columns=['name','description'])
            curves_df = cluster_dataframe(df=curves_df, subsection_name='curves', cluster_columns=['name','description'])

            print(parameter_df.to_csv(rf"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\parameter_data_df.csv", index=False))
            print(equipment_df.to_csv(rf"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\equipment_data_df.csv", index=False))
            print(tool_df.to_csv(rf"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\tool_data_df.csv", index=False))
            print(zone_df.to_csv(rf"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\zone_data_df.csv", index=False))
            print(frame_df.to_csv(rf"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\frames_data_df.csv", index=False))
            print(curves_df.to_csv(rf"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed\channels_data_df.csv", index=False))

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
        print(f'Error in json to text conversion: {str(e)}')
        print(traceback.format_exc())