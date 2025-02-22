import json
import os
import traceback
from venv import logger

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from mappings.WellLogsFormat import WellLogFormat
from utils.cluster_dataframe import cluster_dataframe
from config.config_loader import get_active_llm, get_prompt_template
from pathlib import Path

class InterpretWellLog:
    def __init__(self, logger, result, json_data):
        """
        Initializes the Well Log Processor.

        Args:
            logger: Logging instance to capture errors.
            result: Progress of the processing job.
            json_data: scanned well log file converted to json dict
        """
        self._logger = logger
        self._result = result
        self._json_data = json_data

        try:
            self._active_llm = get_active_llm(logger=logger)
        except Exception as e:
            self._logger.error(f"Unable to instantiate class due to invalid llm: {str(e)}")
            raise e

    def _cleanup_headers(self):
        """
        Cleans up headers by removing None values, unnecessary fields, and renaming "name" to "logical_file_name".

        Returns:
            dict: Cleaned-up headers.
        """
        if not isinstance(self._result, dict):
            raise Exception("Headers are not in dictionary format")

        # Remove None values
        headers = {k: v for k, v in self._result.items() if v is not None}

        # Additional cleanup for DLIS format
        if headers.get('input_file_format') == WellLogFormat.DLIS.value:
            keys_to_remove = {
                "status",
                "message",
                "FILE-ID",
                "input_file_creation_user",
                "input_file_creation_date",
            }

            # Dynamically remove keys that start with "output"
            keys_to_remove.update({key for key in headers if key.startswith("output")})

            # Remove specified keys
            for key in keys_to_remove:
                headers.pop(key, None)

        # Rename "name" to "logical_file_name" if it exists
        if "name" in headers:
            headers["logical_file_name"] = headers.pop("name")

        return headers

    def _cleanup_data(self, data_section_df, subsection_name, null_values=[-999.25]):
        """
        Cleans up the data by removing None values and unnecessary columns.

        Args:
            data_section_df (pd.DataFrame): DataFrame to be cleaned.
            subsection_name (str): Name of the subsection.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            if data_section_df.empty:
                self._logger.warning(f"The DataFrame is empty for {subsection_name}. Skipping cleanup.")
                return data_section_df

            # Convert lists to comma-separated strings
            for column in data_section_df.select_dtypes(include=['object']).columns:
                data_section_df[column] = data_section_df[column].apply(
                    lambda x: ",".join(map(str, x)) if isinstance(x, list) and len(x) > 1
                    else str(x[0]) if isinstance(x, list) and len(x) == 1
                    else x
                )

            data_section_df = data_section_df.dropna(axis=1, how='all')  # Remove empty columns
            data_section_df = data_section_df.replace(null_values, None)  # Replace null values
            data_section_df = data_section_df.where(pd.notnull(data_section_df), None)  # Replace NaN with None
            data_section_df = data_section_df.drop_duplicates()  # Remove duplicate rows

            return data_section_df
        except Exception as e:
            self._logger.warning(f"Error cleaning data for {subsection_name}: {str(e)}")
            return pd.DataFrame()

    def _merge_dicts_to_dataframe(self, data_dicts, subsection_name):
        """
        Merges structured dictionaries into a single Pandas DataFrame.

        Args:
            data_dicts (list): List of dictionaries containing 'attributes' and 'objects' keys.
            subsection_name (str): Name of the subsection.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        try:
            all_dfs = []

            for data_dict in data_dicts:
                try:
                    if subsection_name == 'curves' and isinstance(data_dict, list):
                        df = pd.DataFrame(data_dict)
                        if "axis" in df.columns:
                            df["axis"] = df["axis"].apply(
                                lambda x: str(x[0]) if len(x) == 1 else ",".join(map(str, x)) if x else None
                            )
                        all_dfs.append(df)
                    else:
                        attributes = data_dict.get('attributes', [])
                        objects = data_dict.get('objects', {})

                        if not attributes or not objects:
                            self._logger.warning(f"Skipping empty subsection {subsection_name}")
                            continue

                        df = pd.DataFrame.from_dict(objects, orient='index', columns=attributes)
                        df.reset_index(inplace=True)
                        df.rename(columns={'index': 'name'}, inplace=True)
                        all_dfs.append(df)
                except Exception as e:
                    self._logger.warning(f"Error merging dictionaries for {subsection_name}: {str(e)}")
                    all_dfs.append(pd.DataFrame())

            merged_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
            return self._cleanup_data(merged_df, subsection_name)
        except Exception as e:
            self._logger.warning(f"Error converting dictionaries to dataframe for {subsection_name}: {str(e)}")
            return pd.DataFrame()

    """
    function to summarise the headers in the form of text.
    """
    def _get_summary_from_llm(self, subsection_name, file_format, data):
        try:
            if file_format == WellLogFormat.DLIS.value and subsection_name == 'header':
                summary_prompt_template = get_prompt_template(logger=logger, prompt_name='dlis_header_paragraph_summary')

                paragraph_chain = summary_prompt_template | self._active_llm | StrOutputParser()

                response = paragraph_chain.invoke({
                    "user_input": data
                })
                self._logger.info(f"\n===== Generated Summary for {subsection_name} of {file_format} file=====\n{response}")
            elif file_format == WellLogFormat.DLIS.value and subsection_name == 'header':
                pass

        except Exception as e:
            self._logger.warning(f'Error: Unable to generate summary for the header. Returning empty string - {str(e)}')
            return ""

    def _get_complete_section_summary(self, file_format, subsections_df, max_rows_per_sub_df=5):
        """
        Processes each DataFrame subsection by creating smaller sub-DataFrames if necessary,
        then generates summaries using LLM.

        Args:
            file_format (str): The format of the well log file (e.g., "DLIS").
            subsections_df (dict): Dictionary containing subsections as Pandas DataFrames.
            max_rows_per_sub_df (int): Maximum number of rows allowed per sub-DataFrame before further splitting.

        Returns:
            dict: Summaries for each subsection.
        """
        summaries = {}

        for subsection_name, df in subsections_df.items():
            self._logger.info(f"Processing subsection: {subsection_name}, total rows: {len(df)}")

            if df.empty:
                self._logger.warning(f"Skipping {subsection_name} as it is empty.")
                summaries[subsection_name] = ""
                continue

            # Get unique clusters
            unique_clusters = df["cluster"].unique()
            self._logger.info(f"Unique clusters in {subsection_name}: {unique_clusters}")

            sub_dfs = []

            for cluster in unique_clusters:
                cluster_df = df[df["cluster"] == cluster]

                # If the cluster size exceeds the limit, further split into smaller DataFrames
                if len(cluster_df) > max_rows_per_sub_df:
                    num_splits = -(-len(cluster_df) // max_rows_per_sub_df)  # Equivalent to ceil(len/limit)
                    self._logger.info(f"Splitting {subsection_name} - Cluster {cluster} into {num_splits} parts")

                    for split_df in np.array_split(cluster_df, num_splits):
                        sub_dfs.append(split_df)
                else:
                    sub_dfs.append(cluster_df)

            # Generate summary for each sub-DataFrame
            subsection_summaries = []
            for idx, sub_df in enumerate(sub_dfs):
                sub_dict = sub_df.to_dict(orient="records")  # Convert DataFrame to dictionary format
                self._logger.info(
                    f"Generating summary for {subsection_name} - Sub-section {idx + 1} with {len(sub_df)} rows")

                summary = self._get_summary_from_llm(subsection_name=subsection_name, file_format=file_format,
                                                     data=sub_dict)
                subsection_summaries.append(summary)

            # Combine all summaries into a single text output
            summaries[subsection_name] = "\n\n".join(subsection_summaries)

        return summaries

    def json_to_text(self):
        """
        Converts JSON well log data into a structured textual summary.

        Args:
            result (dict): JSON header information.
            data (list): JSON well log data.
        """
        try:
            result = self._cleanup_headers()

            if result.get('input_file_format') == WellLogFormat.DLIS.value:

                dlis_subsections = set()
                for entry in self._json_data:
                    if entry:  # Only process non-empty dictionaries
                        dlis_subsections.update(entry.keys())
                        dlis_subsections.discard("data")
                        dlis_subsections.discard("header")

                # dlis_subsections = ["parameters", "equipments", "tools", "zones", "frame", "curves"]
                combined_data = {sub: [] for sub in dlis_subsections}

                for sample in self._json_data:
                    for subsection_name in dlis_subsections:
                        combined_data[subsection_name].append(sample.get(subsection_name, []))

                dlis_subsection_dfs = {sub: self._merge_dicts_to_dataframe(combined_data[sub], sub) for sub in dlis_subsections}

                # Perform clustering on relevant sections
                cluster_columns = {
                    "parameters": ["name", "description"],
                    "equipments": ["name"],
                    "tools": ["name", "description"],
                    "zones": ["name", "description"],
                    "frame": ["name", "description"],
                    "curves": ["name", "description"]
                }

                for sub, df in dlis_subsection_dfs.items():
                    dlis_subsection_dfs[sub] = cluster_dataframe(logger=self._logger, df=df, subsection_name=sub, cluster_columns=cluster_columns[sub])

                # # Save CSV outputs
                # for sub, df in dlis_subsection_dfs.items():
                #     file_path = f"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed/{sub}_data.csv"
                #     df.to_csv(file_path, index=False)
                #     self._logger.info(f"Saved processed {sub} data to {file_path}")


            #methods to call llms for summarisation
            self._logger.info("Starting the header summarisation")
            self._summarise_headers(headers=result)
            self._logger.info("Headers summarised successfully")

            #methods to call llms for other section of well log file


        except Exception as e:
            self._logger.error(f"Error in JSON to text conversion: {str(e)}")
            self._logger.debug(traceback.format_exc())
            raise e