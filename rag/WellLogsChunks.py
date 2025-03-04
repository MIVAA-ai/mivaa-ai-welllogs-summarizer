import traceback
from mappings.WellLogsSections import WellLogsSections
import pandas as pd
from mappings.WellLogsFormat import WellLogFormat
from utils.cluster_dataframe import cluster_dataframe
import numpy as np
from langchain_core.documents import Document
import json


class WellLogsChunks:
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
            self._logger.warning(f"Unable to clean data for {subsection_name}: {str(e)}")
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
                    if subsection_name == WellLogsSections.curves.value and isinstance(data_dict, list):
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
                    self._logger.warning(f"Unable to merge dictionaries for {subsection_name}: {str(e)}")
                    all_dfs.append(pd.DataFrame())

            merged_df = pd.concat(all_dfs, axis=0).reset_index(drop=True)
            return self._cleanup_data(merged_df, subsection_name)
        except Exception as e:
            self._logger.warning(f"Unable to convert dictionaries to dataframe for {subsection_name}: {str(e)}")
            return pd.DataFrame()

    def _get_complete_section_documents(self, file_format, subsections_df, headers, max_rows_per_sub_df=5):
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
        documents = {}

        for subsection_name, df in subsections_df.items():
            self._logger.info(f"Processing subsection: {subsection_name}, total rows: {len(df)}")

            if df.empty:
                self._logger.warning(f"Skipping {subsection_name} as it is empty.")
                documents[subsection_name] = None
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

                    array_split_dfs = np.array_split(cluster_df.to_numpy(), num_splits)

                    for split_arr in array_split_dfs:
                        split_df = pd.DataFrame(split_arr, columns=cluster_df.columns)  # Convert back to DataFrame
                        sub_dfs.append(split_df)
                else:
                    sub_dfs.append(cluster_df)

            # Generate documents chunk for each sub-DataFrame
            subsection_documents = []
            for idx, sub_df in enumerate(sub_dfs):
                # Remove "cluster" column if it exists
                if "cluster" in sub_df.columns:
                    sub_df = sub_df.drop(columns=["cluster"])

                sub_dict = sub_df.to_dict(orient="records")  # Convert DataFrame to dictionary format

                self._logger.info(
                    f"Generating sub dictionary for {subsection_name} - Sub-section {idx + 1} with {len(sub_df)} rows")

                # #getting the combined subsection summary
                # summary = self._get_summary_from_llm(subsection_name=subsection_name,
                #                                      file_format=file_format,
                #                                      data=sub_dict)
                #creating a document for this sub section
                document_content = json.dumps(sub_dict, indent=4)  # Convert to JSON string for better readability

                # Create a LangChain Document object
                document = Document(
                    page_content=document_content,  # Store structured text
                    metadata=headers
                )

                subsection_documents.append(document)

            # Combine all summaries into a single text output
            documents[subsection_name] = subsection_documents

        return documents

    def get_documents(self):
        """
        Converts JSON well log data into a structured langchain documents object.

        Args:
            result (dict): JSON header information.
            data (list): JSON well log data.
        """
        try:
            result = self._cleanup_headers()

            #methods to call llms for summarisation
            # self._logger.info("Starting the header summarisation")
            # #directly call _get_summary_from_llm to fix this issue
            # self._summarise_headers(headers=result)
            # self._logger.info("Headers summarised successfully")

            if result.get('input_file_format') == WellLogFormat.DLIS.value:

                dlis_subsections = set()
                for entry in self._json_data:
                    if entry:  # Only process non-empty dictionaries
                        dlis_subsections.update(entry.keys())
                        dlis_subsections.discard(WellLogsSections.data.value)
                        dlis_subsections.discard(WellLogsSections.header.value)

                # dlis_subsections = ["parameters", "equipments", "tools", "zones", "frame", "curves"]
                combined_data = {sub: [] for sub in dlis_subsections}

                for sample in self._json_data:
                    for subsection_name in dlis_subsections:
                        combined_data[subsection_name].append(sample.get(subsection_name, []))

                dlis_subsection_dfs = {sub: self._merge_dicts_to_dataframe(combined_data[sub], sub) for sub in dlis_subsections}

                # Perform clustering on relevant sections
                cluster_columns = {
                    WellLogsSections.parameters.value: ["name", "description"],
                    WellLogsSections.equipments.value: ["name"],
                    WellLogsSections.tools.value: ["name", "description"],
                    WellLogsSections.zones.value: ["name", "description"],
                    WellLogsSections.frame.value: ["name", "description"],
                    WellLogsSections.curves.value: ["name", "description"]
                }

                for sub, df in dlis_subsection_dfs.items():
                    dlis_subsection_dfs[sub] = cluster_dataframe(logger=self._logger, df=df, subsection_name=sub, cluster_columns=cluster_columns[sub])

                documents = self._get_complete_section_documents(file_format=result.get('input_file_format'),
                                                     subsections_df=dlis_subsection_dfs,
                                                     headers=result)
                documents[WellLogsSections.header.value] = [Document(page_content=json.dumps(result, indent=4))]

                return documents
                # # Save CSV outputs
                # for sub, df in dlis_subsection_dfs.items():
                #     file_path = f"F:\PyCharmProjects\mivaa-ai-welllogs-summarizer\processed/{sub}_data.csv"
                #     df.to_csv(file_path, index=False)
                #     self._logger.info(f"Saved processed {sub} data to {file_path}")
        except Exception as e:
            self._logger.error(f"Error in JSON to text conversion: {str(e)}")
            self._logger.debug(traceback.format_exc())
            raise e