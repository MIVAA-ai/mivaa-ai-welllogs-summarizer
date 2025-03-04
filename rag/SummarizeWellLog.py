from config.config_loader import get_active_llm

class SummarizeWellLog:
    def __init__(self, documents, logger):
        try:
            self._active_llm = get_active_llm(logger=logger)
            self._logger = logger
            self._documents = documents
        except Exception as e:
            self._logger.error(f"Unable to instantiate class due to invalid llm: {str(e)}")
            raise e

    """
    function to rag the headers in the form of text.
    """

    def get_summary_from_llm(self):
        """
        Generates a summary using the LLM based on the subsection, file format, and appropriate input structure.

        Args:
            subsection_name (str): Name of the well log subsection (e.g., "header", "parameters").
            file_format (str): File format, e.g., "DLIS" or "LAS".
            data (dict): The data to be summarized.

        Returns:
            str: The generated summary or an empty string in case of an error.
        """
        try:
            # # Retrieve prompt configuration
            # prompt_config = SummaryPromptConfig.get_prompt_config(file_format, subsection_name)
            #
            # if not prompt_config:
            #     self._logger.warning(
            #         f"No matching prompt found for {subsection_name} in {file_format}. Skipping summary generation.")
            #     return ""
            #
            # prompt_name = prompt_config["prompt"]
            #
            # # Fill in input data dynamically
            # input_data = SummaryPromptConfig.fill_input_data(file_format, subsection_name, data)
            #
            # # Retrieve prompt template
            # summary_prompt_template = get_prompt_template(logger=self._logger, prompt_name=prompt_name)
            #
            # # Log the final input data before invoking the LLM
            # self._logger.info(f"Final input data for {subsection_name} ({file_format}): {input_data}")
            #
            # print(summary_prompt_template)
            # # Create and execute the summarization chain
            # paragraph_chain = summary_prompt_template | self._active_llm | StrOutputParser()
            # response = paragraph_chain.invoke(input_data)
            #
            # # Log the summary
            # self._logger.info(
            #     f"\n===== Generated Summary for {subsection_name} of {file_format} file =====\n{response}")
            #
            # return response
            #
        except Exception as e:
            # self._logger.warning(
            #     f"Error: Unable to generate summary for {subsection_name}. Returning empty string - {str(e)}")
            # return ""
