import json
from config.config_exceptions import ConfigLoadError
from mappings.WellLogsSections import WellLogsSections
from config.config_loader import get_active_llm, get_summary_prompt_template, get_summary_prompt_name
import traceback
from langchain.chains.combine_documents import create_stuff_documents_chain

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
    def summarise(self):
        try:
            header_content = json.loads(self._documents[WellLogsSections.header.value][0].page_content)
            file_format = header_content['input_file_format']

            summaries = {}

            for section, docs in self._documents.items():
                if not docs:
                    self._logger.warning(f"No documents found for {section}. Skipping summary generation.")
                    continue

                try:
                    prompt_name = get_summary_prompt_name(self._logger, file_format, section)
                except ConfigLoadError as e:
                    self._logger.warning(str(e))
                    continue

                summary_prompt_template = get_summary_prompt_template(self._logger, prompt_name)

                chain = create_stuff_documents_chain(
                    llm=self._active_llm,
                    prompt=summary_prompt_template
                )

                summary = chain.invoke({"context": docs})
                summaries[section] = summary
                self._logger.info(f"Generated summary for {section}")

            return summaries


        except Exception as e:
            self._logger.warning(f"Error: Unable to generate summary. Returning empty string - {str(e)}")
            self._logger.debug(traceback.format_exc())
            return ""