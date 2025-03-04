from mappings.WellLogsFormat import WellLogFormat
from mappings.WellLogsSections import WellLogsSections

class SummaryPromptConfig:
    """
    Stores prompt templates and expected input data mappings for different file formats and subsections.
    """

    FORMAT_PROMPT_MAP = {
        WellLogFormat.DLIS.value: {
            WellLogsSections.header.value: {
                "prompt": "dlis_header_paragraph_summary",
                "input_data": {"user_input": None}
            },
            WellLogsSections.parameters.value: {
                "prompt": "dlis_sections_paragraph_summary",
                "input_data": {"user_input": None,
                               "section": WellLogsSections.parameters.value}
            },
            WellLogsSections.equipments.value: {
                "prompt": "dlis_sections_paragraph_summary",
                "input_data": {"user_input": None,
                               "section": WellLogsSections.equipments.value}
            },
            WellLogsSections.zones.value: {
                "prompt": "dlis_sections_paragraph_summary",
                "input_data": {"user_input": None,
                               "section": WellLogsSections.zones.value}
            },
            WellLogsSections.tools.value: {
                "prompt": "dlis_sections_paragraph_summary",
                "input_data": {"user_input": None,
                               "section": WellLogsSections.tools.value}
            },
            WellLogsSections.frame.value: {
                "prompt": "dlis_sections_paragraph_summary",
                "input_data": {"user_input": None,
                               "section": WellLogsSections.frame.value}
            },
            WellLogsSections.curves.value: {
                "prompt": "dlis_sections_paragraph_summary",
                "input_data": {"user_input": None,
                               "section": WellLogsSections.curves.value}
            },
        },
        WellLogFormat.LAS.value: {
            WellLogsSections.header.value: {
                "prompt": "las_header_paragraph_summary",
                "input_data": {"header_info": lambda d: d.get("header", "N/A")}
            },
            WellLogsSections.parameters.value: {
                "prompt": "las_sections_paragraph_summary",
                "input_data": {"las_data": lambda d: d.get("parameters", "N/A")}
            },
            WellLogsSections.equipments.value: {
                "prompt": "las_equipments_summary",
                "input_data": {"equipment_list": lambda d: d.get("equipments", "N/A")}
            },
            WellLogsSections.zones.value: {
                "prompt": "las_zones_summary",
                "input_data": {"zone_details": lambda d: d.get("zones", "N/A")}
            },
            WellLogsSections.tools.value: {
                "prompt": "las_tools_summary",
                "input_data": {"tool_details": lambda d: d.get("tools", "N/A")}
            },
            WellLogsSections.frame.value: {
                "prompt": "las_frames_summary",
                "input_data": {"frame_info": lambda d: d.get("frame", "N/A")}
            },
            WellLogsSections.curves.value: {
                "prompt": "las_curves_summary",
                "input_data": {"curve_info": lambda d: d.get("curves", "N/A")}
            },
        }
    }

    @classmethod
    def get_prompt_config(cls, file_format, subsection_name):
        """
        Retrieves the prompt configuration for a given file format and subsection.

        Args:
            file_format (str): The format of the well log file (e.g., "DLIS", "LAS").
            subsection_name (str): The subsection name (e.g., "header", "parameters").

        Returns:
            dict: Contains "prompt" (template name) and "input_data" (expected input keys and placeholders).
        """
        return cls.FORMAT_PROMPT_MAP.get(file_format, {}).get(subsection_name, None)

    @classmethod
    def fill_input_data(cls, file_format, subsection_name, data):
        """
        Populates the input_data dictionary with actual values from 'data'.

        Args:
            file_format (str): The format of the well log file (e.g., "DLIS", "LAS").
            subsection_name (str): The subsection name (e.g., "header", "parameters").
            data (dict): The input well log data.

        Returns:
            dict: Populated input_data with actual values from 'data'.
        """
        config = cls.get_prompt_config(file_format, subsection_name)
        if not config:
            return {}

        input_data = config["input_data"].copy()  # Copy to avoid modifying original dictionary

        # Replace `None` placeholders with actual values from `data`
        for key, value in input_data.items():
            if value is None and key == "user_input":
                input_data[key] = data  # Default to "N/A" if missing

        return input_data
