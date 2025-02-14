from scanners.DLISOriginsProcessor import DLISOriginsProcessor
from scanners.DLISParametersProcessor import DLISParametersProcessor
from scanners.DLISToolsProcessor import DLISToolsProcessor
from scanners.DLISChannelsProcessor import DLISChannelsProcessor
from scanners.DLISEquipmentsProcessor import DLISEquipmentsProcessor
from scanners.DLISFramesProcessor import DLISFramesProcessor
from scanners.DLISZonesProcessor import DLISZoneProcessor
from utils.dlis_utils import transform_curves_to_json_well_log_format


class DLISLogicalFile:
    """
    Represents a logical file in a DLIS physical file.
    Extracts, processes, and transforms origin data using pandas DataFrame.
    """

    def __init__(self, logical_file, logger, extract_bulk):
        """
        Initialize the DLISLogicalFile.

        Args:
            logical_file: The DLIS logical file object.
        """
        self._logical_file = logical_file
        self._logical_file_id = logical_file.fileheader.id
        self._logger = logger  # Store the logger
        self._extract_bulk = extract_bulk

    def scan_logical_file(self):
        """
        Scans the logical file, extracts, transforms, and prints origin data as JSON.
        """
        # try:
        # Delegate origin processing to DLISOriginsProcessor
        self._logger.info(f"Extracting origins for {self._logical_file_id}")

        origins_processor = DLISOriginsProcessor(
            logical_file_id=self._logical_file_id,
            origins=self._logical_file.origins,
            logger=self._logger
        )
        header = origins_processor.map_headers()

        self._logger.info(f"Extracting origins for {self._logical_file_id} is successful")

        self._logger.info(f"Extracting parameters for {self._logical_file_id}")

        parameters_processor = DLISParametersProcessor(
            logical_file_id=self._logical_file_id,
            items=self._logical_file.parameters,
            logger=self._logger
        )
        parameters = parameters_processor.extract_parameters()

        self._logger.info(f"Extracting parameters for {self._logical_file_id} is successful")

        self._logger.info(f"Extracting equipments for {self._logical_file_id}")

        equipments_processor = DLISEquipmentsProcessor(
            logical_file_id=self._logical_file_id,
            items=self._logical_file.equipments,
            logger=self._logger
        )
        equipments = equipments_processor.extract_equipments()

        self._logger.info(f"Extracting equipments for {self._logical_file_id} is successful")

        self._logger.info(f"Extracting zones for {self._logical_file_id}")

        zones_processor = DLISZoneProcessor(
            logical_file_id=self._logical_file_id,
            items=self._logical_file.zones,
            logger=self._logger
        )
        zones = zones_processor.extract_zones()

        self._logger.info(f"Extracting zones for {self._logical_file_id} is successful")

        self._logger.info(f"Extracting tools for {self._logical_file_id}")

        tools_processor = DLISToolsProcessor(
            logical_file_id=self._logical_file_id,
            items=self._logical_file.tools,
            logger=self._logger
        )
        tools = tools_processor.extract_tools()

        self._logger.info(f"Extracting tools for {self._logical_file_id} is successful")

        self._logger.info(f"Extracting channels for {self._logical_file_id}")

        # Process frames, channels, and curves
        combined_output = []
        for frame in self._logical_file.frames:
            # Extract frame-level metadata
            frames_processor = DLISFramesProcessor(
                logical_file_id=self._logical_file_id,
                items=[frame],
                logger=self._logger
            )
            frame_data = frames_processor.extract_frames()

            # Extract channels and curves for the current frame
            channels_processor = DLISChannelsProcessor(
                logical_file_id=self._logical_file_id,
                items=frame.channels,
                logger=self._logger
            )
            channels = channels_processor.extract_channels()
            formatted_channels = transform_curves_to_json_well_log_format(channels, logger=self._logger)

            #extract curves only if extract_bulk is True
            if self._extract_bulk:
                curves = channels_processor.extract_bulk_data()
            else:
                curves = []

            # Combine all data into the frame-specific dictionary
            frame_output = {
                "header": header,
                "parameters": parameters,
                "equipments": equipments,
                "zones": zones,
                "tools": tools,
                "frame": frame_data,
                "curves": formatted_channels,
                "data": curves
            }

            combined_output.append(frame_output)

        self._logger.info(f"Extracting channels for {self._logical_file_id} is successful")

        return combined_output