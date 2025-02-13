"""
    to convert dlis into json well log format. we will assume the following

    The hierarchy of DLIS file is as follows
    Physical file
    |_Logical file1
        |_Frame1
            |_Channel1(Curve1)
            |_Channel2(Curve2)
        |_Frame2
    |_Logical file2

    A JSON Well Log file consists of one or more log sets each containing a log header, curve definitions and the corresponding measurement data.
    it means that one JSON Well Log file with represent one Logical File.
"""
from scanners.DLISLogicalFile import DLISLogicalFile
class DLISScanner:
    """
       Scans a DLIS physical file and processes its logical files.
    """

    def __init__(self, file_path, logical_file, logger):
        self._file_path = file_path
        self._logical_file = logical_file
        self._logger = logger

    def scan(self):
        """
            Load and process all logical files in the DLIS physical file.
        """
        self._logger.info(f"Starting scan for logical file {self._logical_file.fileheader.id}")

        logical_file_object = DLISLogicalFile(logical_file=self._logical_file, logger=self._logger)
        return logical_file_object.scan_logical_file()