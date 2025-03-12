import dlisio

class DLISLoader:
    """
    A static class to load and cache DLIS logical files, allowing access without reloading.
    """
    _logical_files = None  # Store logical files globally
    _dlis_file_path = None  # Store the DLIS file path to avoid redundant loading

    @staticmethod
    def load_dlis_file(dlis_file_path):
        """
        Loads the DLIS file and caches its logical files.

        Args:
            dlis_file_path (str): Path to the DLIS file.

        Returns:
            list: List of LogicalFile objects.
        """
        if DLISLoader._logical_files is None or DLISLoader._dlis_file_path != dlis_file_path:
            try:
                DLISLoader._logical_files, *_ = dlisio.dlis.load(dlis_file_path)
                DLISLoader._dlis_file_path = dlis_file_path
                print(f"DLIS file loaded: {dlis_file_path}")
            except Exception as e:
                print(f"Error loading DLIS file {dlis_file_path}: {e}")
                DLISLoader._logical_files = None

        return DLISLoader._logical_files

    @staticmethod
    def get_logical_file(logical_file_id):
        """
        Retrieves a LogicalFile object using its logical file ID.

        Args:
            logical_file_id (str): Logical File ID.

        Returns:
            dlisio.dlis.LogicalFile or None: The corresponding LogicalFile object if found.
        """
        if DLISLoader._logical_files is None:
            print("No DLIS file loaded. Call load_dlis_file() first.")
            return None

        for logical_file in DLISLoader._logical_files:
            if str(logical_file.fileheader.id) == logical_file_id:
                return logical_file

        print(f"Logical file ID {logical_file_id} not found.")
        return None