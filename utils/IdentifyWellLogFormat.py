from mappings.WellLogsFormat import WellLogFormat

class IdentifyWellLogFormat:
    """
    File path will only take the path from the fixed url
    """
    @classmethod
    def GetFormat(cls, filepath):
        try:
            with open(filepath, 'rb') as file:
                header = file.read(256)  # Read the first 256 bytes to check for format

                # Check if it's a LAS file (starts with ~VERSION or ~V)
                if header[:7].decode(errors='ignore').startswith('~VERSION') or header[:2].decode(
                        errors='ignore').startswith('~V'):
                    return WellLogFormat.LAS

                # Check if it's a DLIS file (binary format with specific structure)
                # DLIS files start with '00' (first byte) and follow with a specific sequence
                # However, there is no fixed magic number, so we check for non-ASCII content
                if b'\x00' in header:
                    return WellLogFormat.DLIS

            return WellLogFormat.UNKNOWN
        except Exception as e:
            return f"Error identifying file format: {e}"