from datetime import datetime, timezone
from dateutil.parser import parse, ParserError

class DateUtils:
    @staticmethod
    def to_iso8601(date_input):
        """
        Convert a datetime, date, or string to ISO 8601 format.
        Handles ambiguous cases like "11/91" as MM/YY format.
        Args:
            date_input (str, datetime, date): The date to convert.
        Returns:
            str: The ISO 8601 formatted string.
        """
        try:
            if isinstance(date_input, datetime):
                # Ensure it's timezone-aware in UTC
                if date_input.tzinfo is None:
                    date_input = date_input.replace(tzinfo=timezone.utc)
                return date_input.isoformat()
            elif hasattr(date_input, "isoformat"):  # For date objects
                return f"{date_input.isoformat()}T00:00:00Z"
            elif isinstance(date_input, str):
                # Try parsing the string
                try:
                    # Try parsing the string
                    parsed_date = parse(date_input, default=datetime(1900, 1, 1))
                    return DateUtils.to_iso8601(parsed_date)
                except ParserError:
                    # Handle ambiguous cases like "MM/YY"
                    if "/" in date_input:
                        parts = date_input.split("/")
                        if len(parts) == 2:  # Assume MM/YY
                            month, year = int(parts[0]), int(parts[1])
                            if year < 100:  # Convert YY to YYYY (assuming 20th century)
                                year += 1900
                            parsed_date = datetime(year, month, 1)
                            return DateUtils.to_iso8601(parsed_date)
                    raise  # Re-raise if parsing still fails
            else:
                raise TypeError("Input must be a string, datetime, or date object.")
        except Exception as e:
            return f"Error: {e}"  # Return the error message for debugging