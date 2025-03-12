# header_mapping.py

class HeaderMapping:
    @staticmethod
    def get_default_mapping():
        """
        Returns the default header mapping for LAS to JSON conversion.
        Value of the dict represents field in the las file
        Key of the dict represents the field in jsonwelllogformat schema
        """
        return {
            "well": ["WELL", "API", "WELL-NAME"],
            "field": ["FLD", "FIELD-NAME"],
            "country": ["CTRY"],
            "date": ["DATE", "CREATION-TIME"],  # Will be generated dynamically
            "operator": ["COMP", "COMPANY"],
            "serviceCompany": ["SRVC", "PRODUCER-NAME"],
            "elevation": ["ELEV"],
            "startIndex": ["STRT"],
            "endIndex": ["STOP"],
            "step": ["STEP"],
            "name": [], # Will be read from file
            "description": [],
            "wellbore": [],
            "null": ["NULL", "null"],
            "runNumber": ["RUN-NUMBER"],  # Not standard in LAS
            "source": [],  # Not standard in LAS
            "dataUri": []  # Not standard in LAS
        }

    @staticmethod
    def get_date_fields():
        """
        Returns a list of fields in the header mapping that should be treated as dates.
        """
        return ["date"]  # Add fields that represent dates here