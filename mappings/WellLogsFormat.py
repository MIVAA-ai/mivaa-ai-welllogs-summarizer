from enum import Enum

class WellLogFormat(Enum):
    LAS = "LAS"
    DLIS = "DLIS"
    UNKNOWN = "UNKNOWN"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, WellLogFormat))