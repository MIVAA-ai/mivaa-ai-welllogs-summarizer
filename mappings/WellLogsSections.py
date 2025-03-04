from enum import Enum

class WellLogsSections(Enum):
    header = "header"
    parameters = "parameters"
    equipments = "equipments"
    zones = "zones"
    tools = "tools"
    frame = "frame"
    curves = "curves"
    data = "data"

    @staticmethod
    def list():
        return list(map(lambda c: c.value, WellLogsSections))