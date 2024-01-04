import enum


class DatasetSchema(str, enum.Enum):
    boston = "boston"
    concrete = "concrete"
    energy_cooling = "energy_cooling"
    energy_heating = "energy_heating"
    wine = "wine"
    yacht = "yacht"
