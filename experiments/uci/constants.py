import enum
from abc import ABC
from dataclasses import dataclass


@dataclass
class Dataset(ABC):
    input_column_names: list[str]
    output_column_name: str


class BostonDataset(Dataset):
    input_column_names = [
        "crim",
        "zn",
        "indus",
        "chas",
        "nox",
        "rm",
        "age",
        "dis",
        "rad",
        "tax",
        "ptratio",
        "b",
        "lstat",
    ]
    output_column_name = "medv"


class ConcreteDataset(Dataset):
    input_column_names = [
        "cement",
        "blast_furnace_slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
    ]
    output_column_name = "concrete_compressive_strength"


class EnergyCoolingDataset(Dataset):
    input_column_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    output_column_name = "Y2"


class EnergyHeatingDataset(Dataset):
    input_column_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    output_column_name = "Y1"


class Kin8nmDataset(Dataset):
    input_column_names = [
        "theta1",
        "theta2",
        "theta3",
        "theta4",
        "theta5",
        "theta6",
        "theta7",
        "theta8",
    ]
    output_column_name = "y"


class NavalCompressorDataset(Dataset):
    input_column_names = [
        "Lever position",
        "Ship speed (v)",
        "GTT",
        "GTn",
        "GGn",
        "Ts",
        "Tp",
        "HP",
        "T1",
        "T2",
        "P48",
        "P1",
        "P2",
        "Pexh",
        "TIC",
        "mf",
    ]
    output_column_name = "Compressor DSC"


class NavalTurbineDataset(Dataset):
    input_column_names = [
        "Lever position",
        "Ship speed (v)",
        "GTT",
        "GTn",
        "GGn",
        "Ts",
        "Tp",
        "HP",
        "T1",
        "T2",
        "P48",
        "P1",
        "P2",
        "Pexh",
        "TIC",
        "mf",
    ]
    output_column_name = "Turbine DSC"


class PowerDataset(Dataset):
    input_column_names = ["AT", "V", "AP", "RH"]
    output_column_name = "PE"


class ProteinDataset(Dataset):
    input_column_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"]
    output_column_name = "rmsd"


class WineDataset(Dataset):
    input_column_names = [
        "fixed" "acidity",
        "volatile" "acidity",
        "citric" "acid",
        "residual" "sugar",
        "chlorides",
        "free" "sulfur" "dioxide",
        "total" "sulfur" "dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    output_column_name = "quality"


class YachtDataset(Dataset):
    input_column_names = ["LC", "PC", "L/D", "B/Dr", "L/B", "Fr"]
    output_column_name = "Rr"


class BreastDataset(Dataset):
    # https://www.kaggle.com/datasets/roustekbio/breast-cancer-csv
    input_column_names = [
        "clump_thickness",
        "size_uniformity",
        "shape_uniformity",
        "marginal_adhesion",
        "epithelial_size",
        "bare_nucleoli",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
    ]
    output_column_name = "class"


class CrabDataset(Dataset):
    # https://www.kaggle.com/datasets/thanhtinl/crabex
    input_column_names = [
        "sp",
        "FL",
        "RW",
        "CL",
        "CW",
        "BD",
    ]
    output_column_name = "sex"


class HeartDataset(Dataset):
    # https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    input_column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    output_column_name = "target"


class IonosphereDataset(Dataset):
    # https://www.kaggle.com/datasets/prashant111/ionosphere
    input_column_names = [
        "column_a",
        "column_b",
        "column_c",
        "column_d",
        "column_e",
        "column_f",
        "column_g",
        "column_h",
        "column_i",
        "column_j",
        "column_k",
        "column_l",
        "column_m",
        "column_n",
        "column_o",
        "column_p",
        "column_q",
        "column_r",
        "column_s",
        "column_t",
        "column_u",
        "column_v",
        "column_w",
        "column_x",
        "column_y",
        "column_z",
        "column_aa",
        "column_ab",
        "column_ac",
        "column_ad",
        "column_ae",
        "column_af",
        "column_ag",
        "column_ah",
    ]
    output_column_name = "column_ai"


class DiabetesDataset(Dataset):
    # https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    input_column_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    output_column_name = "Outcome"


class RegressionDatasetSchema(str, enum.Enum):
    # boston = "boston"
    # concrete = "concrete"
    # energy_cooling = "energy_cooling"
    # energy_heating = "energy_heating"
    kin8nm = "kin8nm"
    power = "power"
    # wine = "wine"
    # yacht = "yacht"


class ClassificationDatasetSchema(str, enum.Enum):
    breast = "breast"
    crab = "crab"
    diabetes = "diabetes"
    heart = "heart"
    ionosphere = "ionosphere"


DATASET_SCHEMA_MAPPING = {
    # RegressionDatasetSchema.boston: BostonDataset,
    # RegressionDatasetSchema.concrete: ConcreteDataset,
    # RegressionDatasetSchema.energy_cooling: EnergyCoolingDataset,
    # RegressionDatasetSchema.energy_heating: EnergyHeatingDataset,
    RegressionDatasetSchema.kin8nm: Kin8nmDataset,
    RegressionDatasetSchema.power: PowerDataset,
    # RegressionDatasetSchema.wine: WineDataset,
    # RegressionDatasetSchema.yacht: YachtDataset,
    ClassificationDatasetSchema.breast: BreastDataset,
    ClassificationDatasetSchema.crab: CrabDataset,
    ClassificationDatasetSchema.diabetes: DiabetesDataset,
    ClassificationDatasetSchema.heart: HeartDataset,
    ClassificationDatasetSchema.ionosphere: IonosphereDataset,
}
