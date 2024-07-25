import enum
from abc import ABC
from dataclasses import dataclass


@dataclass
class Dataset(ABC):
    input_column_names: list[str]
    output_column_name: str


class BostonDataset(Dataset):
    # https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
    # https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd
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


class WineQualityDataset(Dataset):
    input_column_names = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
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
    # https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
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


class CaffeineDataset(Dataset):
    # https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified
    input_column_names = [
        "age",
        "gender",
        "education",
        "country",
        "ethnicity",
        "nscore",
        "escore",
        "oscore",
        "ascore",
        "cscore",
        "impulsive",
        "ss",
    ]
    output_column_name = "caffeine"


class ChocolateDataset(Dataset):
    # https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified
    input_column_names = [
        "age",
        "gender",
        "education",
        "country",
        "ethnicity",
        "nscore",
        "escore",
        "oscore",
        "ascore",
        "cscore",
        "impulsive",
        "ss",
    ]
    output_column_name = "chocolate"


class CrabDataset(Dataset):
    # https://www.stats.ox.ac.uk/pub/PRNN/crabs.dat
    # https://datarepository.wolframcloud.com/resources/Sample-Data-Crab-Measures
    input_column_names = [
        "sp",
        "FL",
        "RW",
        "CL",
        "CW",
        "BD",
    ]
    output_column_name = "sex"


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


class HeartDataset(Dataset):
    # https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    # https://archive.ics.uci.edu/dataset/45/heart+disease
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
    # https://archive.ics.uci.edu/dataset/52/ionosphere
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


class MushroomsDataset(Dataset):
    # https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified
    input_column_names = [
        "age",
        "gender",
        "education",
        "country",
        "ethnicity",
        "nscore",
        "escore",
        "oscore",
        "ascore",
        "cscore",
        "impulsive",
        "ss",
    ]
    output_column_name = "mushrooms"


class RiceDataset(Dataset):
    # https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    input_column_names = [
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Eccentricity",
        "Convex_Area",
        "Extent",
    ]
    output_column_name = "Class"


class SpamDataset(Dataset):
    # https://www.kaggle.com/datasets/colormap/spambase
    input_column_names = [
        "word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]
    output_column_name = "spam"


class WineColourDataset(Dataset):
    input_column_names = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
    output_column_name = "colour"


class YeastDataset(Dataset):
    # https://archive.ics.uci.edu/dataset/110/yeast
    # Only using CYT and NUC classes for binary classification
    input_column_names = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]
    output_column_name = "class"


class RegressionDatasetSchema(str, enum.Enum):
    # boston = "boston"
    concrete = "concrete"
    energy_cooling = "energy_cooling"
    energy_heating = "energy_heating"
    kin8nm = "kin8nm"
    # naval_compressor = "naval_compressor"
    # naval_turbine = "naval_turbine"
    # power = "power"
    # protein = "protein"
    wine_quality = "wine_quality"
    # yacht = "yacht"


class ClassificationDatasetSchema(str, enum.Enum):
    breast = "breast"
    # caffeine = "caffeine"
    # chocolate = "chocolate"
    # crab = "crab"
    diabetes = "diabetes"
    heart = "heart"
    ionosphere = "ionosphere"
    mushrooms = "mushrooms"
    rice = "rice"
    spam = "spam"
    wine_colour = "wine_colour"
    yeast = "yeast"


DATASET_SCHEMA_MAPPING = {
    # RegressionDatasetSchema.boston: BostonDataset,
    RegressionDatasetSchema.concrete: ConcreteDataset,
    RegressionDatasetSchema.energy_cooling: EnergyCoolingDataset,
    RegressionDatasetSchema.energy_heating: EnergyHeatingDataset,
    RegressionDatasetSchema.kin8nm: Kin8nmDataset,
    # RegressionDatasetSchema.naval_compressor: NavalCompressorDataset,
    # RegressionDatasetSchema.naval_turbine: NavalTurbineDataset,
    # RegressionDatasetSchema.power: PowerDataset,
    # RegressionDatasetSchema.protein: ProteinDataset,
    RegressionDatasetSchema.wine_quality: WineQualityDataset,
    # RegressionDatasetSchema.yacht: YachtDataset,
    ClassificationDatasetSchema.breast: BreastDataset,
    # ClassificationDatasetSchema.caffeine: CaffeineDataset,
    # ClassificationDatasetSchema.chocolate: ChocolateDataset,
    # ClassificationDatasetSchema.crab: CrabDataset,
    ClassificationDatasetSchema.diabetes: DiabetesDataset,
    ClassificationDatasetSchema.heart: HeartDataset,
    ClassificationDatasetSchema.ionosphere: IonosphereDataset,
    ClassificationDatasetSchema.mushrooms: MushroomsDataset,
    ClassificationDatasetSchema.rice: RiceDataset,
    ClassificationDatasetSchema.spam: SpamDataset,
    ClassificationDatasetSchema.wine_colour: WineColourDataset,
    ClassificationDatasetSchema.yeast: YeastDataset,
}
