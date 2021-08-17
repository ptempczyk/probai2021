DATA_DIR = "../dt8122-2021/datasets/"

DATASETS = """boston_housing
concrete
energy_heating_load
kin8nm
naval_compressor_decay
power
protein
wine
yacht
year_prediction_msd""".split()

LR_SWAG = {
    "boston_housing": 0.01,
    "concrete": 0.2,
    "energy_heating_load": 0.1,
    "kin8nm": 0.1,
    "naval_compressor_decay": 0.01,
    "power": 0.2,
    "protein": 0.2,
    "wine": 0.05,
    "yacht": 0.1,
    "year_prediction_msd": 0.01,
}

TOLERANCE = {
    "boston_housing": 0.06,
    "concrete": 0.05,
    "energy_heating_load": 0.05,
    "kin8nm": 0.04,
    "naval_compressor_decay": 0.04,
    "power": 0.05,
    "protein": 0.05,
    "wine": 0.02,
    "yacht": 0.03,
    "year_prediction_msd": 0.05,
}

device = "cuda"
