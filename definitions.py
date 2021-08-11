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
    "naval_compressor_decay": 0.01,  # maybe try more
    "power": 0.2,
    "protein": 0.2,
    "wine": 0.05,
    "yacht": 0.1,
    "year_prediction_msd": 0.01,
}
