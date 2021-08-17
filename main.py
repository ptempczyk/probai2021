import argparse
import numpy as np
import torch

from noisy_adam import run_noisy_ADAM
from swag import run_SWAG
from swag_mod import run_MCSWAG

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, help="name of the dataset file (change path to dataset dir in 'definitions.py' if needed)"
)
parser.add_argument(
    "--method",
    type=int,
    help="""
                    0 - New method (MCSWAG)
                    1 - Matrix Gaussian Posteriors (not implemented)
                    2 - Multiplicative NF (not implemeted)
                    3 - SWAG
                    4 - Noisy Adam""",
)
parser.add_argument(
    "--seed", type=int, help="set random seed (leave default value to reproduce results from report)", default=556
)

args = parser.parse_args()
dataset_path = args.dataset
dataset_name = dataset_path.split("/")[-1].split(".")[0]

if args.method == 0:
    method = run_MCSWAG
elif args.method == 1:
    raise NotImplementedError("Selected method not implemented.")
elif args.method == 2:
    raise NotImplementedError("Selected method not implemented.")
elif args.method == 3:
    method = run_SWAG
elif args.method == 4:
    method = run_noisy_ADAM
else:
    raise ValueError("Method number have to be between 0-4.")

print(f"Dataset: {dataset_name}")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
method(dataset_name)
