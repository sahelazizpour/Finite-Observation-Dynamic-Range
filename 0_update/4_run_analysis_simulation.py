# arguement parser
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--N", type=int, default=int(1e4))
parser.add_argument("--K", type=int, default=100)
parser.add_argument("--mu", type=float, default=0.2)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--epsilon", type=float, default=0.1)
parser.add_argument("--sigma", type=float, default=0.01)
parser.add_argument("--window", type=float, required=True)
parser.add_argument("--database", type=str, required=True)
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize
from tqdm import tqdm
import pandas as pd

import os, sys

rootpath = os.path.join(os.getcwd(), ".")
sys.path.append(rootpath)
from src.utils import *
from src.analysis import *
from src.approximation import *


def __main__(args):
    params = {
        "N": args.N,
        "K": args.K,
        "mu": args.mu,
        "seed": args.seed,
    }
    # analysis
    params_results = params.copy()
    params_results["epsilon"] = args.epsilon
    params_results["sigma"] = args.sigma
    params_results["window"] = args.window

    print("start analysis")
    # IMPORTANT: this is the resolution and range of lambda for the plots
    list_lambda = 1 - 10 ** np.linspace(0, -4, 32 + 1)

    # this runs in parallel
    result = analysis_beta_approximation(params_results, list_lambda, args.database)
    save_analysis_beta_approximation(result, args.path, args.database)

if __name__ == "__main__":
    __main__(args)
