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
parser.add_argument("--redo", action="store_true")
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

import sqlite3
from src.utils import *
from src.analysis import *
from src.theory import *
#import dask
from dask.distributed import Client, LocalCluster, as_completed

def __main__(args):
    # define parameters
    params = {
        "N": args.N,
        "K": args.K,
        "mu": args.mu,
        "sigma": args.sigma,
        "epsilon": args.epsilon,
        "window": args.window,
        "seed": args.seed,
    }
    # variables
    database = args.database
    path = args.path

    print("start analysis")
    # IMPORTANT: this is the resolution and range of lambda for the plots
    list_lambda = 1 - 10 ** np.linspace(0, -4, 64 + 1)

    ### Uses beta_interpolation based on simulation results from `database` to estimate the number of discriminable intervals and dynamic range for the given parameters `params`
    ## Function parallelizes by splitting up different lamda values into different processes
    con = sqlite3.connect(database)
    cur = con.cursor()

    # check if results already exist in database unless redo is specified; if so, return
    if exists_in_database(con, cur, "results", params) & (not args.redo):
        # warning if results already exist
        print(f"Results already exist in database for params: {params}")
        # close database
        con.close()
        return

    # load function approximation from database
    beta_interpolation = pd.read_sql_query(
        f"SELECT * FROM beta_interpolations WHERE N={params['N']} AND K={params['K']} AND mu={params['mu']} AND seed={params['seed']}",
        con,
    )
    # check that there is a unique function approximation
    if not len(beta_interpolation) == 1:
        raise ValueError(
            f"No unique function approximation in database for params (either 0 or multiple entries)"
        )
    print(
        f"Load beta interpolation from file: {beta_interpolation['filename'].values[0]}"
    )
    beta_approx = FunctionApproximation(
        filename=beta_interpolation["filename"].values[0]
    )
    con.close()

    # parameters of the beta distribution
    loc = beta_approx.params["loc"]
    scale = beta_approx.params["scale"]
    # support for the gaussian noise
    delta = 1 / beta_approx.params["N"]
    x_gauss = support_gauss(5 * params["sigma"], delta)
    # pmf of the gaussian noise (needs delta for normalization)
    pmf_gauss = stats.norm.pdf(x_gauss, 0, params["sigma"]) * delta 

    # full support for convolution (add another delta to the right because beta distribution is calculated from the difference of the cdf)
    x_beta = support_conv_pmf_gauss([0,1+delta], x_gauss)
    x_noise = support_conv_pmf_gauss([0,1], x_gauss)

    def pmf_noise(window, lam, h):
        a,b = beta_approx(lam, window, h)
        # pmf as difference of cdf to ensure that the pmf is normalized
        pmf = np.diff(stats.beta.cdf(x_beta, a, b, loc=loc, scale=scale))
        return np.convolve(pmf, pmf_gauss, mode="same")

    # parallel processing of different lambda values and store results in dataframe
    df = pd.DataFrame(columns=["lambda", "number_discriminable", "dynamic_range"])

    # specify h_range (math the range function is trained on)
    h_range = beta_approx.input_range[beta_approx.input_names.index("h")]
    print(f"Using h_range = {h_range}")

    ## define function for dask
    def analyse(lam, verbose=False):
        """
            return lambda, number of discriminable intervals, dynamic range
        """
        pmf_o_given_h = lambda h: pmf_noise(params['window'], lam, h)

        # get reference distributions from mean-field solution (needs delta for normalization because in domain [0,1] with stepsize delta)
        pmf_refs = [stats.norm.pdf(x_noise, mean_field_activity(lam, params['mu'], h), params['sigma'])*delta for h in [0, np.inf]]

        # get dynamic range and number of discriminable states
        dr, nd = analysis_dr_nd(pmf_o_given_h, h_range, pmf_refs, params["epsilon"], verbose=verbose)
        return lam, dr, nd

    print("execute independent lambda analyses in parallel with dask")
    cluster = LocalCluster()
    dask_client = Client(cluster)
    # map analysis function to list of lambda values and run these futures in parallel
    futures = dask_client.map(analyse, list_lambda)
    data = []
    for future in tqdm(as_completed(futures), total=len(list_lambda)):
        data.append(future.result())

    # sort data by first column
    data = np.array(sorted(data, key=lambda x: x[0]))
    
    print("write results to file and database")
    # save dataframe to ASCI files (tab-separated)
    filename = f"{path}/sigma={params['sigma']}_epsilon={params['epsilon']}/N={params['N']}_K={params['K']}_mu={params['mu']}/results_simulation_seed={params['seed']}_window={params['window']}.txt"
    # create directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data to file
    np.savetxt(
        filename,
        data,
        delimiter="\t",
        header="#lambda\tnumber of discriminable inputs\tdynamic_range",
        comments="",
    )
    params["filename"] = filename

    # store in database
    # con = sqlite3.connect(database)
    # cur = con.cursor()
    # insert_into_database(con, cur, "results", params)
    # con.commit()
    # con.close()

if __name__ == "__main__":
    __main__(args)
