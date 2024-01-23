import numpy as np
from scipy import stats, signal, optimize
import sqlite3
from src.utils import *
import pandas as pd

def calc_overlap_beta(ab1, ab2, loc, scale):
    """calculate the overlap between two beta distributions?"""
    a1,b1 = ab1
    a2,b2 = ab2
    # calculate overlap defined as 0.5*int(min(f1,f2)) where f1 = beta(a1,b1,loc,scale) and f2 = beta(a2,b2,loc,scale)
    # this is equivalent to 0.5*int(min(beta(a1,b1,loc,scale),beta(a2,b2,loc,scale)))

    # # calculate the intersection of the two beta distributions
    # # this is equivalent to the minimum of the two beta distributions
    # a = np.minimum(a1,a2)
    # b = np.minimum(b1,b2)
    # # calculate the integral of the intersection
    # # this is equivalent to the overlap between the two beta distributions
    # overlap = stats.beta.cdf(1+2*delta, a, b, loc=loc, scale=scale) - stats.beta.cdf(-delta, a, b, loc=loc, scale=scale)
    # return overlap
    return None

def calc_overlap(pmf1, pmf2):
    """
    calculates the overlap between two discrete probability mass functions
    ATTENTION: needs user to ensure that domains are identical!
    """
    assert len(pmf1) == len(pmf2)
    return np.sum(np.minimum(pmf1, pmf2)) * 0.5

def find_discriminable_inputs(pmf, h_range, pmf_refs, epsilon:float, start="left", verbose=False):
    """
    Determine all inputs h in range h_range such that the overlap between all pmfs is less than epsilon
    The pmfs of h_range[0] and h_range[1] sever as boundaries

    # Parameter
    - pmf: function
    - h_range: array-like with length two
    - epsilon: float
        discrimination error that specifies maximal overlap between two probability mass functions
    """
    assert len(h_range) == 2
    h_left, h_right = h_range
    assert len(pmf_refs) == 2
    pmf_ref_left, pmf_ref_right = pmf_refs
    
    # make sure that boundaries of h_range overlap with the reference pmfs
    pmf_left = pmf(h_left)
    overlap_left = calc_overlap(pmf_ref_left, pmf_left)
    assert overlap_left > epsilon
    pmf_right = pmf(h_right)
    overlap_right = calc_overlap(pmf_ref_right, pmf_right)
    assert overlap_right > epsilon
    
    # pmf_ref declares the pmf to which the overlap is calculated; at beginning this deviates from pmf_cur
    # pmf_end declares pmf that constraints the search for h
    if start == "left": 
        pmf_ref = pmf_ref_left
        pmf_end = pmf_ref_right
        h_cur = h_left
        pmf_cur = pmf_left
        overlap_beg = overlap_left
    if start == "right":
        pmf_ref = pmf_ref_right
        pmf_end = pmf_ref_left
        h_cur = h_right
        pmf_cur = pmf_right
        overlap_beg = overlap_right

    overlap_end = calc_overlap(pmf_end, pmf_cur)

    assert overlap_beg > epsilon
    assert overlap_end < epsilon

    hs = []
    # loop until ovelap between pmf_cur and pmf_end is smaller than epsilon
    while overlap_end < epsilon:
        def func(h):
            return calc_overlap(pmf_ref, pmf(h)) - epsilon

        if start == "left":
            h_cur = optimize.bisect(func, h_cur, h_right)
        elif start=="right":
            h_cur = optimize.bisect(func, h_left, h_cur)
        pmf_cur = pmf(h_cur)

        overlap_end = calc_overlap(pmf_end, pmf_cur)
        if verbose:
            print(f"possible solution: h={h_cur} with overlap to end of {overlap_end}", end = " ... ")
        # if overlap with pmf_end is smaller than epsilon, add h_cur to list and take current pmf as new reference
        if overlap_end < epsilon:
            hs.append(h_cur)
            pmf_ref = pmf_cur
            if verbose:
                print("accepted")
        else:
            if verbose:
                print("rejected")
            break
    return hs

def dynamic_range(h_range):
    """
    Calculate the dynamic range from the range h_range
    """
    assert len(h_range) == 2
    h_left, h_right = h_range
    return 10 * (np.log10(h_right) - np.log10(h_left))


import torch
from torch import nn

def analysis(params, database, path_out='./dat/', verbose=True):
    """
        Uses beta_interpolation based on simulation results from `database` to estimate the number of discriminable intervals and dynamic range for the given parameters `params`
        Stores result in `path_out`
        Function parallelizes by splitting up different lamda values into different processes

        Parameters
        ----------
        params : dict
            Dictionary with simulation parameters.
        database : str
            Path to database.
    """
    # get relevant information from database
    import sqlite3
    con = sqlite3.connect(database)
    cur = con.cursor()

    if exists_in_database(con, cur,'results', params):
        raise ValueError(f"Results already in database for params")

    # load function approximation from database
    beta_interpolation = pd.read_sql_query(f"SELECT * FROM beta_interpolations WHERE N={params['N']} AND K={params['K']} AND mu={params['mu']} AND seed={params['seed']}", con)
    # check that there is a unique function approximation
    if not len(beta_interpolation)==1:
        raise ValueError(f"No unique function approximation in database for params (either 0 or multiple entries)")
    if verbose:
        print(f"Load beta interpolation from file: {beta_interpolation['filename'].values[0]}")
    beta_approx = FunctionApproximation(filename=beta_interpolation['filename'].values[0])
    con.close()

    # define pmf from convolution of beta distribution with Gaussian noise
    delta = 1/beta_approx.params['N']
    support = np.arange(0, 1+4*params['sigma'], delta)
    support = np.concatenate((-support[::-1], support[1:]))
    loc = beta_approx.params['loc']
    scale = beta_approx.params['scale']

    def ml_pmf(window, lam, h, verbose=False):
        a,b = beta_approx(lam, window, h)
        # pmf as difference of cdf to ensure that the pmf is normalized
        pmf_beta = np.diff(stats.beta.cdf(support, a, b, loc=loc, scale=scale))
        # convolution with a Gaussian distribution at every point of the support
        pmf_norm = stats.norm.pdf(support, 0, params['sigma'])*delta
        return np.convolve(pmf_beta, pmf_norm, mode="same")

    # parallel processing of different lambda values and store results in dataframe
    df = pd.DataFrame(columns=['lambda', 'number_discriminable', 'dynamic_range'])

    # specify h_range (need to exclude the zero here because of logh fit)
    # TODO: make this part of the params from the function approximation!!!!
    h_range = [10**-6.5,10]

    # TODO: make this parallel
    for i, lam in tqdm(enumerate(lams_data), total=len(lams_data)):
        # distribution is given by Beta-distribution specified by a and b
        def pmf_o_given_h(h):
            return ml_pmf(params['window'], lam, h)

        pmf_ref_left = stats.norm.pdf(support, a_inf(lam, params['mu'], 0), sigma) * delta
        pmf_ref_right = stats.norm.pdf(support, a_inf(lam, params['mu'], 1e3), sigma) * delta
        hs_left = find_discriminable_inputs(
            pmf_o_given_h, h_range, [pmf_ref_left, pmf_ref_right], epsilon
        )
        hs_right = find_discriminable_inputs(
            pmf_o_given_h, h_range, [pmf_ref_left, pmf_ref_right], epsilon, start="right"
        )
        if len(hs_left) > 0 and len(hs_right) > 0:
            df.loc[i] = {
                "lambda": lam,
                "number_discriminable": 0.5 * (len(hs_left) + len(hs_right)),
                "dynamic_range": dynamic_range((hs_left[0], hs_right[0])),
            }
            # drs[i] = dynamic_range((hs_left[0], hs_right[0]))
            # nds[i] = 0.5 * (len(hs_left) + len(hs_right))
        else:
            df.loc[i] = {
                "lambda": lam,
                "number_discriminable": np.nan,
                "dynamic_range": np.nan,
            }
    
    # write results to file and database
    # save dataframe to ASCI files (tab-separated)
    filename = f"{path_out}/N={params['N']}_K={params['K']}_mu={params['mu']}/results_simulation_seed={params['seed']}_window={params['window']}_sigma={params['sigma']}_epsilon={params['epsilon']}.txt"
    # create directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(f'{filename}', index=False, sep='\t')
    params['filename']=filename

    # store in database
    con = sqlite3.connect(database)
    cur = con.cursor()
    insert_into_database(con, cur, 'results', params)
    con.close()

