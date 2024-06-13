import numpy as np
from scipy import optimize
from src.theory import *

def analysis_dr_nd(pmf_o_given_h, h_range, pmf_refs, epsilon, verbose=False, return_h=False):    
    """
        Function to consistently analyze the dynamic range and number of discriminable inputs for different approaches (analytic, numerical)

        # Parameters
        - pmf_o_given_h: probability mass function of the output given the input
        - h_range: range of possible h values (needs to match approach, e.g., numerical has limited h_range due to neural network approximation range)
        - pmf_refs: reference probability mass functions that define the boundaries of the dynamic range
        - epsilon: discrimination error that specifies maximal overlap between two probability mass functions
    """
    assert len(h_range) == 2
    assert len(pmf_refs) == 2

    hs_left = find_discriminable_inputs(pmf_o_given_h, h_range, pmf_refs, epsilon, start="left", verbose=verbose)
    hs_right = find_discriminable_inputs(pmf_o_given_h, h_range, pmf_refs, epsilon, start="right", verbose=verbose)

    # return dynamic range and number of discriminable inputs
    if len(hs_left) > 0 and len(hs_right) > 0:
        dr = dynamic_range((hs_left[0], hs_right[0]))
        nd = 0.5 * (len(hs_left) + len(hs_right))
    else:
        dr = np.nan
        nd = np.nan

    if return_h:
        return dr, nd, hs_left, hs_right
    else:
        return dr, nd
    

def support_gauss(bound, delta):
    """
    Create support for Gaussian distribution with standard deviation std and resolution delta
    """
    support = np.arange(0.0, bound + delta, delta)
    support = np.concatenate((-support[::-1][:-1], support))
    return support

def support_conv_pmf_gauss(xlim, support_gauss):
    assert xlim[0] < xlim[-1]

    # shift support_gauss by left boundary of xlim
    support_1 = support_gauss + xlim[0]
    # extract delta from support_gauss (by definition symmetric around 0) - this solves floating point precision issues that ca occur when defining delta = support_gauss[1] - support_gauss[0]
    delta = support_gauss[len(support_gauss)//2+1]
    # continue support until right boundary of xlim
    support_2 = np.arange(support_1[-1], xlim[-1] + support_gauss[-1] + delta, delta)

    return np.concatenate((support_1[:-1], support_2))

def h_range_0(lam, params, verbose=False):
    """
    Determine appropriate h_range for the analysis_0 function based on system parameters and $\lambda$
    """
    # determine h range self-consistently from mean-field solution
    # for low h, assume a population that receives mu*h!
    # a = 1 - (1-lambda*a)(1-p_ext) st. (1-p_ext) = exp(-mu*h) = (1-a)/(1-lambda*a)
    a_min = 0.1*params["sigma"] * params["mu"]
    #h_left = -np.log((1 - a_min) / (1 - lam * params["mu"] * a_min))
    h_left = -np.log((1 - a_min) / (1 - lam * a_min))/params["mu"]
    # for high h, we can assume a_in = a such that a_in = a = 1-(1-lambda*a)(1-p_ext) and (1-p_ext) = exp(-h) = (1-a)/(a-lambda*a)
    a_max = 1 - 0.1*params["sigma"]
    h_right = -np.log((1 - a_max) / (1 - lam * a_max))
    h_range = (h_left, h_right)
    if verbose:
        print(f"lambda: {lam}, h_range: {h_range}")
    return h_range

def dynamic_range(h_range):
    """
    Calculate the dynamic range from the range h_range
    """
    assert len(h_range) == 2
    h_left, h_right = h_range
    return 10 * (np.log10(h_right) - np.log10(h_left))

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
            if func(h_cur) * func(h_right) > 0:
                if verbose:
                    print(f"no further solution possible in range [{h_cur},{h_right}]")
                break
            h_cur = optimize.bisect(func, h_cur, h_right)
        elif start=="right":
            if func(h_cur) * func(h_left) > 0:
                if verbose:
                    print(f"no further solution possible in range [{h_left},{h_cur}]")
                break
            h_cur = optimize.bisect(func, h_left, h_cur)
        pmf_cur = pmf(h_cur)

        overlap_end = calc_overlap(pmf_cur, pmf_end)
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

def calc_overlap(pmf1, pmf2):
    """
    calculates the overlap between two discrete probability mass functions
    ATTENTION: needs user to ensure that domains are identical!
    """
    assert len(pmf1) == len(pmf2)
    return np.sum(np.minimum(pmf1, pmf2)) * 0.5