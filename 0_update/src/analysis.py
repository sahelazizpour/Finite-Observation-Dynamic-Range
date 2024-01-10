import numpy as np
from scipy import stats, signal, optimize

def fit_beta_distribution(data, delta, seed=1234):
    for i in range(10):
        try:
            np.random.seed(seed)
            # add very small jitter to avoid problems with delta distributions
            data = data + np.random.randn(len(data))*delta/7
            # IMPORTANT: The beta distribution is defined on the interval [0,1] but because of the discretization we need to shift the support by delta and scale it by 1+2*delta
            a,b,loc,scale = stats.beta.fit(data, floc=-delta, fscale=1+2*delta) 
            return a,b,loc,scale
        except:
            seed+=1


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
            print("possible solution: ",h_cur, overlap_end, end=" ... ")
        # if new overlap is smaller than epsilon, add h_cur to list and take current pmf as new reference
        if overlap_end < epsilon:
            hs.append(h_cur)
            pmf_ref = pmf_cur
            print("accepted")
        else:
            print("rejected")
    return hs


def dynamic_range(h_range):
    """
    Calculate the dynamic range from the range h_range
    """
    assert len(h_range) == 2
    h_left, h_right = h_range
    return 10 * (np.log10(h_right) - np.log10(h_left))