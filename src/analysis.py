import numpy as np
from scipy import stats, signal, optimize

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

def dynamic_range(h_range):
    """
    Calculate the dynamic range from the range h_range
    """
    assert len(h_range) == 2
    h_left, h_right = h_range
    return 10 * (np.log10(h_right) - np.log10(h_left))