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