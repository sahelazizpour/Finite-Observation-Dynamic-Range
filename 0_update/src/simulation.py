import numpy as np
import scipy 
from scipy import stats, signal, optimize

#TODO: unit testing?

def coupling_weights(N, K, lambda_, seed):
    """
    Generate coupling weights.

    Parameters
    ----------
    N : int
        Number of neurons.
    K : int
        Avg. number of connections per neuron
    lambda_ : float
        total incomming synaptic weight per neuron (lambda_ = sum_j w_ij)
    seed : int
        Random seed.   

    Returns
    -------
    w : sparse matrix of shape (N,N)
        Connectivity matrix.
    """
    w = scipy.sparse.random(N, N, density=K/N, random_state=seed, format='csr')
    # normalize each row such that sum_j w_ij = lambda
    w = w.multiply(lambda_/w.sum(axis=1))
    w = w.tocsr()
    return w

def external_spiking_probability(N, mu, h_, seed, dt=1):
    """
    Generate external spiking probability vector.

    Parameters
    ----------
    N : int
        Number of neurons.
    mu : float
        Fraction of neurons that receive external input
    h_ : float
        External input rate that selected neurons receive
    seed : int
        Random seed.
    dt : float
        Time step.

    Returns
    -------
    h : array of shape (2,mu*N)
        External spiking probability.
    """
    assert mu<=1 and mu>=0, "mu must be in [0,1]"
    np.random.seed(seed)
    # select mu*N random neurons
    id_neurons = np.random.choice(N, int(mu*N), replace=False)
    p_h = 1 - np.exp(-h_*dt)
    # return 2d array where first column is selected neurons and second column is p_h
    return (id_neurons, p_h*np.ones_like(id_neurons))

def transfer(x):
    """
    Linear transfer function of the neurons.
    """
    x[x<0] = 0
    x[x>1] = 1
    return x

def step(x, w, p_h, rng):
    """
    Simulate one time step of the network.

    Parameters
    ----------
    x : array of shape (N,)
        Current activity of the network.
    w : (sparse) matrix of shape (N,N)
        Connectivity matrix.
    p_h : tuple of structure (ids, probs) both of lenght mu*N 
        External spiking probability.
    rng : numpy.random.RandomState
        Random number generator.

    Returns
    -------
    x : array of shape (N,)
        Updated activity of the network.
    """
    # spike with probability w * x
    p_rec = transfer(w @ x)
    x = np.zeros_like(x)
    # continue only for non-zero entries to save computation
    id_nonzero = np.nonzero(p_rec)[0]
    if np.size(id_nonzero) > 0:
        id_spike = rng.random(np.size(id_nonzero)) < p_rec[id_nonzero]
        x[id_nonzero[id_spike]] = 1
    # or spike with probability h
    id_nonzero = p_h[0]
    if np.size(id_nonzero) > 0:
        id_spike = rng.random(np.size(id_nonzero)) < p_h[1]
        # add spikes to x (if already spiked this will remain 1)
        x[id_nonzero[id_spike]] = 1
    return x