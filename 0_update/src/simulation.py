import numpy as np
import scipy 
from scipy import stats, signal, optimize
from tqdm import tqdm

# does not seem to speed up things so may not be worth it
from numba import jit
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

@jit(nopython=True)
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

def simulation(params, steps={'burn':'self', 'equil':'self', 'record':'self'}, windows=np.array([1e-6,1e0, 1e1, 1e2, 1e3, 1e4])):
    """
    run simulation and return sliding window estimates

    Parameters
    ----------
    params : dict
        dictionary with parameters that needs to inlcude
        - N : int
            number of neurons
        - K : int
            number of incoming connections per neuron
        - lambda : float
            coupling strength
        - mu : float
            fraction of neurons that receive external input
        - h : float
            external input
        - seed : int
            random seed
        - dt : float    
            time step
    steps : dict
        dictionary with steps to run (default is self-consistently determined)
    windows : array
        array with windows to use for sliding window estimates

    Returns
    -------
    windows : array
        array with windows used for sliding window estimates
    samples : array
        array with sliding window estimates of length steps['record']
    """
    # create system
    w = coupling_weights(params['N'], params['K'], params['lambda'], params['seed'])
    p_h = external_spiking_probability(params['N'], params['mu'], params['h'], params['seed'])
    tau = - params['dt'] / np.log(params['lambda'])
    rng = np.random.RandomState(params['seed'])

    # current estimate with exponential smoothing
    alphas = 1 - np.exp(-params['dt'] / windows)
    print(alphas)
    def update(estimates, x):
        estimates = (1-alphas)*estimates + alphas * np.mean(x)
        return estimates
    
    window_max = np.max(windows)
    # get self-consistent times from timecale of network dynamics determined by lambda
    steps_burn = steps['burn']
    if steps_burn == "self":
        steps_burn = int(10*tau)
        print(f'# COMMENT: burn-in steps self-consistently set to {steps_burn:.2e} = 50 * tau with tau = -dt / ln(lambda) = {tau:.2e}')
    
    steps_equil = steps['equil']
    if steps_equil == "self":
        steps_equil = int(window_max)
        print(f'# COMMENT: equilibration steps self-consistently set to {steps_equil:.2e} = window_max = {window_max:.2e}')
    
    steps_record = steps['record']
    if steps_record == "self":
        # get self-consistent recording time from timecale of network dynamics determined by lambda
        steps_record = int(max(1000*tau, 10*window_max))
        print(f'# COMMENT: recording steps self-consistently set to {steps_record:.2e} = max(1000 * tau, 10*window_max) with tau = -dt / ln(lambda) = {tau:.2e} and window_max={window_max:.2e}')

    # run simulation until stationary and then record mean activity; to speed up equilibration we start from random initial spiking condition
    print(f'initialize with random spiking condition')
    x = rng.randint(0,2,params['N'])
    for t in tqdm(range(steps_burn), desc="burn-in dynamics"):
        x = step(x, w, p_h, rng)

    # start with the estimation that requires equilibration
    estimates = np.ones_like(windows)*np.mean(x)  
    for t in tqdm(range(steps_equil), desc="equilibration for running estimates"):
        x = step(x, w, p_h, rng)
        estimates = update(estimates, x)
    
    # record sliding window estimates
    samples = np.zeros((len(windows), steps_record))
    for t in tqdm(range(steps_record), desc="recording"):
        x = step(x, w, p_h, rng)
        estimates = update(estimates, x)
        samples[:,t] = estimates

    # save result as a dictionary where windows are the keys and samples are the values
    result = dict()
    result['windows'] = windows
    for (window, sample) in zip(windows, samples):
        result[f'samples/{window}'] = sample
    return result

def save_simulation(params, result, path='./dat/'):
    """
        Save simulation results to file.

        Parameters
        ----------
        params : dict
            Dictionary with simulation parameters.
        windows : array_like
            Array with window sizes.
        samples : array_like
            Array with samples.
        path : str, optional
            Path to save file to. Default is '.dat/'.

        Returns
        -------
        filename : str
            Name of the saved file.
    """
    filename = f'{path}/simulation_N={params["N"]}_K={params["K"]}_lambda={params["lambda"]:.2f}_mu={params["mu"]:.2f}_h={params["h"]:.2e}_seed={params["seed"]}.h5'
    
    #create h5 file from dictionary
    with h5py.File(filename, 'w') as f:
        for key in result.keys():
            f.create_dataset(key, data=result[key])

    return filename
