import numpy as np
import scipy 
from tqdm import tqdm
import h5py
import os
import sqlite3
from src.utils import *

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
    print(f'number input neurons: {len(id_neurons)}')

    return (id_neurons, p_h*np.ones_like(id_neurons))

def output_mask(N, nu, seed):
    """
    Generate output mask.

    Parameters
    ----------
    N : int
        Number of neurons.
    nu : float
        Fraction of neurons that are connected to the output estimate
    seed : int
        Random seed.

    Returns
    -------
    mask_output : array of shape (N,)
        Mask that specifies from which neurons the output is recorded.
    """
    assert nu<=1 and nu>=0, "nu must be in [0,1]"
    np.random.seed(seed+1) # make sure it is NOT the same seed as for the input neurons!!!
    # boolean mask that specifies from which neurons the output is recorded
    mask = np.zeros(N, dtype=bool)
    id_output = np.random.choice(N, int(nu*N), replace=False)
    mask[id_output] = True
    print(f'number output neurons: {np.sum(mask)}')
    return mask

def transfer(x):
    """
    Linear transfer function of the neurons.
    """
    x[x<0] = 0
    x[x>1] = 1
    return x

def step(x, w, p_h, rng, numba=False):
    """
    Simulate one time step of the network.

    Parameters
    ----------
    x : array of shape (N,)
        Current activity of the network.
    w : sparse matrix of shape (N,N)
        Connectivity matrix.
    p_h : tuple of structure (ids, probs) both of length mu*N 
        External spiking probability.
    rng : numpy.random.RandomState
        Random number generator.

    Returns
    -------
    x : array of shape (N,)
        Updated activity of the network.
    """
    # neurons spike with probability w * x
    p_rec = transfer(w @ x)

    # reset x to zero
    x.fill(0)
    id_rec_nonzero = np.nonzero(p_rec)[0]
    if id_rec_nonzero.size > 0:
        spike = rng.random(size=id_rec_nonzero.size) < p_rec[id_rec_nonzero]
        x[id_rec_nonzero[spike]] = 1
    
    # some neurons spike independently with probability h
    id_h = p_h[0]
    if id_h.size > 0:
        spike = rng.random(size=id_h.size) < p_h[1]
        # set state to 1 (if already 1 it just stays 1)
        x[id_h[spike]] = 1

    return x

def simulation(params, steps={'burn':'self', 'equil':'self', 'record':'self'}, windows=np.array([1e0, 1e1, 1e2, 1e3, 1e4])):
    """
    run simulation and return sliding window estimates

    Parameters
    ----------
    params : dict
        dictionary with parameters that needs to include
        - N : int
            number of neurons
        - K : int
            number of incoming connections per neuron
        - lambda : float
            coupling strength
        - mu : float
            fraction of neurons that receive external input
        - nu : float
            fraction of neurons that are connected to the output estimate
        - h : float
            external input
        - seed : int
            random seed static setup (recurrent weights and external coupling)
        - seed_d : int
            random seed for dynamics
    steps : dict
        dictionary with update steps for different phases of the simulation (default is self-consistently determined with autorcorelation time tau(lambda) and processing window )
        - burn : int
            burn-in steps (steps that are completely discarded at beginning of simulation so that dynamics reach steady state). Self: max(30 * tau, window_max)
        - equil : int
            equilibration steps (steps during which we estimate running averages, but do not save them). Self: 3 * window_max
        - record : int
            recording steps (steps during which we save estimates). Self: max(1000 * tau, 100 * window_max)
    windows : array
        array with windows to use for sliding window estimates

    Returns
    -------
    windows : array
        array with windows used for sliding window estimates
    samples : array
        array with sliding window estimates of length steps['record']
    """
    # timescale of simulation is in ms
    dt = 1 #ms

    print("simulation with parameters:", params)
    print("recording windows:", windows)    

    # create system
    w = coupling_weights(params['N'], params['K'], params['lambda'], params['seed'])
    p_h = external_spiking_probability(params['N'], params['mu'], params['h'], params['seed'])
    mask_output = output_mask(N=params['N'], nu=params['nu'], seed=params['seed'])
    lam = params['lambda']

    # Autocorrelation time of the recurrent network dynamics tau is connected to the largest eigenvalue of the connectivity matrix
    # We ensured that this by normalizing the sum over each column to be lambda, which becomes the largest eigenvalue. 
    # Then $\tau = -dt / \ln(\lambda)$ (which numerically can only be computed for lambda > 0)
    if lam > 1e-2:
        tau = - dt/ np.log(lam)
    else: 
        tau = 0
    rng = np.random.RandomState(params['seed_d'])

    # estimate running average with exponential kernel:
    # at each time point, we want to measure O(t) \propto int ds x(t-s) exp(-s/window)
    # we can get this online by writing O(t) = (1-alpha) O(t-dt) + alpha x(t) with alpha = 1 - exp(-dt/window)
    alphas = 1 - np.exp(-dt / windows)
    estimate = lambda x : np.mean(x[mask_output])
    def update(estimates, x):
        estimates = (1-alphas)*estimates + alphas * estimate(x)
        return estimates
    
    window_max = np.max(windows)
    # get self-consistent times from timescale of network dynamics determined by lambda
    steps_burn = steps['burn']
    if steps_burn == "self":
        steps_burn = int(max(30*tau,window_max))
        print(f'# COMMENT: burn-in steps self-consistently set to {steps_burn:.2e} = max(30 * tau, window_max) with tau = -dt / ln(lambda) = {tau:.2e} and window_max={window_max:.2e}')
    
    steps_equil = steps['equil']
    if steps_equil == "self":
        steps_equil = int(3*window_max)
        print(f'# COMMENT: equilibration steps self-consistently set to {steps_equil:.2e} = 3 * window_max = {window_max:.2e}')
    
    steps_record = steps['record']
    if steps_record == "self":
        # get self-consistent recording time from timecale of network dynamics determined by lambda
        steps_record = int(max(1000*tau, 100*window_max))
        print(f'# COMMENT: recording steps self-consistently set to {steps_record:.2e} = max(1000 * tau, 100*window_max) with tau = -dt / ln(lambda) = {tau:.2e} and window_max={window_max:.2e}')

    # run simulation until stationary and then record mean activity; to speed up equilibration we start from random initial spiking condition
    print(f'initialize with random spiking condition')
    x = rng.randint(0,2,params['N'])
    for t in tqdm(range(steps_burn), desc="burn-in dynamics"):
        x = step(x, w, p_h, rng)

    # start with the estimation that requires equilibration
    estimates = np.ones_like(windows)*estimate(x)
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
    result['samples'] = dict(zip(windows, samples))
    result['params'] = params
    result['steps'] = steps
    return result

def get_filename(path, params):
    if params["nu"] < 1:
        return f'{path}/nu={params["nu"]}/N={params["N"]}_K={params["K"]}/seed={params["seed"]}/1-lambda={1-params["lambda"]:.2e}/simulation_mu={params["mu"]:.2f}_h={params["h"]:.2e}.h5'
    else:    
        return f'{path}/N={params["N"]}_K={params["K"]}/seed={params["seed"]}/1-lambda={1-params["lambda"]:.2e}/simulation_mu={params["mu"]:.2f}_h={params["h"]:.2e}.h5'

def save_simulation(result, path='./dat/', database='./simulations.db', verbose=False):
    """
        Save simulation results to file.

        Parameters
        ----------
        result : dict
            Dictionary with simulation parameters and results.
        path : str, optional
            Path to save file to. Default is '.dat/'.

        Returns
        -------
        filename : str
            Name of the saved file.
    """
    params = result['params']
    filename = get_filename(path, params)
    # create folder if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    def save_dict(f, name, dict):
        for key in dict.keys():
            if verbose:
                print(f'hdf5 saving {name}/{key}')
            f.create_dataset(f'{name}/{key}', data=dict[key])
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('windows', data=result['windows'])
        # save samples (measurements)
        save_dict(f, 'samples', result['samples'])
        # save steps (burn, equil, record)
        save_dict(f, 'steps', result['steps'])
        # save params as attributes
        for key in params.keys():
            f.attrs[key] = params[key]

    # add one entry per window
    con = sqlite3.connect(database)
    cur = con.cursor()
    # create new dictionary where we only keep parameters relevant for database (N,K,lambda,mu,h,seed)
    params_db = {key: params[key] for key in ['N','K','lambda','mu','h','seed']}
    params_db['raw_file'] = filename
    for window in result['windows']:
        if verbose:
            print(f'sqlite3 saving {window}')
        params_db['window'] = window
        params_db['dataset'] = f'samples/{window}'
        insert_into_database(con, cur, 'simulations', params_db)
    # commit the changes
    con.commit()
    con.close()
    
