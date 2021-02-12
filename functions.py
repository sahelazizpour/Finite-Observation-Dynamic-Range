import numpy as np
import networkx as nx
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import eigs
import time
import scipy as sp


def draw_connections(lambda_,k, N, alpha, gamma, homogeneity=1, hyp=1):
    sp.random.seed()
    b = int(alpha * N)
    epsilon = gamma  # eps determines the half-width of the uniform dist of connectivity weights
    ## Hyper regular graph
    if hyp == 1:
        kk = int(k / 5)
        NN = int(N / 5)
        for i in range(5):
            for j in range(5):
                temp = nx.random_regular_graph(kk, NN)
                A0 = nx.to_scipy_sparse_matrix(temp)  # , dtype=np.int8)
                if j == 0:
                    tempA = A0
                else:
                    tempA = bmat([[A0, tempA]])
            if i == 0:
                tempAA = tempA
            else:
                tempAA = bmat([[tempA], [tempAA]])
        A = tempAA.tolil()

        if homogeneity == 1:
            A = gamma * A
        else:
            ind = A.nonzero()
            s = len(ind[0])
            values = np.random.uniform(gamma - epsilon, gamma + epsilon, size=s)
            A[ind] = values

        if alpha != 0:
            ind = A.nonzero()
            a=ind[1]>=(N-b)
            A[ind[0][a],ind[1][a]]=-A[ind[0][a],ind[1][a]]

    else:
        A = coo_matrix((N, N))
        A = A.tolil()
        if homogeneity == 1:
            for i in range(N):
                A[i,np.random.randint(N, size=int(k))]=gamma
        else:
            for i in range(N):
                A[i,np.random.randint(N, size=int(k))]=np.random.uniform(gamma - epsilon, gamma + epsilon, size=int(k))

        if alpha != 0:
            c = np.random.permutation(N)
            ind = c[0:b]
            A[:, ind] = -A[:, ind]
        evals_large, evecs_large = eigs(A, 1, which='LR')
        fr = lambda_ / evals_large[0].real
        A = A * fr
    return A.tocsr()


def transfer(x):
    # piecewise linear function
    x[x<0]=0
    x[x>1]=1
    return x


def do_one_timestep(input_type,N,sparseA,s,p_input):
    '''
    :param s: 1xN array of binary values [0,1] indicating whether neuron is active or not
    '''
    if input_type == 'multiplicative':
        internal_input = sparseA @ s
        p = np.asarray( transfer(internal_input))
        ind_nonzero = np.nonzero(p)[0]
        s=np.zeros(N)
        if np.size(ind_nonzero) > 0 :  # for speeding up the code
            ind_spike = (np.random.uniform(0, 1, np.size(ind_nonzero)) < p[ind_nonzero])
            s[ind_nonzero[ind_spike]] = 1
        s = np.logical_or((np.random.uniform(0, 1, N) < p_input), s)     #neuron spikes either because of internal input or external

    elif input_type == 'additive':
        internal_input = sparseA @ s
        value = internal_input + np.ones(N) * p_input
        p = transfer(value)
        ind_nonzero = np.nonzero(p)[0]
        s = np.zeros(N)
        if np.size(ind_nonzero) > 0:  # for speeding up the code
            ind_spike = (np.random.uniform(0, 1, np.size(ind_nonzero)) < p[ind_nonzero])
            s[ind_nonzero[ind_spike]] = 1
    else:
        print('input type error')
    return s


def do_realization_save_window_average(path,N,n_stationary,n_data,init_activity,input_type,window_size,sparseA,lambda_,h,n):
    '''
    :param n_stationary:
    :param n_data: number of minimum response mean samples to compute
    :param window_size: the observation window sizes to average network responses over
    :param path0: path to save data
    :param h: input intensity
    :param n: indicating which of the graph morphologies is used
    '''
    print('lambda=' + str(lambda_) + ', h=' + str(h) + ',n_realization=', n)

    sp.random.seed()
    n_max=1000000           # maximum number of response mean samples to save
    chunk_size=1000000      # produced trajectory in each round of the loop
    tau=10

    p_input = 1 - np.exp(-h)  # probability of external Poissonian input
    # pickle.dump(sparseA, open(path + '/sparseA_realization='+str(n), 'wb'))
    #generate initial activity
    s = np.zeros(N)         #activity vector
    temp = np.int(N * init_activity)
    index = np.random.permutation(N)
    ind = index[0:temp]
    s[ind] = 1

    mean_response=[[]]
    for i in range(len(window_size)-1):
        mean_response.append([])

    # run simulation until it reaches steady state
    for i in range(n_stationary):
        s=do_one_timestep(input_type,N,sparseA,s,p_input)

    while len(mean_response[-1])<n_data:
        s_avg_temp = []
        # print('n_data=',len(mean_response[-1]))
        # t = time.time()
        # produce trajectories of length chunck_size
        for step in range(chunk_size):
            s = do_one_timestep(input_type, N, sparseA, s, p_input)
            s_avg_temp.append(np.sum(s) * (1 / N))

        #average over window sizes
        for i in range(len(window_size)):
            if len(mean_response[i]) < n_max:
                if window_size[i] <= tau:
                    groups = [s_avg_temp[x:x + window_size[i]] for x in np.arange(0, int(len(s_avg_temp) / tau)) * tau]
                else:
                    groups = [s_avg_temp[x:x + window_size[i]] for x in np.arange(0, int(len(s_avg_temp) / window_size[i])) * window_size[i]]
                mean_response[i] = np.append(mean_response[i], [sum(group) / len(group) for group in groups])
        # print((time.time() - t)/60)

    for i in range(len(window_size)):
        np.savez_compressed(path + '/window'+str(i)+'_lambda='+str(lambda_)+'_logh='+str(np.round(np.log10(h),3))+'_realization=' + str(n)
                            , logh=np.log10(h), mean_response=mean_response[i])


def do_n_realizations_save_window_average(path,N,n_realization,n_stationary,n_data,init_activity,input_type,window_size,sparseA,lambda_,h):
    '''
    :param n_stationary:
    :param n_data: number of minimum response mean samples to compute
    :param window_size: the observation window sizes to average network responses over
    :param path0: path to save data
    :param h: input intensity
    '''


    print('lambda=' + str(lambda_) + ', h=' + str(h) )
    sp.random.seed()
    n_max=1000000           # maximum number of response mean samples to save
    chunk_size=1000000      # produced trajectory in each round of the loop
    tau=1000               # ~ auto correlation time
    p_input = 1 - np.exp(-h)        #probability of external Poissonian input

    for n in range(n_realization):
        mean_response=[[]]
        for i in range(len(window_size)-1):
            mean_response.append([])

        s = np.zeros(N)  # activity vector
        # generate initial activity
        g = np.int(N * init_activity)
        c = np.random.permutation(N)
        ind = c[0:g]
        s[ind] = 1
        # run simulation until it reaches steady state
        for i in range(n_stationary):
            s=do_one_timestep(input_type,N,sparseA,s,p_input)
        t = time.time()
        while len(mean_response[-1])<n_data:
            s_avg_temp = []
            print('n_data=',len(mean_response[-1]))
            # produce trajectories of length chunck_size
            for step in range(chunk_size):
                s = do_one_timestep(input_type, N, sparseA, s, p_input)
                s_avg_temp.append(np.sum(s) * (1 / N))

            #average over window sizes
            for i in range(len(window_size)):
                if len(mean_response[i]) < n_max:
                    # if window_size[i]<tau:
                    #     groups = [s_avg_temp[x:x + window_size[i]] for x in np.arange(0,int(len(s_avg_temp) / tau))*tau]
                    # else:
                    groups = [s_avg_temp[x:x + window_size[i]] for x in np.arange(0, int(len(s_avg_temp) / window_size[i])) * window_size[i]]
                    mean_response[i] = np.append(mean_response[i], [sum(group) / len(group) for group in groups])
        print(time.time() - t)

        for i in range(len(window_size)):
            np.savez_compressed(path + '/window'+str(i)+'_lambda='+str(lambda_)+'_logh='+str(np.round(np.log10(h),3))+'_realization=' + str(n), logh=np.log10(h), mean_response=mean_response[i])


def do_one_realization(path,N,k,alpha,n_timestep,init_activity,input_type,lambda_,h,n):
    '''
    :param n_stationary:
    :param n_data: number of minimum response mean samples to compute
    :param window_size: the observation window sizes to average network responses over
    :param path0: path to save data
    :param h: input intensity
    :param n: indicating which of the graph morphologies is used
    '''
    print('lambda=' + str(lambda_) + ', h=' + str(np.round(h,3)) + ',n_realization=', n)
    sp.random.seed()
    gamma = lambda_ / (k * (1 - 2 * alpha))     #connection weight
    p_input = 1 - np.exp(-h)  # probability of external Poissonian input
    sparseA=draw_connections(lambda_, k, N, alpha, gamma)
    #generate initial activity
    s = np.zeros(N)         #activity vector
    temp = np.int(N * init_activity)
    index = np.random.permutation(N)
    ind = index[0:temp]
    s[ind] = 1
    # run simulation
    s_avg=[]
    for i in range(n_timestep):
        s=do_one_timestep(input_type,N,sparseA,s,p_input)
        s_avg.append(np.sum(s) * (1 / N))

    np.savez_compressed(path + '/' + str(n), S=s_avg)
    # pickle.dump([s_avg],open(path + '/' + str(n), 'wb'))

def do_n_realization(path,N,k,alpha,n_timestep,init_activity,input_type,n_realization,h_list,lambda_,n_h):
    '''
    :param n_stationary:
    :param n_data: number of minimum response mean samples to compute
    :param window_size: the observation window sizes to average network responses over
    :param path0: path to save data
    :param h: input intensity
    :param n: indicating which of the graph morphologies is used
    '''

    sp.random.seed()
    gamma = lambda_ / (k * (1 - 2 * alpha))     #connection weight
    h=h_list[n_h]
    p_input = 1 - np.exp(-h)  # probability of external Poissonian input
    for n in range(n_realization):
        print('lambda=' + str(lambda_) + ', n_h=' + str(n_h) + ',n_realization=', n)
        sparseA=draw_connections(lambda_, k, N, alpha, gamma)
        #generate initial activity
        s = np.zeros(N)         #activity vector
        temp = np.int(N * init_activity)
        index = np.random.permutation(N)
        ind = index[0:temp]
        s[ind] = 1
        # run simulation
        s_avg=[]
        for i in range(n_timestep):
            s=do_one_timestep(input_type,N,sparseA,s,p_input)
            s_avg.append(np.sum(s) * (1 / N))

        np.savez_compressed(path + '/n_h='+str(n_h) +'_n='+ str(n), S=s_avg)
