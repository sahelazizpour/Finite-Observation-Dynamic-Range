from functools import partial
import multiprocessing as mp
import functions
import numpy as np
import os
import pickle
import scipy as sp
import warnings
import time
warnings.filterwarnings("ignore")
sp.random.seed()
# os.system("taskset -p 0xff %d" % os.getpid())

mainpath =  'data/newDR/alpha=0/'
mainpath =  'alpha=0/'

lambda_list=[0.9999]                                 # list of lambda values
h_list = np.logspace(-8,2,400).tolist()              # list of input intensities
# h_list=[1e-1]
alpha = 0                                            # fraction of inhibitory neurons
N=1000                                               # number of neurons
k=20                                                 # in/out connectivity degree (choose multiples of 5 or change the function "draw_connections)
input_type='multiplicative'                          # input can be added multiplicatively or additively
homogeneity = 1                                      # homogeneity of network graph
hyperregularity=1                                    # hyperregularity of network graph
n_data = 500                                          # minimum number of mean responses computed at each realization
n_realization =32                                     # number of realizations with different network topologies
init_activity=0.01                                   # fraction of initial active neurons
n_stationary=3000                                    # time steps after which activity becomes stationary
window_size=np.array([1e1,1e2,1e3,1e4],dtype=np.int) #window sizes for computing the mean response, in ascending order, factors of 10
n_process=32                                          #number of available system processors

t = time.time()
#save parameters
pickle.dump([lambda_list, h_list, alpha,N, k, homogeneity, input_type, hyperregularity,n_data,n_realization, init_activity, n_stationary, window_size], open(mainpath +'/parameters', 'wb'))
#simulate
batch_n=int(n_realization/n_process)
for l,lambda_ in enumerate(lambda_list):              #loop over all lambda values
    print('lambda=', lambda_)
    path0 = mainpath + 'lambda=' + str(lambda_)
    os.mkdir(path0)
    gamma = lambda_ / (k * (1 - 2 * alpha))           #connection weight
    sparseA_list = []
    for i in range(n_realization):                    #compute connectivity matrices for all different topologies
        sparseA_list.append(functions.draw_connections(lambda_,k, N, alpha, gamma, homogeneity, hyperregularity))
    pickle.dump(sparseA_list, open(path0 + '/sparseA_list', 'wb'))
    for n_h,h in enumerate(h_list):                   #loop over all input intensities
        print('estimated remaining time for this lambda: '+str((len(h_list)-n_h)*0.1)+' hours')
        n_h = [i for i in range(len(h_list)) if h_list[i] == h][0]
        print('n_h=',n_h)
        path = path0 + '/n_h=' + str(n_h)
        os.mkdir(path)
        func = partial(functions.do_realization_save_window_average,N,n_stationary,n_data,window_size,init_activity,path,h,input_type,sparseA_list)
        for j in range(len(window_size) ):
            path_temp = path + '/window' + str(j)
            os.mkdir(path_temp)
        for n in range(batch_n):                       #simulate realizations in parallel
            pool = mp.Pool(processes=n_process)
            range_n=np.arange(n*n_process,(n+1)*n_process)
            print(range_n)
            pool.map(func,range_n)
            pool.close()
            pool.join()
print((time.time() - t))