from functools import partial
# import multiprocessing as mp
import functions
import numpy as np
import os
import pickle
import scipy as sp
import warnings
import sys
import time
warnings.filterwarnings("ignore")

n_lambda = int(sys.argv[1])
n_h = int(sys.argv[2])
n_realization = int(sys.argv[3])

sp.random.seed()
path_to_save =  'data/blah/'


lambda_list=[0.9999]                                 # list of lambda values
h_list = np.logspace(-8,2,400).tolist()              # list of input intensities
alpha = 0                                            # fraction of inhibitory neurons
N=10000                                               # number of neurons
k=20                                                 # in/out connectivity degree (choose multiples of 5 or change the function "draw_connections)
input_type='multiplicative'                          # input can be added multiplicatively or additively
homogeneity = 1                                      # homogeneity of network graph
hyperregularity=1                                    # hyperregularity of network graph
n_data = 1                                     # minimum number of mean responses computed at each realization
init_activity=0.01                                   # fraction of initial active neurons
n_stationary=10                                    # time steps after which activity becomes stationary
window_size=np.array([1e1,1e2,1e3,1e4],dtype=np.int) #window sizes for computing the mean response, in ascending order, factors of 10

#save parameters
pickle.dump([lambda_list, h_list, alpha,N, k, homogeneity, input_type, hyperregularity,n_data, init_activity, n_stationary,
             window_size], open(path_to_save +'/parameters', 'wb'))


func = partial(functions.do_realization_save_window_average,path_to_save, N, k, alpha, h_list,lambda_list, n_stationary, n_data,
               init_activity, input_type,window_size)
# t=time.time()
func(n_lambda,n_h,n_realization)
# print(time.time()-t)


