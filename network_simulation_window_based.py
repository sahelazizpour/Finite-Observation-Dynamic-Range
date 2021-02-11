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

lambda_ = float(sys.argv[1])
n_h = int(sys.argv[2])
n = int(sys.argv[3])

# lambda_list=1-np.logspace(-4,0,9)
# lambda_=lambda_list[n_lambda]

logh_list = np.arange(-7,-0.5,0.05).tolist()+ np.arange(-0.5,1.5,0.025).tolist()       # list of input intensities
h_list=np.power(10,logh_list)
h=h_list[n_h]

sp.random.seed()
path_to_save =  'FODR/'


n_realization=10
alpha = 0                                            # fraction of inhibitory neurons
N=10000                                               # number of neurons
k=20                                                 # in/out connectivity degree (choose multiples of 5 or change the function "draw_connections)
input_type='multiplicative'                          # input can be added multiplicatively or additively
homogeneity = 1                                      # homogeneity of network graph
hyperregularity=0                                    # hyperregularity of network graph
n_data = 10000                                     # minimum number of mean responses computed at each realization
init_activity=0.01                                   # fraction of initial active neurons
n_stationary=10000                                    # time steps after which activity becomes stationary
window_size=np.array([1e1,1e2,1e3,1e4],dtype=np.int) #window sizes for computing the mean response, in ascending order, factors of 10

#save parameters
pickle.dump([ alpha,N, k, homogeneity, input_type, hyperregularity,n_data, init_activity, n_stationary,
             window_size], open(path_to_save +'/parameters', 'wb'))

func = partial(functions.do_realization_save_window_average,path_to_save, N, k, alpha, n_stationary, n_data,
               init_activity, input_type,window_size)
# t=time.time()
func(lambda_,h,n)
# print(time.time()-t)


