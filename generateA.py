import numpy as np
import random
import time
import scipy as sp
import functions
import pickle
import warnings
import sys
import os
warnings.filterwarnings("ignore")

alpha= float(sys.argv[1])
n_lambda = int(sys.argv[2])
n = int(sys.argv[3])
path_to_save0 = sys.argv[4]


n_realization=10
# alpha = 0                                            # fraction of inhibitory neurons
N=10000                                               # number of neurons
k=20                                                 # in/out connectivity degree (choose multiples of 5 or change the function "draw_connections)
input_type='multiplicative'                          # input can be added multiplicatively or additively
homogeneity = 1                                      # homogeneity of network graph
hyperregularity = 1

if alpha==0:
    lambda_list = 1 - np.logspace(-4, 0, 9)
else:
    lambda_list = [0.1,0.3,0.5,0.6,0.65,0.675,0.7,0.725,0.75,0.7625,0.775,0.7825,0.8,0.8125,0.85,0.9,0.925,0.95,0.975,0.99,0.999,1,1.001,1.01]

lambda_=lambda_list[n_lambda]
gamma = lambda_ / (k * (1 - 2 * alpha))              # connection weight

try:
    os.mkdir(path_to_save0 +'/adjacencyMatrices')
except:
    pass

sparseA=functions.draw_connections(lambda_, k, N, alpha, gamma,n)
pickle.dump(sparseA, open( path_to_save0+'/adjacencyMatrices/sparseA_lambda='+str(lambda_)+'_realization='+str(n), 'wb'))

