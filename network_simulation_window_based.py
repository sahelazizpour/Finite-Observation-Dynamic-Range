from functools import partial
import functions
import numpy as np
import os
import pickle
import warnings
import sys
import json
warnings.filterwarnings("ignore")

# n_lambda in [0,8]
# n_h in [0,209]
# n_real [0,9]

alpha = float(sys.argv[1])
n_lambda = int(sys.argv[2])
n_h = int(sys.argv[3])
n = int(sys.argv[4])
path_to_save0 = sys.argv[5]

# alpha = 0                                            # fraction of inhibitory neurons
n_realization=10
N=10000                                               # number of neurons
k=20                                                 # in/out connectivity degree (choose multiples of 5 or change the function "draw_connections)
input_type='multiplicative'                          # input can be added multiplicatively or additively
hyperregularity = 1
n_data = 10000                                     # minimum number of mean responses computed at each realization
init_activity=0.01                                   # fraction of initial active neurons
n_stationary=10000                                    # time steps after which activity becomes stationary
window_size=np.array([1e1,1e2,1e3,1e4],dtype=np.int) #window sizes for computing the mean response, in ascending order, factors of 10

if alpha==0:

    lambda_list = 1 - np.logspace(-4, 0, 9)
    logh_list = np.arange(-7, 1.5, 0.05).tolist()  # list of input intensities
    lambda_ = lambda_list[n_lambda]
    path_to_save  =path_to_save0 + "epsilon=%.2e" % (1 - lambda_)
    os.system("mkdir -p %s" % path_to_save)
else:
    lambda_list = [0.1,0.3,0.5,0.6,0.65,0.675,0.7,0.725,0.75,0.7625,0.775,0.7825,0.8,0.8125,0.85,0.9,0.925,0.95,0.975,0.99,0.999,1,1.001,1.01]
    logh_list = np.arange(-7, 1.5, 0.05).tolist()  # list of input intensities
    lambda_ = lambda_list[n_lambda]
    path_to_save =path_to_save0 + "/lambda=%.3e" % (lambda_)
    os.system("mkdir -p %s" % path_to_save)

#save parameters
parameters = {
  "alpha": alpha,"N": N,"k": k,"input_type" : input_type,"init_activity": init_activity,"n_stationary": n_stationary,"window_size": window_size.tolist()
    ,"hyperregularity": hyperregularity
}
temp = json.dumps(parameters)
with open(path_to_save0+'/parameters.txt', 'w') as outfile:
    json.dump(temp, outfile)

#set input
h_list=np.power(10,logh_list)
h=h_list[n_h]

#open adjacency matrix
sparseA=pickle.load(open( path_to_save0 +'/adjacencyMatrices/sparseA_lambda='+str(lambda_)+'_realization='+str(n), 'rb'))

func = partial(functions.do_realization_save_window_average,path_to_save,N,n_stationary,n_data,init_activity,input_type,window_size,sparseA)
func(lambda_,h,n)



