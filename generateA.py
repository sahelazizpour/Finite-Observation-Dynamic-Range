import functions
import pickle
import warnings
import sys
warnings.filterwarnings("ignore")


lambda_ = float(sys.argv[1])
n = int(sys.argv[3])

path_to_save =  'FODR/'
n_realization=10
alpha = 0                                            # fraction of inhibitory neurons
N=10000                                               # number of neurons
k=20                                                 # in/out connectivity degree (choose multiples of 5 or change the function "draw_connections)
gamma = lambda_ / (k * (1 - 2 * alpha))              # connection weight
input_type='multiplicative'                          # input can be added multiplicatively or additively
homogeneity = 1                                      # homogeneity of network graph
hyperregularity=0                                    # hyperregularity of network graph

sparseA=functions.draw_connections(lambda_, k, N, alpha, gamma)
pickle.dump(sparseA, open(path_to_save + '/sparseA_lambda='+str(lambda_)+'_realization='+str(n), 'wb'))

