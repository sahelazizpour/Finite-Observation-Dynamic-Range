import numpy as np
import warnings
from scipy import stats
import pandas as pd
warnings.filterwarnings("ignore")

path_to_data= 'data/network_simulation/subpop_k=100/fixed_indegree/'
path_to_save='data/betaParams/subpop_k=100/betaParams_March2023/'

N=10000
eps_list=np.logspace(-4,0,9)
eps_list=eps_list[7:8]
logh_list = np.arange(-7, 1.5, 0.05).tolist()  # list of input intensities

typ='Full' #or 'Random' or 'Fix'
h_list=np.power(10,logh_list)
window_size_orig=[1,10,100,1000,10000]
window_size=[10]#,100,1000,10000]
n_realization=10
for i, epsilon in enumerate(eps_list):
    path = path_to_data+'/epsilon=' + '{:.2e}'.format(epsilon)+'/'
    logh_list = np.log10(h_list)
    print('lambda=' + str(1-epsilon))
    for j,window in enumerate(window_size):
        j = [k for k in range(len(window_size_orig)) if window_size_orig[k] == window][0]
        for n in range(n_realization):
            dat = {'logh': logh_list, 'a': -1, 'b': -1, 'loc': 0, 'scale': 1, 'meanResponse': ''}
            df = pd.DataFrame(data=dat)
            print(typ,'_n_window='+str(j)+'_realization: ',n)
            s=[]
            for g, logh_ in enumerate(logh_list):
                df.loc[g, 'logh'] = logh_
                n_h = [i for i in range(len(logh_list)) if logh_list[i] == logh_][0]

                mean_response = np.load(
                    path + '/sub' + typ + '_window' + str(j) + '_lambda=' + str(
                        1-epsilon) + '_logh=' + str(np.round(logh_, 3)) + '_realization=' + str(n) + '.npz')['mean_response']

                ## because stats.beta.fit does not accept 0 and 1 values
                mean_response[np.where(mean_response == 0)[0]] = 1e-16
                mean_response[np.where(mean_response == 1)[0]] = 1 - 1e-16

                try:
                    param = stats.beta.fit(mean_response, floc=0, fscale=1)
                    df.loc[g, 'a'] = param[0];
                    df.loc[g, 'b'] = param[1]

                except:
                    print('failed to fit Beta for item no:', g)
                    pass
            df.to_pickle(path_to_save+'/epsilon=' + '{:.2e}'.format(eps_list[i]) + '_sub' + typ + '_window=' + str(window) + '_realization=' + str(n) + '.pkl')

