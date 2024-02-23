from functools import partial
import numpy as np
import FODR_utils
import FODR_utils_cleaning
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import multiprocessing as mp


plot = 1
only_FODR = 1
window_list = [1, 10, 100, 1000, 10000]
window_list = [1]  # ,10,100,1000,10000]
n_realization = 1
# sub_typ='subFix'
sub_typ = 'subFull'
# sub_typ='subRandom'
# input_output='inSub1_outSub2'
input_output = 'inSub1_outAll'
# input_output='inSub1_outSub3'
path_data = 'betaParams_Oct2021/'

N = 10000
error = 0.2
sigma_base = 1e-2
eps_list = np.logspace(-4, 0, 9)[:1]
# eps_list=np.logspace(-5,0,25)
l_list = 1 - eps_list

path = '6.3/'
#os.mkdir(path)
#os.mkdir(path +'/'+sub_typ)

if 0:

    typ_computation = 'simu'
    path_to_save = path + '/' + sub_typ + '/' + typ_computation + '/'
    #os.mkdir(path +'/'+sub_typ+'/'+typ_computation)
    for window in window_list:
        print('window=', window)
        sigma_noise = sigma_base
        funct = partial(FODR_utils.compute_FODR, typ_computation, only_FODR, sub_typ, n_realization, input_output, path_data,
                        path_to_save, eps_list, sigma_noise, window, error)
        pool = mp.Pool(processes=len(eps_list))
        results = pool.map(funct, np.arange(len(eps_list)))
        pool.close()
        pool.join()

#        with ProcessPoolExecutor() as executor:
#            outputs = executor.map(funct, np.arange(len(eps_list)))
if 1:
    window = 1
    typ_computation = 'theo'
    path_to_save = path + '/' + sub_typ + '/' + typ_computation + '/'
    #os.mkdir(path + '/' + sub_typ + '/' + typ_computation)
    sigma_noise = sigma_base
    funct = partial(FODR_utils_cleaning.compute_FODR, typ_computation, only_FODR, sub_typ, n_realization, input_output, path_data,
                    path_to_save, eps_list, sigma_noise, window, error)
    pool = mp.Pool(processes=len(eps_list))
    results = pool.map(funct, np.arange(len(eps_list)))
    pool.close()
    pool.join()

if 0:
    eps_list = np.logspace(-5, 0, 30)
    l_list = 1 - eps_list
    typ_computation = 'inf'
    path_to_save = path + '/' + sub_typ + '/' + typ_computation + '/'
    os.mkdir(path + '/' + sub_typ + '/' + typ_computation)
    sigma_noise = sigma_base
    funct = partial(FODR_utils.compute_FODR, typ_computation, only_FODR, sub_typ, n_realization, input_output, path_data,
                    path_to_save, eps_list, sigma_noise, window, error)
    pool = mp.Pool(processes=len(eps_list))
    results = pool.map(funct, np.arange(len(eps_list)))
    pool.close()
    pool.join()