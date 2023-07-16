import matplotlib.pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
from scipy import signal
from scipy import stats
import FODR_utils
warnings.filterwarnings("ignore")

## This script computes mutual information for conditions [simu, theo, inf] with noise

##Todo integrate this into compute_measures script

alpha=0;N=10000;N_record= N; mu=0.2
eps_list=np.logspace(-4,0,9)
l_list=1-eps_list
typ='Full';input_output='inSub1_outAll'
logh_list = np.arange(-7, 1.5, 0.05).tolist()[:]#[:164]  # list of input intensities
h_list=np.power(10,logh_list); p_list=1-np.exp(-h_list)
window_size0=[1,10,100,1000,10000]
window_size=[1,10,100,1000,10000]
sigma_noise=0.01; sigma_noise_adapted=sigma_noise*N_record
n_realization=10

bin_size=1; bins_edge=np.arange(0,N_record+2,bin_size)
n_bins=len(bins_edge)-1

path_to_save = 'results/MI_withNoise_noWall/sub'+typ+'/'
path_to_data= 'data/network_simulation/subpop_k=100/fixed_indegree/'

#from simulation data
MI_simu=np.zeros(len(eps_list))
for j, window in enumerate(window_size):
    j = [k for k in range(len(window_size0)) if window_size0[k] == window][0]
    for realization_count in range(n_realization):
        print(typ, '_n_window=' + str(j) + '_realization: ', realization_count)
        for l, lambda_ in enumerate(l_list):
            path = path_to_data+'/epsilon=' + '{:.2e}'.format(eps_list[l]) + '/'
            p_joint = np.zeros((len(logh_list), n_bins))
            p_joint_with_noise = np.zeros((len(logh_list), n_bins))
            for g, logh_ in enumerate(logh_list):
                n_h = [i for i in range(len(logh_list)) if logh_list[i] == logh_][0]
                mean_response = np.load(
                    path + '/sub' + typ + '_window' + str(j) + '_lambda=' + str(
                        lambda_) + '_logh=' + str(np.round(logh_, 3)) + '_realization=' + str(realization_count) + '.npz')[
                    'mean_response']
                n_active=(N_record*mean_response).astype(int)
                p_joint[n_h,:]=np.histogram(n_active, bins_edge)[0]
                k_signal = 4;k_noise = 4;
                mean=int(np.mean(mean_response)*N_record)
                sigma=np.sqrt(np.var(mean_response)*N_record*N_record)
                bound_left = mean - int(max(k_signal * sigma, k_noise * sigma_noise_adapted))
                bound_right = mean + int(max(k_signal * sigma, k_noise * sigma_noise_adapted))
                x=np.arange(bound_left, bound_right,bin_size)
                temp = np.zeros(len(x))
                temp[np.logical_and(x >= 0, x <= N_record)] = p_joint[n_h,x[np.logical_and(x >=0, x <= N_record)]]
                pdf_original = temp / (sum(temp) * bin_size)
                gaussian_noise = stats.norm.pdf(x, mean, sigma_noise_adapted)
                temp_conv = signal.convolve(pdf_original, gaussian_noise, mode='same')
                prob = temp_conv / sum(temp_conv)
                p_at_N_record = sum(prob[np.where(x >= N_record)[0]])
                p_at_zero = sum(prob[np.where(x <= 0)[0]])
                pdf_final = prob / bin_size  # so that the integral is 1
                p_joint_with_noise[n_h, x[np.logical_and(x >=0, x <= N_record)]]=pdf_final[np.logical_and(x >=0, x <= N_record)]
                p_joint_with_noise[n_h,0]=p_at_zero
                p_joint_with_noise[n_h,-1] = p_at_N_record

            p_joint_with_noise = p_joint_with_noise / np.sum(p_joint_with_noise)
            p_joint=p_joint/np.sum(p_joint)
            p_x=np.ones(len(logh_list))*(1/len(logh_list))
            y=bins_edge[:-1]+0.5
            MI_simu[l]=FODR_utils.entropy(p_x)-FODR_utils.entropy_x_given_y(p_joint_with_noise,y)
        with open(path_to_save + '/simu/MI_window=' + str(window)+'_realization='+str(realization_count)+ '_sigma_noise=' + str(sigma_noise)+'.pickle','wb') as f:
            pickle.dump(MI_simu, f)


eps_list=np.logspace(-4,0,18)
l_list=1-eps_list
#theoretical T=1ms
MI_theo=np.zeros(len(eps_list))
for l, lambda_ in enumerate(l_list):
    print('lambda=' + str(lambda_))
    p_joint = np.zeros((len(logh_list), n_bins))
    p_joint_with_noise = np.zeros_like(p_joint)
    expr = FODR_utils2023.analytical_expr(1-lambda_)
    for g, logh_ in enumerate(logh_list):
        n_h = [i for i in range(len(logh_list)) if logh_list[i] == logh_][0]
        x, pdf_discrete, mean = FODR_utils.generate_analyic_pdf_for_MI(input_output,expr,N,mu,lambda_, logh_)
        mean=int(mean)
        p_joint[g, :] = pdf_discrete
        k_signal = 5; k_noise = 5;
        max_p=max(pdf_discrete)
        ind1=0
        ind2=N_record
        try:
            ind1=[i for i in range(len(pdf_discrete)-1) if np.logical_and(pdf_discrete[i+1]>max_p/2,pdf_discrete[i]<=max_p/2)][0]
        except:
            pass
        try:
            ind2 = [i for i in range(len(pdf_discrete) - 1) if np.logical_and(pdf_discrete[i + 1] <= max_p / 2, pdf_discrete[i] > max_p / 2)][0]
        except:
            pass
        sigma=ind2-ind1
        bound_left = mean - int(max(k_signal * sigma, k_noise * sigma_noise_adapted))
        bound_right = mean + int(max(k_signal * sigma, k_noise * sigma_noise_adapted))
        x = np.arange(bound_left, bound_right)
        temp = np.zeros(len(x))
        temp[np.logical_and(x >= 0, x <= N_record)] = p_joint[n_h, x[np.logical_and(x >= 0, x <= N_record)]]
        pdf_original = temp / (sum(temp) * (x[1] - x[0]))
        gaussian_noise = stats.norm.pdf(x, mean, sigma_noise_adapted)
        plt.plot(x, gaussian_noise)
        temp_conv = signal.convolve(pdf_original, gaussian_noise, mode='same')
        prob = temp_conv / sum(temp_conv)
        p_at_N_record = sum(prob[np.where(x >= N_record)[0]])
        p_at_zero = sum(prob[np.where(x <= 0)[0]])
        pdf_final = prob / (x[1] - x[0])  # so that the integral is 1
        p_joint_with_noise[n_h, x[np.logical_and(x >= 0, x <= N_record)]] = pdf_final[
            np.logical_and(x >= 0, x <= N_record)]
        p_joint_with_noise[n_h, 0] = p_at_zero
        p_joint_with_noise[n_h, -1] = p_at_N_record

    bins_edge = np.arange(n_bins)
    p_joint_with_noise = p_joint_with_noise / np.sum(p_joint_with_noise)
    p_joint = p_joint / np.sum(p_joint)
    p_x = np.ones(len(logh_list)) * (1 / len(logh_list))
    y = bins_edge + (bins_edge[1]-bins_edge[0]) / 2
    MI_theo[l] = FODR_utils.entropy(p_x) - FODR_utils.entropy_x_given_y(p_joint_with_noise, y)

with open(
        path_to_save + '/theo/MI_sigma_noise=' + str(sigma_noise) + '.pickle', 'wb') as f:
    pickle.dump(MI_theo, f)


#theoretical T=inf
MI_inf=np.zeros(len(eps_list))
if 1:
    for l, lambda_ in enumerate(l_list):
        print('lambda=' + str(lambda_))
        p_joint = np.zeros((len(logh_list), n_bins))
        p_joint_with_noise = np.zeros((len(logh_list), n_bins))
        epsilon=eps_list[l]
        for g, logh_ in enumerate(logh_list):
            print('g=',g)
            n_h = [i for i in range(len(logh_list)) if logh_list[i] == logh_][0]
            p_val = 1 - np.exp(-1 * np.power(10, logh_))
            mean = int(FODR_utils.compute_meanfield_n('inSub1_outAll',N, mu, p_val, lambda_))
            p_joint[g, mean] = 1

            k_noise = 4
            bound_left = mean - int(k_noise * sigma_noise_adapted)
            bound_right = mean + int( k_noise * sigma_noise_adapted)
            x = np.arange(bound_left, bound_right)
            temp=stats.norm.pdf(x, mean, sigma_noise_adapted)
            prob = temp / sum(temp)
            p_at_N_record = sum(prob[np.where(x >= N_record)[0]])
            p_at_zero = sum(prob[np.where(x <= 0)[0]])
            p_joint_with_noise[n_h, x[np.logical_and(x >= 0, x <= N_record)]] = prob[
                np.logical_and(x >= 0, x <= N_record)]
            p_joint_with_noise[n_h, 0] = p_at_zero
            p_joint_with_noise[n_h, -1] = p_at_N_record

        p_joint = p_joint / np.sum(p_joint)
        p_joint_with_noise = p_joint_with_noise / np.sum(p_joint_with_noise)
        p_x = np.ones(len(logh_list)) * (1 / len(logh_list))
        y = bins_edge[:-1] + (bins_edge[1]-bins_edge[0]) / 2
        MI_inf[l] = FODR_utils.entropy(p_x) - FODR_utils.entropy_x_given_y(p_joint_with_noise, y)
    with open(
            path_to_save + '/inf/MI_sigma_noise=' + str(sigma_noise) + '.pickle', 'wb') as f:
        pickle.dump(MI_inf, f)

