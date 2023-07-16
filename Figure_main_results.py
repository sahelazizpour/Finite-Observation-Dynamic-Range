import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import warnings
import pandas as pd
import pickle
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker
warnings.filterwarnings("ignore")

### run this script for each measure : simu, theo, inf
measure='MI'

#####simu

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
window_list=[1,10,100,1000,10000]
error=0.2
sigma_noise=1e-2
sub_typ='Full' #'Random'
mainpath='/home/sahelazizpour/PycharmProjects/BranchingNetwork/2023/results/FODR_withNoise_noWall/sub'+sub_typ
path=mainpath+'/simu/'
path_to_MI='/home/sahelazizpour/PycharmProjects/BranchingNetwork/2023/results//MI_withNoise_noWall/sub'+sub_typ+'/'

# path='/data/network_simulation/subpop_k=100/fixed_indegree/MI/subFull/with_noise/simu/'

eps_list=np.logspace(-4,0,9).tolist()
n_realization=10
markers=['o','s','^','P','d']
legends=['$1$ ms','$10$ ms','$10^2$ ms','$10^3$ ms','$10^4$ ms']
colors = np.flip(pl.cm.Blues(np.linspace(1,0,9))[:-4],axis=0)
for n,window in enumerate(window_list):
    dr=np.zeros((len(eps_list),n_realization));n_d=np.zeros_like(dr); res=np.zeros_like(dr); MI =np.zeros_like(dr)
    for i, nn in enumerate(range(n_realization)):
        with open(path_to_MI + '/simu/MI_window=' + str(window) + '_realization=' + str(nn) + '_sigma_noise=' + str(sigma_noise) + '.pickle', 'rb') as f:
                    MI[:, nn] = pickle.load(f)
        for l, epsilon in enumerate(eps_list):
            print('epsilon=', epsilon)
            df_output = pd.read_pickle(path +'/FODR_epsilon='+'{:.2e}'.format(epsilon)+'_window=' + str(window)+'_realization='+str(nn)+'_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) +'.pkl')
            dr[l,i]=np.array(df_output.loc[0, 'FODR'])
            n_d[l, i] = np.array(df_output.loc[0, 'no. points'])
            res[l, i] = n_d[l, i] / dr[l, i]

    if measure=='DR':
        ax.errorbar(eps_list, np.median(dr,axis=1),np.sqrt(dr.var(axis=1))/np.sqrt(n_realization),capsize=3,color=colors[n],linewidth=1.5, label='T=' + str((window_list[n]/1000))+'s')
        ax.scatter(eps_list, np.median(dr, axis=1),marker=markers[n], color=colors[n], s=100,label='$T=$'+ legends[n])
    if measure=="n_d":
        ax.errorbar(eps_list, np.median(n_d,axis=1),np.sqrt(n_d.var(axis=1))/np.sqrt(n_realization),capsize=3,color=colors[n],linewidth=1.5, label='T=' + str((window_list[n]/1000))+'s')
        ax.scatter(eps_list, np.median(n_d, axis=1),marker=markers[n], color=colors[n], s=100,label='$T=$'+ legends[n])
    if measure == "res":
        ax.errorbar(eps_list, np.median(res,axis=1),np.sqrt(res.var(axis=1))/np.sqrt(n_realization),capsize=3,color=colors[n],linewidth=1.5, label='T=' + str((window_list[n]/1000))+'s')
        ax.scatter(eps_list, np.median(res, axis=1),marker=markers[n], color=colors[n], s=100,label='$T=$'+ legends[n])
    if measure == "MI":
        ax.errorbar(eps_list, np.median(MI,axis=1),np.sqrt(MI.var(axis=1))/np.sqrt(n_realization),capsize=3,color=colors[n],linewidth=1.5, label='T=' + str((window_list[n]/1000))+'s')
        ax.scatter(eps_list, np.median(MI, axis=1),marker=markers[n], color=colors[n], s=100,label='$T=$'+ legends[n])


plt.xscale('log')
plt.gca().invert_xaxis()
ax.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0])
ax.set_xticklabels(['0.9999','0.999','0.99','0.9','0'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='y', length=4, width=2, labelsize=22)
ax.tick_params(axis='x', length=4, width=2, labelsize=22)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 10)
ax.xaxis.set_major_locator(x_major)
x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
ax.xaxis.set_minor_locator(x_minor)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.tick_params(which='minor',width=2, length=5)
ax.tick_params(which='major',width=3, length=10)
if measure=='DR':
    ax.set_ylabel('dynamic range $\Delta$', color='black', fontsize=28)
if measure=='n_d':
    ax.set_ylabel('no. inputs $n_d$', color='black', fontsize=28)
if measure=='res':
    ax.set_ylabel('resolution $n_d/\Delta$', color='black', fontsize=28)
if measure=='MI':
    ax.set_ylabel('mutual info $I$', color='black', fontsize=28)
ax.set_xlabel('control parameter $\lambda$', color='black', fontsize=28)
plt.tight_layout()

##############    theo
eps_list=np.logspace(-4,0,9).tolist()
window=1
n_realization=1
path=mainpath+'/theo/'
dr_theo=np.zeros(len(eps_list)); n_points_theo = np.zeros_like(dr_theo);res_theo=np.zeros_like(dr_theo); MI_theo=np.zeros_like(dr_theo)
for i, nn in enumerate(range(n_realization)):
    for l, epsilon in enumerate(eps_list):
        df_output = pd.read_pickle(path + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(
            window) + '_realization=' + str(nn) + '_error=' + str(error) + '_sigma_noise=' + str(
            sigma_noise) + '.pkl')
        dr_theo[l] = np.array(df_output.loc[0, 'FODR_theo'])
        n_points_theo[l] = np.array(df_output.loc[0, 'no. points_theo'])
        res_theo[l] = n_points_theo[l] / dr_theo[l]
if measure=='DR':
    ax.plot(eps_list, dr_theo, '--', linewidth=4, color='maroon',label='$T=1$ ms')
if measure=='n_d':
    ax.plot(eps_list, n_points_theo, '--', linewidth=4, color='maroon', label='$T=1$ ms' )
if measure=='res':
    ax.plot(eps_list, res_theo, '--', linewidth=4, color='maroon',label='$T=1$ ms')
if measure == "MI":
    with open(path_to_MI + '/theo/MI_sigma_noise=' + str(sigma_noise) + '.pickle', 'rb') as f:
                MI_theo = pickle.load(f)
    ax.plot(eps_list, MI_theo,'--', color='maroon', linewidth=4,label='T=1 ms, theo')


##############    inf
path=mainpath+'/inf/'
dr_inf = np.zeros(len(eps_list));n_points_inf = np.zeros_like(dr_inf);res_inf = np.zeros_like(dr_inf);MI_inf= np.zeros_like(dr_inf)
for i, nn in enumerate(range(n_realization)):
    for l, epsilon in enumerate(eps_list):
        df_output = pd.read_pickle(path +'/FODR_epsilon='+'{:.2e}'.format(epsilon)+'_window=' + str(window)+'_realization='+str(nn)+'_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) +'.pkl')
        dr_inf[l] = np.array(df_output.loc[0, 'FODR_inf'])
        n_points_inf[l]=np.array(df_output.loc[0, 'no. points_inf'])
        res_inf[l]=n_points_inf[l]/dr_inf[l]

if measure=='DR':
    ax.plot(eps_list, dr_inf, linewidth=4, color='maroon', label=r'$T=\rightarrow \infty$')
if measure == 'n_d':
    ax.plot(eps_list, n_points_inf,linewidth=4, color='maroon', label=r'$T=\rightarrow \infty$')
if measure == 'res':
    ax.plot(eps_list, res_inf,linewidth=4, color='maroon', label=r'$T=\rightarrow \infty$')
if measure == "MI":
    with open(path_to_MI + '/inf/MI_sigma_noise=' + str(sigma_noise) + '.pickle', 'rb') as f:
        MI_inf = pickle.load(f)
    ax.plot(eps_list, MI_inf, color='maroon', linewidth=4, label='T=$\infty$ ms, theo')


