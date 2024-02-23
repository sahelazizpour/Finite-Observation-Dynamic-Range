import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker
import FODR_utils

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)

n_realization = 1
error = 0.2
sigma_noise = 0.01

window_list = [1, 10, 100, 1000, 10000]
sub_typ = 'Full'
# sub_typ='Random'
path='results/FODR_withNoise_noWall/sub'+sub_typ+'/simu/'
path_to_MI='results//MI_withNoise_noWall/sub'+sub_typ+'/'

measure_type='FODR'
eps_list = np.logspace(-4, 0, 9)
xpoints = np.log10(eps_list)
eps_peak = np.zeros((len(window_list), n_realization))
lambda_peak = np.zeros((len(window_list), n_realization))

markers = ['o', 's', '^', 'P', 'd']
legends = ['$1$ ms', '$10$ ms', '$10^2$ ms', '$10^3$ ms', '$10^4$ ms']
colors =['darkorange','darkcyan','darkslateblue' ]
measure_list=['FODR','MI','no. points']
legend_list=['$\Delta$','$\I$','n_{d}']

for k, measure_type in enumerate(measure_list):
    for n, window in enumerate(window_list):
        measure = np.zeros((len(eps_list), n_realization))
        for i, nn in enumerate(range(n_realization)):
            with open(path_to_MI + '/simu/MI_window=' + str(window) + '_realization=' + str(nn) + '_sigma_noise=' + str(sigma_noise) + '.pickle', 'rb') as f:
                        measure[:, i] = pickle.load(f)
            if measure_type != 'MI':
                for l, epsilon in enumerate(eps_list):
                    df_output = pd.read_pickle(
                        path + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                            nn) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')
                    measure[l, i] = np.array(df_output.loc[0, measure_type])

            ypoints = measure[:, i]
            points = []
            for j in range(len(xpoints)):
                points.append([xpoints[j], ypoints[j]])
            # Get the Bezier parameters based on a degree.
            data = FODR_utils.get_bezier_parameters(xpoints, ypoints, degree=4)
            x_val = [x[0] for x in data]
            y_val = [x[1] for x in data]
            xvals, yvals = FODR_utils.bezier_curve(data, nTimes=1000)
            # plt.scatter(xpoints, ypoints, label='Original Points')
            # plt.plot(xvals, yvals, "b_", label='fitted line')
            eps_peak[n, i] = np.power(10, xvals[np.argwhere(yvals == np.max(yvals))[0][0]])
            lambda_peak[n, i] = 1 - np.power(10, xvals[np.argwhere(yvals == np.max(yvals))[0][0]])
    # ax.errorbar(window_list,np.median(lambda_peak,axis=1),yerr=np.sqrt(lambda_peak.var(axis=1))/np.sqrt(n_realization) )
    ax.scatter(window_list, np.median(eps_peak, axis=1), s=100, color=colors[k])
    ax.plot(window_list, np.median(eps_peak, axis=1),color=colors[k] ,label=legend_list[k])

plt.xscale('log')
plt.yscale('log')
plt.gca().invert_yaxis()
ax.set_yticks([1e-3, 1e-2, 1e-1])
ax.set_yticklabels(['0.999', '0.99', '0.9'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='y', length=4, width=2, labelsize=20)
ax.tick_params(axis='x', length=4, width=2, labelsize=20)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
x_major = matplotlib.ticker.LogLocator(base=10.0, numticks=10)
ax.xaxis.set_major_locator(x_major)
x_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
ax.xaxis.set_minor_locator(x_minor)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.tick_params(which='minor', width=2, length=5)
ax.tick_params(which='major', width=3, length=10)
ax.set_xlabel('time window $T$', color='black', fontsize=22)
ax.set_ylabel('control parameter $\lambda$', color='black', fontsize=22)
# plt.legend(fontsize='large', loc=3)
plt.tight_layout()
# plt.savefig('/home/sahelazizpour/PycharmProjects/BranchingNetwork/Results/lambda_peak.pdf')