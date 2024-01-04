# arguement parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log10_eps', type=float, required=True)
parser.add_argument('--log10_h', type=float, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--N', type=int, default=int(1e4))
parser.add_argument('--K', type=int, default=100)
parser.add_argument('--mu', type=float, default=0.2)
parser.add_argument('--dt', type=float, default=1)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

import os, sys
rootpath = os.path.join(os.getcwd(), '.')
sys.path.append(rootpath)
from src.simulation import *

def __main__(args):
    params = {
        'N': args.N,
        'K': args.K,
        'lambda': 1-10**args.log10_eps,
        'mu': args.mu,
        'h': 10**args.log10_h,
        'seed': args.seed,
        'dt': args.dt,
    }
    print('start simulation')
    result = simulation(params)
    save_simulation(result, args.path)

if __name__ == '__main__':
    __main__(args)