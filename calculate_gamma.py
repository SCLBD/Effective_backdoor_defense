import numpy as np
import argparse

parser = argparse.ArgumentParser('argument for calculating gamma')
parser.add_argument('--checkpoint_load', type=str, default=None, help='path to save backdoored model')
parser.add_argument('--clean_ratio', type=float, default=0.20, help='ratio of clean data')
parser.add_argument('--poison_ratio', type=float, default=0.05, help='ratio of poisoned data')
arg = parser.parse_args()

path = arg.checkpoint_load
f_all = path[:-4] + '_all.txt'
f_clean = path[:-4] + '_clean.txt'
f_poison = path[:-4] + '_poison.txt'

all_data = np.loadtxt(f_all)
all_size = all_data.shape[0] # 50000

clean_size = int(all_size * arg.clean_ratio) # 10000
poison_size = int(all_size * arg.poison_ratio) # 2500

new_data = np.sort(all_data) # in ascending order
gamma_low = new_data[clean_size]
gamma_high = new_data[all_size-poison_size]
print("gamma_low: ", gamma_low)
print("gamma_high: ", gamma_high)
