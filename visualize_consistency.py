import numpy as np
from matplotlib import pyplot as plt
from utils import args

global arg
arg = args.get_args()

f_clean = arg.checkpoint_load
f_clean = f_clean[:-4] + '_clean.txt'
f_poison = arg.checkpoint_load
f_poison = f_poison[:-4] + '_poison.txt'

clean = np.loadtxt(f_clean)
poison = np.loadtxt(f_poison)
all = np.hstack((clean,poison))

num_bar = 1000
x_axis = np.linspace(np.min(all), np.max(all), num=num_bar+1)
count_clean, count_poison = [], []
for i in range(x_axis.shape[0]-1):
    left = x_axis[i]
    right = x_axis[i+1]
    if i != x_axis.shape[0]-2:
        count_clean.append(np.sum(((clean >= left) & (clean < right))))
        count_poison.append(np.sum(((poison >= left) & (poison < right))))
    else:
        count_clean.append(np.sum(((clean >= left) & (clean <= right))))
        count_poison.append(np.sum(((poison >= left) & (poison <= right))))
count_clean, count_poison = np.array(count_clean), np.array(count_poison)
step = (np.max(all) - np.min(all)) / num_bar
x_axis = x_axis[:-1] + step/2
print(np.sum(count_poison) + np.sum(count_clean))

plt.figure()
plt.bar(x=x_axis, height=count_clean, width=step, color='g', label = 'clean samples')
plt.bar(x=x_axis, height=count_poison, width=step, color='r', label= 'poisoned samples')

plt.xlabel(r'$\Delta_{trans}(x;\tau,f)$', fontsize=20)
plt.ylabel('Num of Samples', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=20)
save_path = arg.checkpoint_load
save_path = save_path[:-4] + '.jpg'
plt.savefig(save_path)
plt.show()
