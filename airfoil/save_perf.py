"""
Saves performances in npy files

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np
from run_experiment import read_config


def replace_nan_inf(perf):
    ind = np.logical_or(np.isinf(perf), np.isnan(perf))
    perf[ind] = 0
    return perf
    

# PaDGAN
config_fname = 'config.ini'
_, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
perf_fname = './trained_gan/{}_{}/gen_ys.npy'.format(lambda0, lambda1)
perf = np.load(perf_fname)
perf[:,1] = perf[:,0]/perf[:,1]
perf = replace_nan_inf(perf)
np.save('perf/ys_padgan.npy', perf)
n = perf.shape[0]

# GAN
perf_fname = './trained_gan/0.0_0.0/gen_ys.npy'
perf = np.load(perf_fname)
perf[:,1] = perf[:,0]/perf[:,1]
perf = replace_nan_inf(perf)
np.save('perf/ys_gan.npy', perf)

# Data
perf_fname = './data/ys_train.npy'
perf = np.load(perf_fname)[:n]
perf[:,1] = perf[:,0]/perf[:,1]
perf = replace_nan_inf(perf)
np.save('perf/ys_data.npy', perf)
