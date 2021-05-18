import numpy as np


def find_ind_inf(ys):
    inf_ind = np.any(np.isinf(ys), axis=1)
    return inf_ind

xs_train_path = 'data/xs_train.npy'
ys_train_path = 'data/ys_train.npy'
xs_train = np.load(xs_train_path)
ys_train = np.load(ys_train_path)
inf_ind = find_ind_inf(ys_train)
ind = np.logical_not(inf_ind)
np.save(ys_train_path, ys_train[ind])
np.save(xs_train_path, xs_train[ind])

xs_test_path = 'data/xs_test.npy'
ys_test_path = 'data/ys_test.npy'
xs_test = np.load(xs_test_path)
ys_test = np.load(ys_test_path)
inf_ind = find_ind_inf(ys_test)
ind = np.logical_not(inf_ind)
np.save(ys_test_path, ys_test[ind])
np.save(xs_test_path, xs_test[ind])