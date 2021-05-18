"""
Airfoil batch simulation

Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import evaluate


if __name__ == "__main__":
    
    old_xs_train_path = './data/old_xs_train.npy'
    old_xs_test_path = './data/old_xs_test.npy'
    
    old_xs_train = np.load(old_xs_train_path)
    old_xs_test = np.load(old_xs_test_path)
    
    xs_train_path = './data/xs_train.npy'
    xs_test_path = './data/xs_test.npy'
    ys_train_path = './data/ys_train.npy'
    ys_test_path = './data/ys_test.npy'
    
    def evaluate_batch(xs_old):
        N = xs_old.shape[0]
        n_points = xs_old.shape[1]
        xs = np.zeros((0, n_points, 2))
        ys = np.zeros((0, 2))
        for i, airfoil in enumerate(xs_old):
            cl, cd = evaluate(airfoil, config_fname='./op_conditions.ini', tmp_dir='./tmp')
            if not (np.isinf(cl) and np.isinf(cd)):
                xs = np.append(xs, np.expand_dims(airfoil, 0), axis=0)
                ys = np.append(ys, np.array([cl, cd], ndmin=2), axis=0)
            print('{}/{}: CL={}, CD={}'.format(i+1, N, cl, cd))
        return xs, ys
        
    if not os.path.exists(ys_train_path) or not os.path.exists(ys_test_path):
        xs_train, ys_train = evaluate_batch(old_xs_train)
        xs_test, ys_test = evaluate_batch(old_xs_test)
        np.save(xs_train_path, xs_train)
        np.save(xs_test_path, xs_test)
        np.save(ys_train_path, ys_train)
        np.save(ys_test_path, ys_test)
        
    else:
        xs_train = np.load(xs_train_path)
        xs_test = np.load(xs_test_path)
        ys_train = np.load(ys_train_path)
        ys_test = np.load(ys_test_path)
    
    plt.figure()
    plt.scatter(ys_train[:,0], ys_train[:,0]/ys_train[:,1], s=10, c='#BCC6CC', marker='.')
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.tight_layout()
    plt.savefig('./data/train.png')
    plt.savefig('./data/train.svg')
    plt.close()
    
    plt.figure()
    plt.scatter(ys_test[:,0], ys_test[:,0]/ys_test[:,1], s=10, c='#BCC6CC', marker='.')
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.tight_layout()
    plt.savefig('./data/test.png')
    plt.savefig('./data/test.svg')
    plt.close()