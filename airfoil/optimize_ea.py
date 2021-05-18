"""
Optimizes airfoil

Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import time
import argparse
import itertools
import gpflow
import gpflowopt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from cfg_reader import read_config
from opt_functions import *
from pymoo.model.problem import Problem
from pymoo.algorithms.ctaea import CTAEA
from pymoo.factory import get_reference_directions, get_termination
from pymoo.optimize import minimize

import sys
sys.path.append('..')
from utils import create_dir


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Optimize')
    parser.add_argument('parameterization', type=str, default='gan', help='airfoil parameterization (GAN, SVD, or FFD)')
    parser.add_argument('--lambda0', type=float, default=None, help='lambda0')
    parser.add_argument('--lambda1', type=float, default=None, help='lambda1')
    parser.add_argument('--id', type=int, default=0, help='run ID')
    args = parser.parse_args()
    
    if args.parameterization == 'gan':
        
        config_fname = 'config.ini'
        latent_dim, noise_dim, bezier_degree, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
        
        if args.lambda0 is not None:
            lambda0 = args.lambda0
        if args.lambda1 is not None:
            lambda1 = args.lambda1
            
        print('#################################')
        print('# Airfoil')
        print('# lambda0 = {}'.format(lambda0))
        print('# lambda1 = {}'.format(lambda1))
        print('# Evolutionary Algorithm (EA)')
        print('# ID: {}'.format(args.id))
        print('#################################')
            
        model_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
        
    elif args.parameterization == 'svd':
        
        print('#################################')
        print('# Airfoil')
        print('# SVD')
        print('# Evolutionary Algorithm (EA)')
        print('# ID: {}'.format(args.id))
        print('#################################')
            
        model_dir = 'svd'
        
    else:
        
        print('#################################')
        print('# Airfoil')
        print('# FFD')
        print('# Evolutionary Algorithm (EA)')
        print('# ID: {}'.format(args.id))
        print('#################################')
            
        model_dir = 'ffd'
        
    create_dir(model_dir)
    res_dir = '{}/optimization_ea'.format(model_dir)
    create_dir(res_dir)
    res_dir = '{}/{}'.format(res_dir, args.id)
    create_dir(res_dir)
    x_hist_path = '{}/x_hist.npy'.format(res_dir)
    y_hist_path = '{}/y_hist.npy'.format(res_dir)
    
    if not os.path.exists(x_hist_path) or not os.path.exists(y_hist_path):
        
        t0 = time.time()
        
        if args.parameterization == 'gan':
            # Use BezierGAN as parameterization
            X = np.load('./data/xs_train.npy')
            af = AirfoilGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, lambda0, lambda1, model_dir)
        elif args.parameterization == 'svd':
            af = AirfoilSVD()
        else:
            af = AirfoilFFD()

        # Setup problem
        class OptProblem(Problem):

            def __init__(self, n_var, xl, xu):
                super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = af.obj_func(X)

        problem = OptProblem(n_var=af.dim,
                             xl=af.bounds[:, 0],
                             xu=af.bounds[:, 1])

        algorithm = CTAEA(ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=14), seed=None)

        termination = get_termination("n_gen", 11)

        print('Run EA ...')
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=None,
                       save_history=True,
                       verbose=True)
        
        run_time = time.time() - t0
        print('Wall time: {:.1f}s'.format(run_time))
        
        # Save optimization history
        c_hist = [res.history[i].pop.get('X') for i in range(len(res.history))]
        c_hist = np.concatenate(c_hist, axis=0)
        x_hist = af.synthesize(c_hist)
        np.save(x_hist_path, x_hist)
        y_hist = [res.history[i].pop.get('F') for i in range(len(res.history))]
        y_hist = np.concatenate(y_hist, axis=0)
        np.save(y_hist_path, y_hist)
    
    print('Loading training history ...')
    y_hist = np.load(y_hist_path)
    T = y_hist.shape[0]
    
    # Plot pareto front
    print('Plotting Pareto points ...')
    colors = ['#ffa600', '#bc5090', '#003f5c']
    iter_colors = itertools.cycle(colors)
    markers = ['s', '^', 'o']
    iter_markers = itertools.cycle(markers)
    
    plt.figure(figsize=(7, 7))
    for t in [T//3, 2*T//3, T]:
        tf.keras.backend.clear_session()
        pf, dom = gpflowopt.pareto.non_dominated_sort(y_hist[:t+1])
        plt.scatter(-pf[:,0], -pf[:,1], c=next(iter_colors), marker=next(iter_markers), label='{}'.format(t))
    plt.legend()
    plt.title('Pareto set')
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.savefig('{}/pareto_pts.svg'.format(res_dir))
    plt.close()
    
    # Plot optimization history
    print('Plotting training history ...')
    reference = np.zeros(2)
    hv_history = []
    for t in range(T):
        tf.keras.backend.clear_session()
        pareto = gpflowopt.pareto.Pareto(y_hist[:t+1])
        hv = pareto.hypervolume(reference)
        hv_history.append(hv)
    np.save('{}/hv_history.npy'.format(res_dir), hv_history)
        
    plt.figure(figsize=(7, 7))
    plt.plot(np.arange(1, T+1), hv_history, '-')
    plt.title('Optimization history')
    plt.xlabel('Number of evaluations')
    plt.ylabel('Hypervolume indicator')
    plt.savefig('{}/opt_history.svg'.format(res_dir))
    plt.close()
    
    
    
    
