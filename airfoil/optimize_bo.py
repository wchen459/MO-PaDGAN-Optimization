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
        print('# Bayesian Optimization (BO)')
        print('# ID: {}'.format(args.id))
        print('#################################')
            
        model_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
        
    elif args.parameterization == 'svd':
        
        print('#################################')
        print('# Airfoil')
        print('# SVD')
        print('# Bayesian Optimization (BO)')
        print('# ID: {}'.format(args.id))
        print('#################################')
            
        model_dir = 'svd'
        
    else:
        
        print('#################################')
        print('# Airfoil')
        print('# FFD')
        print('# Bayesian Optimization (BO)')
        print('# ID: {}'.format(args.id))
        print('#################################')
            
        model_dir = 'ffd'
        
    create_dir(model_dir)
    res_dir = '{}/optimization_bo'.format(model_dir)
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
        
        # Setup input domain
        bounds = af.bounds
        domain = gpflowopt.domain.ContinuousParameter('x1', bounds[0,0], bounds[0,1])
        for d in range(1, af.dim):
            domain += gpflowopt.domain.ContinuousParameter('x{}'.format(d+1), bounds[d,0], bounds[d,1])
            
        # Set number of evaluations
        n_init_eval = 15#3 * af.dim
        n_eval = 150#30 * af.dim
    
        tf.keras.backend.clear_session()
        
        # Initial evaluations
        print('Initial evaluations ...')
        design = gpflowopt.design.LatinHyperCube(n_init_eval, domain)
        X = design.generate()
        Y = af.obj_func(X)
    
        tf.keras.backend.clear_session()
        
        # Models (one model for each objective)
        objective_models = [gpflow.gpr.GPR(X.copy(), Y[:,[i]].copy(), gpflow.kernels.Matern52(2, ARD=True)) for i in range(Y.shape[1])]
        for model in objective_models:
            model.likelihood.variance = 0.01
        
        hvpoi = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)
        
        # First setup the optimization strategy for the acquisition function
        # Combining MC step followed by L-BFGS-B
        acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, 1000),
                                                           gpflowopt.optim.SciPyOptimizer(domain)])
        
        # Then run the BayesianOptimizer for 20 iterations
        print('Run BO ...')
        optimizer = gpflowopt.BayesianOptimizer(domain, hvpoi, optimizer=acquisition_opt, verbose=True)
        
        result = optimizer.optimize([af.obj_func], n_iter=n_eval)
        
        run_time = time.time() - t0
        print('Wall time: {:.1f}s'.format(run_time))
        
        # Save optimization history
        x_hist = af.synthesize(hvpoi.data[0])
        y_hist = hvpoi.data[1]
        np.save(x_hist_path, x_hist)
        np.save(y_hist_path, y_hist)
    
        print('Results:')
        print(result)
    
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
    
    
    
    
