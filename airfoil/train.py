"""
Trains a BezierGAN, and visulizes results

Author(s): Wei Chen (wchen459@gmail.com)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from bezier_gan import BezierGAN
from shape_plot import plot_samples
from simulation import evaluate
from cfg_reader import read_config

import sys
sys.path.append('..')
from utils import ElapsedTimer, create_dir, safe_remove


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    parser.add_argument('--lambda0', type=float, default=None, help='lambda0')
    parser.add_argument('--lambda1', type=float, default=None, help='lambda1')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    config_fname = 'config.ini'
    latent_dim, noise_dim, bezier_degree, train_steps, batch_size, disc_lr, gen_lr, lambda0, lambda1, save_interval = read_config(config_fname)
    bounds = (0., 1.)
    
    if args.lambda0 is not None:
        lambda0 = args.lambda0
    if args.lambda1 is not None:
        lambda1 = args.lambda1
        
    print('#################################')
    print('# Airfoil')
    print('# lambda0 = {}'.format(lambda0))
    print('# lambda1 = {}'.format(lambda1))
    print('# disc_lr = {}'.format(disc_lr))
    print('# gen_lr = {}'.format(gen_lr))
    print('#################################')
    
    # Read dataset
    data_fname = './data/xs_train.npy'
    perf_fname = './data/ys_train.npy'
    X = np.load(data_fname)
    Y = np.load(perf_fname)
    N = X.shape[0]
    assert N == Y.shape[0]
    
    # Prepare save directory
    create_dir('./trained_gan')
    save_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
    create_dir(save_dir)
    
#    print('Plotting training samples ...')
#    samples = X[np.random.choice(N, size=36, replace=False)]
#    plot_samples(None, samples, scale=1.0, scatter=False, lw=1.2, alpha=.7, c='k', fname='{}/samples'.format(save_dir))
    
    # Train
    surrogate_dir = './surrogate/trained_surrogate'
    model = BezierGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, bounds, lambda0, lambda1)
    if args.mode == 'train':
        safe_remove(save_dir)
        timer = ElapsedTimer()
        model.train(X, batch_size=batch_size, train_steps=train_steps, disc_lr=disc_lr, gen_lr=gen_lr, 
                    save_interval=save_interval, directory=save_dir, surrogate_dir=surrogate_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
        runtime_file = open('{}/runtime.txt'.format(save_dir), 'w')
        runtime_file.write('%s\n' % runtime_mesg)
        runtime_file.close()
    else:
        model.restore(directory=save_dir)
    
    print('Plotting synthesized shapes ...')
    airfoils = model.synthesize(36)
    plot_samples(None, airfoils, scale=1.0, scatter=False, lw=1.2, alpha=.7, c='k', fname='{}/synthesized'.format(save_dir))
    
    # # Plot quality distribution
    # n = 1000
    # ind = np.random.choice(X.shape[0], size=n)
    # airfoils_data = np.squeeze(X[ind])
    # airfoils_gen = model.synthesize(n)
    # with tf.Session() as sess:
    #     surrogate_model = SM(sess, X.shape[1])
    #     surrogate_model.restore(directory=surrogate_dir)
    #     quality_data = surrogate_model.predict(airfoils_data)
    #     quality_gen = surrogate_model.predict(airfoils_gen)
    
    gen_data_fname = '{}/gen_xs.npy'.format(save_dir)
    gen_perf_fname = '{}/gen_ys.npy'.format(save_dir)
    
    if os.path.exists(gen_perf_fname):
        
        perf_gen = np.load(gen_data_fname)
        
    else:
        
        n = 1000
        airfoils_gen = model.synthesize(n)
        perf_gen = np.zeros((0, 2))
        for i, airfoil in enumerate(airfoils_gen):
            cl, cd = evaluate(airfoil, config_fname='./op_conditions.ini', tmp_dir='./tmp')
            if np.isinf(cl) or np.isinf(cd):
                cl = 0
                cd = 1e-8
            perf_gen = np.append(perf_gen, np.array([cl, cd], ndmin=2), axis=0)
            print('{}/{}: CL={}, CD={}'.format(i+1, n, cl, cd))
            
        np.save(gen_data_fname, airfoils_gen)
        np.save(gen_perf_fname, perf_gen)
    
    plt.figure()
    plt.scatter(Y[:n,0], Y[:n,0]/Y[:n,1], s=10, c='#BCC6CC', marker='.', label='Data')
    plt.scatter(perf_gen[:,0], perf_gen[:,0]/perf_gen[:,1], s=10, c='k', marker='.', label='Generated')
    plt.legend(frameon=False)
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.tight_layout()
    plt.savefig('{}/perf_space.png'.format(save_dir))
    plt.savefig('{}/perf_space.svg'.format(save_dir))
    plt.close()
    
    
