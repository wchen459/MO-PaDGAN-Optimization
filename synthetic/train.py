"""
Author(s): Wei Chen (wchen459@gmail.com)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import seaborn as sns

import datasets
import functions
from gan import GAN
from visualization import visualize_2d
from cfg_reader import read_config

import sys
sys.path.append('..')
from utils import create_dir, ElapsedTimer


if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default='train', help='train or evaluate')
    parser.add_argument('data', type=str, default='Donut2D', help='dataset')
    parser.add_argument('func', type=str, default='VLMOP2', help='function')
    parser.add_argument('--lambda0', type=float, default=None, help='lambda0')
    parser.add_argument('--lambda1', type=float, default=None, help='lambda1')
    parser.add_argument('--disc_lr', type=float, default=None, help='learning rate for D')
    parser.add_argument('--gen_lr', type=float, default=None, help='learning rate for G')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    parser.add_argument('--train_steps', type=int, default=None, help='training steps')
    parser.add_argument('--save_interval', type=int, default=None, help='save interval')
    args = parser.parse_args()
    assert args.mode in ['train', 'evaluate']
    
    # Data
    N = 10000
    data_obj = getattr(datasets, args.data)(N)
    data = data_obj.data
    
    # Functions
    func_obj = getattr(functions, args.func)()
    func = func_obj.evaluate
    
    # Values (>=0)
    valuess = func(data)
    
    # Hyperparameters for GAN
    config_fname = 'config.ini'
    example_name = '{}+{}'.format(args.data, args.func)
    noise_dim, train_steps, batch_size, disc_lr, gen_lr, lambda0, lambda1, save_interval = read_config(config_fname, example_name)
    if args.lambda0 is not None:
        lambda0 = args.lambda0
    if args.lambda1 is not None:
        lambda1 = args.lambda1
    if args.disc_lr is not None:
        disc_lr = args.disc_lr
    if args.gen_lr is not None:
        gen_lr = args.gen_lr
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.train_steps is not None:
        train_steps = args.train_steps
    if args.save_interval is not None:
        save_interval = args.save_interval
        
    print('#################################')
    print('# {}'.format(args.data))
    print('# {}'.format(args.func))
    print('# lambda0 = {}'.format(lambda0))
    print('# lambda1 = {}'.format(lambda1))
    print('# disc_lr = {}'.format(disc_lr))
    print('# gen_lr = {}'.format(gen_lr))
    print('#################################')
            
    # Prepare save directory
    create_dir('./trained_gan')
    example_dir = './trained_gan/{}_{}'.format(args.data, args.func)
    create_dir(example_dir)
    save_dir = '{}/{}_{}'.format(example_dir, lambda0, lambda1)
    create_dir(save_dir)
    
    # Visualize data
    visualize_2d(data, func_obj, save_path='{}/data.svg'.format(example_dir), xlim=(-0.5,0.5), ylim=(-0.5,0.5))
    visualize_2d(data, func_obj, save_path='{}/data.png'.format(example_dir), xlim=(-0.5,0.5), ylim=(-0.5,0.5))
    
    # Train
    model = GAN(noise_dim, 2, lambda0, lambda1)
    if args.mode == 'train':
        timer = ElapsedTimer()
        model.train(data_obj, func_obj, batch_size=batch_size, train_steps=train_steps, 
                    disc_lr=disc_lr, gen_lr=gen_lr, save_interval=save_interval, save_dir=save_dir)
        elapsed_time = timer.elapsed_time()
        runtime_mesg = 'Wall clock time for training: %s' % elapsed_time
        print(runtime_mesg)
    else:
        model.restore(save_dir=save_dir)
    
    print('##########################################################################')
    print('Plotting generated samples ...')
        
    # Plot generated samples
    n = 1000
    gen_data = model.synthesize(n)
    visualize_2d(data[:n], func_obj, gen_data=gen_data, save_path='{}/synthesized.svg'.format(save_dir), 
                 xlim=(-0.5,0.5), ylim=(-0.5,0.5), axis_off=False)
    visualize_2d(data[:n], func_obj, gen_data=gen_data, save_path='{}/synthesized.png'.format(save_dir), 
                 xlim=(-0.5,0.5), ylim=(-0.5,0.5), axis_off=False)
    
    print('##########################################################################')
    print('Plotting quality distribution ...')
    
    # Plot quality distribution
    n = 1000
    ind = np.random.choice(N, size=n)
    y_data = func(data[ind])
    y_gen = func(gen_data)
    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    sns.kdeplot(y_data[:,0], color='g', shade=True, Label='Data')
    sns.kdeplot(y_gen[:,0], color='b', shade=True, Label='Generated')
    plt.xlabel(r'$y_0$') 
    plt.ylabel('Probability density')
    plt.legend()
    plt.subplot(122)
    sns.kdeplot(y_data[:,1], color='g', shade=True, Label='Data')
    sns.kdeplot(y_gen[:,1], color='b', shade=True, Label='Generated')
    plt.xlabel(r'$y_1$') 
    plt.ylabel('Probability density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/quality_dist.svg'.format(save_dir))
    plt.savefig('{}/quality_dist.png'.format(save_dir))
    plt.close()
    
    print('##########################################################################')
    print('Plotting performance space ...')
    
    # Plot performance space
    plt.figure()
    plt.scatter(y_gen[:,0], y_gen[:,1], s=10, c='#003f5c', marker='s', label='Generated')
    plt.scatter(y_data[:,0], y_data[:,1], s=10, c='#ffa600', marker='o', label='Data')
    true_pf = func_obj.pareto_y
    plt.plot(true_pf[:,0], true_pf[:,1], c='#bc5090', label='True Pareto front')
    plt.legend(frameon=True)
    plt.xlabel(r'$y_0$')
    plt.ylabel(r'$y_1$')
    plt.tight_layout()
    plt.savefig('{}/perf_space.png'.format(save_dir))
    plt.savefig('{}/perf_space.svg'.format(save_dir))
    plt.close()
    
    print('Completed!')
