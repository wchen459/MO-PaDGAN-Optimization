"""
Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import itertools
import numpy as np
import gpflowopt
import matplotlib.pyplot as plt

import functions
from cfg_reader import read_config


if __name__ == "__main__":
    
    list_data_function = [('Ring2D', 'VLMOP2'), ('Ring2D', 'NKNO1')]
    list_models = ['MO-PaDGAN', 'GAN']
    config_fname = 'config.ini'
    n_runs = 10
    
#    for (dataset, function) in list_data_function:
#        example_name = '{}+{}'.format(dataset, function)
#        for model_name in list_models:
#            lambda0, lambda1 = 0., 0.
#            if model_name == 'MO-PaDGAN':
#                _, _, _, _, _, lambda0_, lambda1_, _ = read_config(config_fname, example_name)
#                lambda0, lambda1 = lambda0_, lambda1_
#            png_path = './trained_gan/{}_{}/{}_{}/synthesized.png'.format(dataset, function, lambda0, lambda1)
#            if not os.path.exists(png_path):
#                os.system('python train.py train {} {} --lambda0={} --lambda1={}'.format(
#                          dataset, function, lambda0, lambda1))
    
    # Plot optimization history
    for (dataset, function) in list_data_function:
        
        example_name = '{}+{}'.format(dataset, function)
        
        linestyles = ['-', '--']
        iter_linestyles = itertools.cycle(linestyles)
        colors = ['#003f5c', '#ffa600']
        iter_colors = itertools.cycle(colors)
        
        plt.figure(figsize=(7, 7))
        
        for model_name in list_models:
                
            if model_name == 'GAN':
                lambda0, lambda1 = 0., 0.
                
            else:
                _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname, example_name)
                
            save_dir = 'trained_gan/{}_{}/{}_{}/optimization'.format(dataset, function, lambda0, lambda1)
            
            list_hv_history = []
            for i in range(n_runs):
                hv_history_path = '{}/{}/hv_history.npy'.format(save_dir, i)
                if not os.path.exists(hv_history_path):
                    os.system('python optimize.py {} {} --lambda0={} --lambda1={} --id={}'.format(dataset, function, lambda0, lambda1, i))
                list_hv_history.append(np.load(hv_history_path))
                
            list_hv_history = np.array(list_hv_history)
            mean_hv_history = np.mean(list_hv_history, axis=0)
            std_hv_history = np.std(list_hv_history, axis=0)
            
            iters = np.arange(1, len(mean_hv_history)+1)
            color = next(iter_colors)
            plt.plot(iters, mean_hv_history, ls=next(iter_linestyles), c=color, label=model_name)
            plt.fill_between(iters, mean_hv_history-std_hv_history, mean_hv_history+std_hv_history, color=color, alpha=.2)
            
        plt.legend(frameon=True)
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Hypervolume Indicator')
        plt.tight_layout()
        plt.savefig('trained_gan/{}_{}/opt_history.svg'.format(dataset, function))
        plt.close()
        
        # Plot Pareto points
        iter_colors = itertools.cycle(colors)
        markers = ['s', '^']
        iter_markers = itertools.cycle(markers)
        
        plt.figure(figsize=(7, 7))
        
        for model_name in list_models:
            
            if model_name == 'GAN':
                lambda0, lambda1 = 0., 0.
                
            else:
                _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname, example_name)
                
            save_dir = 'trained_gan/{}_{}/{}_{}/optimization'.format(dataset, function, lambda0, lambda1)
            
            list_pareto = []
            for i in range(n_runs):
                y_history_path = '{}/{}/y_hist.npy'.format(save_dir, i)
                y_history = np.load(y_history_path)
                pf, _ = gpflowopt.pareto.non_dominated_sort(y_history)
                list_pareto.append(pf)
            list_pareto = np.concatenate(list_pareto, axis=0)
            
            plt.scatter(-list_pareto[:,0], -list_pareto[:,1], c=next(iter_colors), marker=next(iter_markers), label=model_name)
            
        func_obj = getattr(functions, function)()
        true_pf = func_obj.pareto_y
        plt.plot(true_pf[:,0], true_pf[:,1], c='#bc5090', label='True Pareto front')
            
        plt.legend(frameon=True)
        plt.xlabel(r'$y_1$')
        plt.ylabel(r'$y_2$')
        plt.tight_layout()
        plt.savefig('trained_gan/{}_{}/pareto_pts.svg'.format(dataset, function))
        plt.close()
                