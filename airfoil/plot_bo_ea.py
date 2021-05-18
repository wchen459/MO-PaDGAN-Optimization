"""
Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from cfg_reader import read_config
from run_batch_experiments_bo import non_dominated_sort, plot_airfoils, novelty_score


if __name__ == "__main__":
    
    list_models = ['MO-PaDGAN', 'GAN', 'SVD', 'FFD']
    list_optimizers = ['bo', 'ea']
    config_fname = 'config.ini'
    n_runs = 10
    
    ''' Plot optimization history '''
    fig, axes = plt.subplots(ncols=len(list_optimizers), nrows=1,
                             sharex=False, sharey=True,
                             figsize=(10, 5))
    
    for k, optimizer_name in enumerate(list_optimizers):
        
        if optimizer_name == 'bo':
            axes[k].set_title('MOBO')
        elif optimizer_name == 'ea':
            axes[k].set_title('MOEA (C-TAEA)')
        
        linestyles = ['-', '--', ':', '-.']
        iter_linestyles = itertools.cycle(linestyles)
        colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
        iter_colors = itertools.cycle(colors)
    
        for model_name in list_models:
            
            if model_name == 'FFD':
                parameterization = 'ffd'
                
            elif model_name == 'SVD':
                parameterization = 'svd'
                
            elif model_name == 'GAN':
                parameterization = 'gan'
                lambda0, lambda1 = 0., 0.
                
            else:
                parameterization = 'gan'
                _, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
                
            if parameterization == 'gan':
                save_dir = 'trained_gan/{}_{}/optimization_{}'.format(lambda0, lambda1, optimizer_name)
            else:
                save_dir = '{}/optimization_{}'.format(parameterization, optimizer_name)
            
            list_hv_history = []
            for i in range(n_runs):
                hv_history_path = '{}/{}/hv_history.npy'.format(save_dir, i)
                list_hv_history.append(np.load(hv_history_path))
                
            list_hv_history = np.array(list_hv_history)
            mean_hv_history = np.mean(list_hv_history, axis=0)
            std_hv_history = np.std(list_hv_history, axis=0)
            
            iters = np.arange(1, len(mean_hv_history)+1)
            color = next(iter_colors)
            axes[k].plot(iters, mean_hv_history, ls=next(iter_linestyles), c=color, label=model_name)
            axes[k].fill_between(iters, mean_hv_history-std_hv_history, mean_hv_history+std_hv_history, color=color, alpha=.2)
        
        axes[k].set_xlabel('Number of Evaluations')
        if k == 0:
            axes[k].set_ylabel('Hypervolume Indicator')
            axes[k].legend(frameon=True, title='Parameterization')
        
    plt.tight_layout()
    plt.savefig('opt_history.svg')
    plt.close()
    
    ''' Plot Pareto points '''
    dict_pf_x_sup = dict()
    dict_pf_y_sup = dict()
    
    fig, axes = plt.subplots(ncols=len(list_optimizers), nrows=1,
                             sharex=False, sharey=True,
                             figsize=(10, 5))
    
    for k, optimizer_name in enumerate(list_optimizers):
        
        if optimizer_name == 'bo':
            axes[k].set_title('MOBO')
        elif optimizer_name == 'ea':
            axes[k].set_title('MOEA (C-TAEA)')
        
        iter_colors = itertools.cycle(colors)
        markers = ['s', '^', 'o', 'v']
        iter_markers = itertools.cycle(markers)
    
        for model_name in list_models:
            
            if model_name == 'FFD':
                parameterization = 'ffd'
                
            elif model_name == 'SVD':
                parameterization = 'svd'
                
            elif model_name == 'GAN':
                parameterization = 'gan'
                lambda0, lambda1 = 0., 0.
                
            else:
                parameterization = 'gan'
                _, _, _, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
                
            if parameterization == 'gan':
                save_dir = 'trained_gan/{}_{}/optimization_{}'.format(lambda0, lambda1, optimizer_name)
            else:
                save_dir = '{}/optimization_{}'.format(parameterization, optimizer_name)
            
            list_pareto_x = []
            list_pareto_y = []
            for i in range(n_runs):
                x_history_path = '{}/{}/x_hist.npy'.format(save_dir, i)
                y_history_path = '{}/{}/y_hist.npy'.format(save_dir, i)
                x_history = np.load(x_history_path)
                y_history = np.load(y_history_path)
                pf_y, pf_ind, _ = non_dominated_sort(y_history)
                pf_x = x_history[pf_ind]
                list_pareto_x.append(pf_x)
                list_pareto_y.append(pf_y)
            list_pareto_x = np.concatenate(list_pareto_x, axis=0)
            list_pareto_y = np.concatenate(list_pareto_y, axis=0)
            
            axes[k].scatter(-list_pareto_y[:,0], -list_pareto_y[:,1], c=next(iter_colors), marker=next(iter_markers), label=model_name)
            
            # Find the non-dominated set from all the Pareto sets
            pf_y_sup, pf_ind_sup, _ = non_dominated_sort(list_pareto_y)
            pf_x_sup = list_pareto_x[pf_ind_sup]
            dict_pf_x_sup[model_name+'+'+optimizer_name] = pf_x_sup
            dict_pf_y_sup[model_name+'+'+optimizer_name] = pf_y_sup
        
        axes[k].set_xlabel(r'$C_L$')
        if k == 0:
            axes[k].set_ylabel(r'$C_L/C_D$')
            axes[k].legend(frameon=True, title='Parameterization')
        
    plt.tight_layout()
    plt.savefig('pareto_pts.svg')
    plt.close()
    
    ''' Plot novelty scores for top-ranked airfoils '''
    airfoils_data = np.load('data/xs_train.npy')
    fig, axes = plt.subplots(ncols=len(list_optimizers), nrows=1,
                             sharex=False, sharey=True,
                             figsize=(10, 4))
    
    for k, optimizer_name in enumerate(list_optimizers):
        
        if optimizer_name == 'bo':
            axes[k].set_title('MOBO')
        elif optimizer_name == 'ea':
            axes[k].set_title('MOEA (C-TAEA)')
    
        gen_airfoils = []
        nearest_airfoils = []
        novelty_scores = []
        for i, model_name in enumerate(list_models):
            print('Computing novelty indicator for {}+{} ...'.format(model_name, optimizer_name.upper()))
            ys = []
            for airfoil in dict_pf_x_sup[model_name+'+'+optimizer_name]:
                y, nearest_airfoil = novelty_score(airfoil, airfoils_data)
                ys.append(y)
                gen_airfoils.append(airfoil)
                nearest_airfoils.append(nearest_airfoil)
                novelty_scores.append(y)
            xs = [i] * len(ys)
            axes[k].scatter(xs, ys, c='#003f5c', marker='o', s=100, alpha=0.3)
        
        axes[k].set_xticks(ticks=range(len(list_models)))
        axes[k].set_xticklabels(labels=list_models)
        if k == 0:
            axes[k].set_ylabel('Novelty indicator')
    
    plt.tight_layout()
    plt.savefig('pareto_novelty.svg')
    plt.savefig('pareto_novelty.pdf')
    plt.close()
        
                