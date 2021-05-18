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
    config_fname = 'config.ini'
    n_runs = 10
    
    ''' Plot optimization history '''
    linestyles = ['-', '--', ':', '-.']
    iter_linestyles = itertools.cycle(linestyles)
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    iter_colors = itertools.cycle(colors)
    
    plt.figure()
    
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
            save_dir = 'trained_gan/{}_{}/optimization_ea'.format(lambda0, lambda1)
        else:
            save_dir = '{}/optimization_ea'.format(parameterization)
        
        list_hv_history = []
        for i in range(n_runs):
            hv_history_path = '{}/{}/hv_history.npy'.format(save_dir, i)
            if not os.path.exists(hv_history_path):
                if parameterization == 'gan':
                    os.system('python optimize_ea.py {} --lambda0={} --lambda1={} --id={}'.format(parameterization, lambda0, lambda1, i))
                else:
                    os.system('python optimize_ea.py {} --id={}'.format(parameterization, i))
            list_hv_history.append(np.load(hv_history_path))
            
        list_hv_history = np.array(list_hv_history)
        mean_hv_history = np.mean(list_hv_history, axis=0)
        std_hv_history = np.std(list_hv_history, axis=0)
        
        iters = np.arange(1, len(mean_hv_history)+1)
        color = next(iter_colors)
        plt.plot(iters, mean_hv_history, ls=next(iter_linestyles), c=color, label=model_name)
        plt.fill_between(iters, mean_hv_history-std_hv_history, mean_hv_history+std_hv_history, color=color, alpha=.2)
        
    plt.legend(frameon=True, title='Parameterization')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Hypervolume Indicator')
    plt.tight_layout()
    plt.savefig('opt_history_ea.svg')
    plt.close()
    
    ''' Plot Pareto points '''
    iter_colors = itertools.cycle(colors)
    markers = ['s', '^', 'o', 'v']
    iter_markers = itertools.cycle(markers)
    dict_pf_x_sup = dict()
    dict_pf_y_sup = dict()
    
    plt.figure()
    
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
            save_dir = 'trained_gan/{}_{}/optimization_ea'.format(lambda0, lambda1)
        else:
            save_dir = '{}/optimization_ea'.format(parameterization)
        
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
        
        plt.scatter(-list_pareto_y[:,0], -list_pareto_y[:,1], c=next(iter_colors), marker=next(iter_markers), label=model_name)
        
        # Find the non-dominated set from all the Pareto sets
        pf_y_sup, pf_ind_sup, _ = non_dominated_sort(list_pareto_y)
        pf_x_sup = list_pareto_x[pf_ind_sup]
        dict_pf_x_sup[model_name] = pf_x_sup
        dict_pf_y_sup[model_name] = pf_y_sup
        
    plt.legend(frameon=True, title='Parameterization')
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.tight_layout()
    plt.savefig('pareto_pts_ea.svg')
    plt.close()
    
    ''' Plot top-ranked airfoils '''
    max_n_airfoils = max([len(dict_pf_x_sup[model_name]) for model_name in list_models])
    fig = plt.figure(figsize=(len(list_models)*2.4, max_n_airfoils*1.5))
    
    for i, model_name in enumerate(list_models):
    
        # Plot top-ranked airfoils
        ax = fig.add_subplot(1, len(list_models), i+1)
        ind = np.argsort(dict_pf_y_sup[model_name][:,0])
        plot_airfoils(dict_pf_x_sup[model_name][ind], -dict_pf_y_sup[model_name][ind], ax)
        ax.set_title(model_name)
        
    plt.tight_layout()
    plt.savefig('pareto_airfoils_ea.svg')
    plt.close()
    
    ''' Plot novelty scores for top-ranked airfoils '''
    airfoils_data = np.load('data/xs_train.npy')
    fig = plt.figure(figsize=(6.5, 3.5))
    
    gen_airfoils = []
    nearest_airfoils = []
    novelty_scores = []
    for i, model_name in enumerate(list_models):
        print('Computing novelty indicator for {} ...'.format(model_name))
        ys = []
        for airfoil in dict_pf_x_sup[model_name]:
            y, nearest_airfoil = novelty_score(airfoil, airfoils_data)
            ys.append(y)
            gen_airfoils.append(airfoil)
            nearest_airfoils.append(nearest_airfoil)
            novelty_scores.append(y)
        xs = [i] * len(ys)
        plt.scatter(xs, ys, c='#003f5c', marker='o', s=100, alpha=0.3)
        
    plt.xticks(ticks=range(len(list_models)), labels=list_models)
    plt.xlim([-.5, len(list_models)-.5])
    plt.ylabel('Novelty indicator')
    plt.tight_layout()
    plt.savefig('pareto_novelty_ea.svg')
    plt.savefig('pareto_novelty_ea.pdf')
    plt.close()
    
    ''' Plot generated airfoils and their nearest neighbors '''
    for i, score in enumerate(novelty_scores):
        plt.figure()
        plt.plot(gen_airfoils[i][:,0], gen_airfoils[i][:,1], 'r-', alpha=.5)
        plt.plot(nearest_airfoils[i][:,0], nearest_airfoils[i][:,1], 'b-', alpha=.5)
        plt.axis('equal')
        plt.title('{:.6f}'.format(score))
        plt.tight_layout()
        plt.savefig('tmp/pareto_novelty_{}.svg'.format(i))
        plt.close()
        
                