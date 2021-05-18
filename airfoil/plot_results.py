""" 
Plot results

Author(s): Wei Chen (wchen459@gmail.com)
"""

import os
import itertools
import numpy as np
from matplotlib import pyplot as plt

from cfg_reader import read_config
from shape_plot import plot_shape
from run_batch_experiments import novelty_score, non_dominated_sort
from simulation import detect_intersect

import sys
sys.path.append('..')
from evaluation import diversity_score


def convert_perf(perf):
    perf[:,1] = perf[:,0]/perf[:,1]
    ind = np.logical_or(np.isinf(perf), np.isnan(perf))
    perf[ind] = 0
    return perf

def select_best_ind(metric, n_selected, feasibility):
    sorted_ind = np.argsort(metric*feasibility)
    selected_ind = sorted_ind[-n_selected:]
    return selected_ind

def plot_airfoils(airfoils, airfoils_nearest, perfs, ax):
    n = airfoils.shape[0]
    zs = np.vstack((np.zeros(n), 0.6*np.arange(n))).T
    for i in range(n):
        plot_shape(airfoils[i], zs[i, 0], zs[i, 1], ax, 1., False, None, c='k', lw=1.2)
        plot_shape(airfoils_nearest[i], zs[i, 0], zs[i, 1], ax, 1., False, None, c='k', lw=1.2, ls='--', alpha=.5)
        plt.annotate(r'$C_L={:.2f}$, $C_L/C_D={:.2f}$'.format(perfs[i,0], perfs[i,1]), xy=(zs[i, 0], zs[i, 1]+0.3), size=14)
    ax.axis('off')
    ax.axis('equal')
    ax.set_ylim(zs[0,1]-0.2, zs[-1,1]+0.6)
    

if __name__ == "__main__":
    
    config_fname = 'config.ini'
    list_models = ['MO-PaDGAN', 'GAN', 'SVD', 'FFD']
    # list_models = ['MO-PaDGAN', 'GAN']
    m = len(list_models)
    
    ###############################################################################
    # Plot diversity, quality, and novelty scores
    print('Plotting scores ...')
    plt.rcParams.update({'font.size': 14})
    
    n = 300 # generated sample size for each trained model
    subset_size = 10 # for computing DDP
    sample_times = n # for computing DDP
    
    # Training data
    x_path = './data/xs_train.npy'
    airfoils_data = np.load(x_path)
    
    list_div = []
    list_qa0 = []
    list_qa1 = []
    list_nvl = []
    list_selected_ind = []
    
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
            save_dir = 'trained_gan/{}_{}'.format(lambda0, lambda1)
        else:
            save_dir = '{}'.format(parameterization)
        airfoils = np.load('{}/gen_xs.npy'.format(save_dir))[:n]
        
        div = diversity_score(airfoils, subset_size, sample_times)
        qa = np.load('{}/gen_ys.npy'.format(save_dir))[:n]
        qa = convert_perf(qa)
        
        nvl_path = '{}/novelty_scores.npy'.format(save_dir)
        nns_path = '{}/nearest_neighbors.npy'.format(save_dir)
        if os.path.exists(nvl_path) and os.path.exists(nns_path):
            nvl = np.load(nvl_path)
            nns = np.load(nns_path)
        else:
            nvl = []
            nns = []
            for i, airfoil in enumerate(airfoils):
                print('{}/{}'.format(i+1, n))
                dist, nearest_airfoil = novelty_score(airfoil, airfoils_data)
                nvl.append(dist)
                nns.append(nearest_airfoil)
            np.save(nvl_path, nvl)
            np.save(nns_path, nns)
        
        feasibility = np.logical_not(np.all(qa==0, axis=1))
        for i, airfoil in enumerate(airfoils):
            if detect_intersect(airfoil):
                feasibility[i] = False
        print('{}: {:.2f}%'.format(model_name, sum(feasibility)/len(feasibility)*100))
        selected_ind = select_best_ind(nvl, 5, feasibility)
            
        list_div.append(div)
        list_qa0.append(qa[:,0])
        list_qa1.append(qa[:,1])
        list_nvl.append(nvl)
        list_selected_ind.append(selected_ind)
    
    list_xlabels = list_models
    
    fig = plt.figure(figsize=(15, 3))
    ax1 = fig.add_subplot(141)
    ax1.set_title('Diversity')
    ax1.boxplot(list_div, 0, '')
    ax1.set_xlim(0.5, len(list_xlabels) + 0.5)
    ax1.set_xticklabels(list_xlabels, rotation=20)
    ax2 = fig.add_subplot(142)
    ax2.set_title(r'$C_L$')
    ax2.boxplot(list_qa0, 0, '')
    ax2.set_xlim(0.5, len(list_xlabels) + 0.5)
    ax2.set_xticklabels(list_xlabels, rotation=20)
    ax3 = fig.add_subplot(143)
    ax3.set_title(r'$C_L/C_D$')
    ax3.boxplot(list_qa1, 0, '')
    ax3.set_xlim(0.5, len(list_xlabels) + 0.5)
    ax3.set_xticklabels(list_xlabels, rotation=20)
    ax3 = fig.add_subplot(144)
    ax3.set_title('Novelty')
    ax3.boxplot(list_nvl, 0, '')
    ax3.set_xlim(0.5, len(list_xlabels) + 0.5)
    ax3.set_xticklabels(list_xlabels, rotation=20)
    plt.tight_layout()
    plt.savefig('./airfoil_scores.svg')
    plt.savefig('./airfoil_scores.png')
    plt.close()
    
    ###############################################################################
    # Plot most novel airfoils
    print('Plotting most novel airfoils ...')
    fig = plt.figure(figsize=(15, 6))
    
    for i, model_name in enumerate(list_models):
        
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
            save_dir = 'trained_gan/{}_{}'.format(lambda0, lambda1)
        else:
            save_dir = '{}'.format(parameterization)
        airfoils = np.load('{}/gen_xs.npy'.format(save_dir))[:n]
        airfoils_nearest = np.load('{}/nearest_neighbors.npy'.format(save_dir))
        perfs = np.load('{}/gen_ys.npy'.format(save_dir))[:n]
        perfs = convert_perf(perfs)
        
        ax = fig.add_subplot(1, m, i+1)
        plot_airfoils(airfoils[list_selected_ind[i]], airfoils_nearest[list_selected_ind[i]], perfs[list_selected_ind[i]], ax)
        ax.set_title(model_name)
        
    # plt.tight_layout()
    plt.savefig('./airfoil_most_novel.svg')
    plt.savefig('./airfoil_most_novel.png')
    plt.close()
    
    ###############################################################################
    # Plot Pareto front for a single run
    print('Plotting Pareto front for a single run ...')
    
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']
    iter_colors = itertools.cycle(colors)
    markers = ['s', '^', 'o', 'v']
    iter_markers = itertools.cycle(markers)
    
    fig = plt.figure()
    
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
            save_dir = 'trained_gan/{}_{}/optimization'.format(lambda0, lambda1)
        else:
            save_dir = '{}/optimization'.format(parameterization)
            
        y_history_path = '{}/0/y_hist.npy'.format(save_dir)
        y_history = np.load(y_history_path)
        pf_y, _, _ = non_dominated_sort(y_history)
        
        plt.scatter(-pf_y[:,0], -pf_y[:,1], c=next(iter_colors), marker=next(iter_markers), label=model_name)
    
    plt.legend(frameon=True, title='Parameterization')
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.tight_layout()
    plt.savefig('pareto_pts_1run.svg')
    plt.savefig('pareto_pts_1run.png')
    plt.close()
    
    