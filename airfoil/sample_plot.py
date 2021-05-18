"""
Sample plot.

Author(s): Wei Chen (wchen459@gmail.com)
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from cfg_reader import read_config
from opt_functions import *
from shape_plot import plot_shape


def gen_grid(points_per_axis):
    ''' Generate a grid in a d-dimensional space 
        within the range [lb, rb] for each axis '''
    
    lincoords = []
    for i in range(2):
        lincoords.append(np.linspace(0., 1.*points_per_axis[i], points_per_axis[i]))
    coords = list(itertools.product(*lincoords))
    
    return np.array(coords)

def plot(func, fig, points_per_axis, position, title):
    n_airfoils = points_per_axis[0]*points_per_axis[1]
    airfoils = func.sample_airfoil(n_airfoils)
    ax = fig.add_subplot(1, 5, position)
    Z = gen_grid(points_per_axis)
    for (i, z) in enumerate(Z):
        plot_shape(airfoils[i], .75*z[0], .3*z[1], ax, 1, False, None, c='k', lw=1.2)
    ax.set_title(title)
    plt.axis('off')
    plt.axis('equal')
    return fig


class AirfoilData(object):
    
    def __init__(self):
        self.data = np.load('./data/xs_train.npy')
        
    def sample_airfoil(self, n_sample):
        ind = np.random.choice(self.data.shape[0], n_sample, replace=False)
        airfoils = self.data[ind]
        return airfoils


if __name__ == "__main__":
    
    points_per_axis = [3, 5]
    fig = plt.figure(figsize=(22.5, 3.15))
    
    ''' Data '''
    func = AirfoilData()
    fig = plot(func, fig, points_per_axis, 1, '(a) Data')
    
    ''' MO-PaDGAN '''
    config_fname = 'config.ini'
    latent_dim, noise_dim, bezier_degree, _, _, _, _, lambda0, lambda1, _ = read_config(config_fname)
    model_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
    X = np.load('./data/xs_train.npy')
    func = AirfoilGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, lambda0, lambda1, model_dir)
    fig = plot(func, fig, points_per_axis, 2, '(b) MO-PaDGAN')
    
    ''' GAN '''
    lambda0, lambda1 = 0.0, 0.0
    model_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
    func = AirfoilGAN(latent_dim, noise_dim, X.shape[1], bezier_degree, lambda0, lambda1, model_dir)
    fig = plot(func, fig, points_per_axis, 3, '(c) GAN')
    
    ''' SVD '''
    func = AirfoilSVD()
    fig = plot(func, fig, points_per_axis, 4, '(d) SVD')
        
    ''' FFD '''
    dim = 12
    func = AirfoilFFD()
    fig = plot(func, fig, points_per_axis, 5, '(e) FFD')
    
    plt.tight_layout()
    plt.savefig('airfoil_samples.svg')
    plt.savefig('airfoil_samples.pdf')
    plt.close()
        