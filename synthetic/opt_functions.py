"""
Optimization functions

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np

from gan import GAN


class SyntheticGAN(object):
    
    def __init__(self, noise_dim, lambda0, lambda1, model_dir, func):
        # Load trained model
        self.model = GAN(noise_dim, 2, lambda0, lambda1)
        self.model.restore(save_dir=model_dir)
        self.bounds = np.ones((noise_dim, 2)) * np.array([-2,2])
        self.synthesize = self.model.synthesize
        self.evaluate = func
        self.dim = noise_dim
        
    def simulate(self, C):
        self.Y = self.evaluate(self.synthesize(C))
    
    def obj_func(self, C):
        return -self.evaluate(self.synthesize(C))
        
        