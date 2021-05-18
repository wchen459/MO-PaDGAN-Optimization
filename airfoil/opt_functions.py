"""
Optimization functions

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np

from bezier_gan import BezierGAN
from ffd import synthesize as synthesize_ffd
from simulation import evaluate



def read_airfoil(file_path):
    try:
        airfoil = np.loadtxt(file_path, skiprows=1)
    except:
        airfoil = np.loadtxt(file_path, delimiter=',')
    return airfoil

class AirfoilGAN(object):
    
    def __init__(self, latent_dim, noise_dim, airfoil_dim, bezier_degree, lambda0, lambda1, model_dir):
        # Load trained model
        bounds = (0., 1.)
        self.model = BezierGAN(latent_dim, noise_dim, airfoil_dim, bezier_degree, bounds, lambda0, lambda1)
        self.model.restore(directory=model_dir)
        self.bounds = np.tile(bounds, reps=(latent_dim, 1))
        self.synthesize = self.model.synthesize
        self.dim = latent_dim
        
    def simulate(self, C):
        self.Y = np.apply_along_axis(lambda x: evaluate(self.synthesize(x)), 1, C)
        self.F = np.any(np.isinf(self.Y), axis=1)
    
    def obj_func(self, C):
        self.simulate(C)
        self.Y[self.F] = [0., 1.]
        return np.vstack((-self.Y[:,0], -self.Y[:,0]/self.Y[:,1])).T # -Cl, -Cl/Cd
    
    def cons_func(self, C):
        return np.expand_dims(2.*self.F - 1., 1)
    
    def sample_design_variables(self, n_sample):
        x = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size=(n_sample, self.dim))
        return np.squeeze(x)
    
    def sample_airfoil(self, n_sample):
        x = self.sample_design_variables(n_sample)
        airfoils = self.synthesize(x)
        return airfoils
    
    
class AirfoilSVD(AirfoilGAN):
    
    '''
    References:
    [1] Poole, D. J., Allen, C. B., & Rendall, T. C. (2015). Metric-based mathematical 
        derivation of efficient airfoil design variables. AIAA Journal, 53(5), 1349-1361.
    [2] Poole, D. J., Allen, C. B., & Rendall, T. (2019). Efficient Aero-Structural 
        Wing Optimization Using Compact Aerofoil Decomposition. In AIAA Scitech 2019 Forum (p. 1701).
    '''
    
    def __init__(self, latent_dim=5, data_path='data/xs_train.npy', 
                 base_path='naca0012_uniform_192.dat'):
        
        self.dim = latent_dim
        
        # Read data
        xs_train = np.load(data_path)
        # Select a subset of data
        n = 500
        ind = np.random.choice(xs_train.shape[0], n, replace=False)
        xs_train = xs_train[ind]
        # Make camber line consistent
        y_te = (xs_train[:,0,1] + xs_train[:,-1,1])/2
        xs_train[:,:,1] -= np.expand_dims(y_te, 1)
        self.airfoil0 = np.mean(xs_train, axis=0, keepdims=True)
        
        # SVD for deformation
        xs_train = np.transpose(xs_train, (0,2,1)).reshape(xs_train.shape[0], -1)
        M = xs_train.shape[0]
        N = xs_train.shape[1]
        print('Computing deformation vectors ...')
        psi = np.zeros((N, M*(M-1)//2)) # N x M(M-1)/2
        for i in range(M):
            for j in range(i+1,M):
                idx = i*(M-1)-i*(i+1)//2+j-1
                psi[:, idx] = np.abs(xs_train[i] - xs_train[j])
        print('Computing SVD ...')
        u, s, vh = np.linalg.svd(psi, full_matrices=False)
        self.u_truncated = u[:,:self.dim] # N x dim
        self.alpha0 = np.zeros(self.dim)
        
        # Compute latent variables
        self.alpha = np.dot(np.diag(s[:latent_dim]), vh[:latent_dim,:]).T
        self.bounds = np.zeros((latent_dim, 2))
        self.bounds[:,0] = np.min(self.alpha, axis=0)
        self.bounds[:,1] = np.max(self.alpha, axis=0)
        
    def synthesize(self, alpha):
        alpha = np.array(alpha, ndmin=2).T # dim x n_samples
        airfoils = self.u_truncated @ alpha # N x n_samples
        airfoils = airfoils.reshape(2, -1, alpha.shape[1])
        airfoils = np.transpose(airfoils, [2,1,0]) + self.airfoil0
        return np.squeeze(airfoils)
    
    
class AirfoilFFD(AirfoilGAN):
    
    def __init__(self, m=4, n=3, perturb=0.05, initial_path='naca0012.dat'):
        
        self.m, self.n = m, n
        
        # NACA 0012 as the original airfoil
        self.airfoil0 = read_airfoil(initial_path)
        x_min = np.min(self.airfoil0[:,0])
        x_max = np.max(self.airfoil0[:,0])
        z_min = np.min(self.airfoil0[:,1])
        z_max = np.max(self.airfoil0[:,1])
        Px = np.linspace(x_min, x_max, m, endpoint=True)
        Py = np.linspace(z_min, z_max, n, endpoint=True)
        x, y = np.meshgrid(Px, Py)
        P0 = np.stack((x, y), axis=-1)
        self.Px = P0[:,:,0]
        self.x0 = P0[:,:,1].flatten()
        
        self.dim = len(self.x0)
        self.bounds = np.zeros((self.dim, 2))
        self.bounds[:,0] = self.x0 - perturb
        self.bounds[:,1] = self.x0 + perturb
        
    def synthesize(self, alpha):
        alpha = np.array(alpha, ndmin=2)
        airfoils = np.apply_along_axis(lambda x: synthesize_ffd(x, self.airfoil0, self.m, self.n, self.Px), 1, alpha)
        return np.squeeze(airfoils)
        
        
if __name__ == "__main__":
    
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Qt5Agg')
        
    # af = AirfoilSVD()
    # alpha = np.random.uniform(af.bounds[:,0], af.bounds[:,1], size=(af.dim))
    # airfoil = af.synthesize(alpha)[0]
    
    # plt.figure()
    # plt.plot(airfoil[:,0], airfoil[:,1], 'o-')
    # plt.axis('equal')
    # plt.show()
    
    af = AirfoilSVD()
    airfoils = af.sample_airfoil(1000)
    np.save('svd/gen_xs.npy', airfoils)
    perf_gen = np.zeros((0, 2))
    for i, airfoil in enumerate(airfoils):
        cl, cd = evaluate(airfoil, config_fname='./op_conditions.ini', tmp_dir='./tmp')
        if np.isinf(cl) or np.isinf(cd):
            cl = 0
            cd = 1e-8
        perf_gen = np.append(perf_gen, np.array([cl, cd], ndmin=2), axis=0)
        print('{}: CL={}, CD={}'.format(i+1, cl, cd))
    np.save('svd/gen_ys.npy', perf_gen)
    
    af = AirfoilFFD()
    airfoils = af.sample_airfoil(1000)
    np.save('ffd/gen_xs.npy', airfoils)
    perf_gen = np.zeros((0, 2))
    for i, airfoil in enumerate(airfoils):
        cl, cd = evaluate(airfoil, config_fname='./op_conditions.ini', tmp_dir='./tmp')
        if np.isinf(cl) or np.isinf(cd):
            cl = 0
            cd = 1e-8
        perf_gen = np.append(perf_gen, np.array([cl, cd], ndmin=2), axis=0)
        print('{}: CL={}, CD={}'.format(i+1, cl, cd))
    np.save('ffd/gen_ys.npy', perf_gen)