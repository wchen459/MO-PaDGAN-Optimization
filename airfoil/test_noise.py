import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

from bezier_gan import BezierGAN
from simulation import evaluate
from cfg_reader import read_config


if __name__ == "__main__":
    
    # Read dataset
    data_fname = './data/xs_train.npy'
    perf_fname = './data/ys_train.npy'
    X = np.load(data_fname)
    Y = np.load(perf_fname)
    N = X.shape[0]
    
    # Read model hyperparameters
    config_fname = 'config.ini'
    latent_dim, noise_dim, bezier_degree, _, _, _, _, _, _, _ = read_config(config_fname)
    lambda0, lambda1 = 0., 0.
    bounds = (0., 1.)
    
    # Load model
    model = BezierGAN(latent_dim, noise_dim, 192, bezier_degree, bounds, lambda0, lambda1)
    model_dir = './trained_gan/{}_{}'.format(lambda0, lambda1)
    model.restore(directory=model_dir)
    
    n = 1000
        
    # Fixed noise
    airfoils_gen0 = model.synthesize(n)
    perf_gen0 = np.zeros((0, 2))
    for i, airfoil in enumerate(airfoils_gen0):
        cl, cd = evaluate(airfoil, config_fname='./op_conditions.ini', tmp_dir='./tmp')
        if np.isinf(cl) or np.isinf(cd):
            cl = 0
            cd = 1e-8
        perf_gen0 = np.append(perf_gen0, np.array([cl, cd], ndmin=2), axis=0)
        print('{}/{}: CL={}, CD={}, CL/CD={}'.format(i+1, n, cl, cd, cl/cd))
    
    # Varying noise
    latent = np.random.uniform(bounds[0], bounds[1], size=(n, latent_dim))
    noise = np.zeros((n, noise_dim))
    airfoils_gen1 = model.synthesize(latent, noise)
    perf_gen1 = np.zeros((0, 2))
    for i, airfoil in enumerate(airfoils_gen1):
        cl, cd = evaluate(airfoil, config_fname='./op_conditions.ini', tmp_dir='./tmp')
        if np.isinf(cl) or np.isinf(cd):
            cl = 0
            cd = 1e-8
        perf_gen1 = np.append(perf_gen1, np.array([cl, cd], ndmin=2), axis=0)
        print('{}/{}: CL={}, CD={}, CL/CD={}'.format(i+1, n, cl, cd, cl/cd))
        
    # Plot
    ind = np.random.choice(N, n, replace=False)
    plt.figure(figsize=(25.4, 9.6))
    
    plt.subplot(121)
    plt.scatter(Y[ind,0], Y[ind,0]/Y[ind,1], s=10, c='#BCC6CC', marker='.', label='Data')
    plt.scatter(perf_gen0[:,0], perf_gen0[:,0]/perf_gen0[:,1], s=10, c='k', marker='.', label='Generated')
    plt.legend(frameon=False)
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.tight_layout()
    plt.title('Fixed noise')
    
    plt.subplot(122)
    plt.scatter(Y[ind,0], Y[ind,0]/Y[ind,1], s=10, c='#BCC6CC', marker='.', label='Data')
    plt.scatter(perf_gen1[:,0], perf_gen1[:,0]/perf_gen1[:,1], s=10, c='k', marker='.', label='Generated')
    plt.legend(frameon=False)
    plt.xlabel(r'$C_L$')
    plt.ylabel(r'$C_L/C_D$')
    plt.tight_layout()
    plt.title('Varying noise')
    
    plt.savefig('{}/test_noise.png'.format(model_dir))
    plt.savefig('{}/test_noise.svg'.format(model_dir))
    plt.close()