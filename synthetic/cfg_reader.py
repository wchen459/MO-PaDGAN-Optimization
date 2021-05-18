"""
MO-PaDGAN configuration reader

Author(s): Wei Chen (wchen459@gmail.com)
"""

import configparser


def read_config(config_fname, example_name):
    
    Config = configparser.ConfigParser()
    Config.read(config_fname)
    noise_dim = int(Config.get(example_name, 'noise_dim'))
    train_steps = int(Config.get(example_name, 'train_steps'))
    batch_size = int(Config.get(example_name, 'batch_size'))
    disc_lr = float(Config.get(example_name, 'disc_lr'))
    gen_lr = float(Config.get(example_name, 'gen_lr'))
    lambda0 = float(Config.get(example_name, 'lambda0'))
    lambda1 = float(Config.get(example_name, 'lambda1'))
    save_interval = int(Config.get(example_name, 'save_interval'))
    
    return noise_dim, train_steps, batch_size, disc_lr, gen_lr, lambda0, lambda1, save_interval