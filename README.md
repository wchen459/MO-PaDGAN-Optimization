# MO-PaDGAN for Design Reparameterization and Optimization

Experiment code associated with our paper on _Applied Soft Computing_: [MO-PaDGAN: Reparameterizing Engineering Designs for Augmented Multi-objective Optimization](https://www.sciencedirect.com/science/article/pii/S1568494621008310?casa_token=sSPX_i-oQsAAAAAA:ZiYmTvp4YjpRMm1PTbTGlBZ6UwizMdUf5hB37TWyUTItQlnnKr-dENYvdkVBslrUh7tBmHxrnS8).

![Alt text](/architecture.svg)

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

   Chen, W., & Ahmed, F. (2021). MO-PaDGAN: Reparameterizing Engineering Designs for augmented multi-objective optimization. _Applied Soft Computing_, 113, 107909.

	@article{chen2021mo,
	     title={MO-PaDGAN: Reparameterizing Engineering Designs for augmented multi-objective optimization},
	     author={Chen, Wei and Ahmed, Faez},
	     journal={Applied Soft Computing},
	     volume={113},
	     pages={107909},
	     year={2021},
	     publisher={Elsevier}
	   }

## Required packages

- tensorflow < 2.0.0
- gpflow
- gpflowopt
- sklearn
- numpy
- matplotlib
- seaborn
- pexpect

## Usage

### Synthetic examples

1. Go to example directory:

   ```bash
   cd synthetic
   ```

2. Train MO-PaDGAN:

   ```bash
   python train.py
   ```

   positional arguments:
    
   ```
   mode	train or evaluate
   data	dataset name (specified in datasets.py; available datasets are Ring2D, Grid2D, Donut2D, ThinDonut2D, and Arc2D)
   func	function name (specified in functions.py; available functions are VLMOP2 and NKNO1)
   ```

   optional arguments:

   ```
   -h, --help            	show this help message and exit
   --lambda0		coefficient controlling the weight of quality in the DPP kernel
   --lambda1		coefficient controlling the weight of the performance augmented DPP loss in the PaDGAN loss
   --disc_lr		learning rate for the discriminator
   --gen_lr		learning rate for the generator
   --batch_size		batch size
   --train_steps		training steps
   --save_interval 	number of intervals for saving the trained model and plotting results
   ```

   The default values of the optional arguments will be read from the file `synthetic/config.ini`.

   The trained model and the result plots will be saved under the directory `synthetic/trained_gan/<data>_<func>/<lambda0>_<lambda1>/`, where `<data>`, `<func>`, `<lambda0>`, and `<lambda1>` are specified in the arguments or in `synthetic/config.ini`. 
   
   Note that we can set `lambda0` and `lambda1` to zeros to train a vanilla GAN.
   
   Datasets and functions are defined in `synthetic/functions.py` and `synthetic/datasets.py`, respectively.
   
   Specifically, here are the dataset and function names (`<data>` and `<func>`) for the two synthetic examples in the paper:
   
   | Example | Dataset name | Function name |
   |---------|--------------|---------------|
   | I       | Ring2D       | NKNO1         |
   | II      | Ring2D       | VLMOP2        |
   
3. Multi-objective optimization by Bayesian optimization:

   i) Run single experiment:
   
      ```bash
      python optimize.py
      ```

      positional arguments:
    
      ```
      data	dataset name (specified in datasets.py; available datasets are Ring2D, Grid2D, Donut2D, ThinDonut2D, and Arc2D)
      func	function name (specified in functions.py; available functions are Linear, MixGrid, MixRing, MixRing4, and MixRing6)
      ```

      optional arguments:

      ```
      -h, --help            	show this help message and exit
      --lambda0		coefficient controlling the weight of quality in the DPP kernel
      --lambda1		coefficient controlling the weight of the performance augmented DPP loss in the PaDGAN loss
      --id 	experiment ID
      ```
      
   ii) Run experments in batch mode to reproduce results from the paper:
   
      ```bash
      python run_batch_experiments.py
      ```

### Airfoil design example

1. Install [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/).

2. Go to example directory:

   ```bash
   cd airfoil
   ```

3. Download the airfoil dataset [here](https://drive.google.com/file/d/1OZfF4Zl31jzJmucBIlSqO4OKq9CKHh4r/view?usp=sharing) and extract the NPY files into `airfoil/data/`.

4. Go to the surrogate model directory:

   ```bash
   cd surrogate
   ```

5. Train a surrogate model to predict airfoil performances:

   ```bash
   python train_surrogate.py train
   ```

   positional arguments:
    
   ```
   mode	train or evaluate
   ```

   optional arguments:

   ```
   -h, --help            	show this help message and exit
   --save_interval		interval for saving checkpoints
   ```

6. Go back to example directory:

   ```bash
   cd ..
   ```

   Train MO-PaDGAN:

   ```bash
   python run_experiment.py train
   ```

   positional arguments:
    
   ```
   mode	train or evaluate
   ```

   optional arguments:

   ```
   -h, --help            	show this help message and exit
   --lambda0		coefficient controlling the weight of quality in the DPP kernel
   --lambda1		coefficient controlling the weight of the performance augmented DPP loss in the PaDGAN loss
   ```

   The default values of the optional arguments will be read from the file `airfoil/config.ini`.

   The trained model and the result plots will be saved under the directory `airfoil/trained_gan/<lambda0>_<lambda1>/`, where `<lambda0>` and `<lambda1>` are specified in the arguments or in `airfoil/config.ini`. Note that we can set `lambda0` and `lambda1` to zeros to train a vanilla GAN.
   
7. Multi-objective optimization by Bayesian optimization:

   i) Run single experiment:
   
      ```bash
      python optimize_bo.py
      ```

      positional arguments:
    
      ```
      parameterization		airfoil parameterization (GAN, SVD, or FFD)
      ```

      optional arguments:

      ```
      -h, --help            	show this help message and exit
      --lambda0		coefficient controlling the weight of quality in the DPP kernel
      --lambda1		coefficient controlling the weight of the performance augmented DPP loss in the PaDGAN loss
      --id 	experiment ID
      ```
      
   ii) Run experments in batch mode to reproduce results from the paper:
   
      ```bash
      python run_batch_experiments_bo.py
      ```
      
8. Multi-objective optimization by Evolutionary Algorithm:

   i) Run single experiment:
   
      ```bash
      python optimize_ea.py
      ```

      positional arguments:
    
      ```
      parameterization		airfoil parameterization (GAN, SVD, or FFD)
      ```

      optional arguments:

      ```
      -h, --help            	show this help message and exit
      --lambda0		coefficient controlling the weight of quality in the DPP kernel
      --lambda1		coefficient controlling the weight of the performance augmented DPP loss in the PaDGAN loss
      --id 	experiment ID
      ```
      
   ii) Run experments in batch mode to reproduce results from the paper:
   
      ```bash
      python run_batch_experiments_ea.py
      ```
