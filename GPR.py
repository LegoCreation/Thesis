#!/usr/bin/env python3
# coding: utf-8

import os
import sys
thread = sys.argv[1]
os.environ["OMP_NUM_THREADS"] = str(thread) # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = str(thread) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(thread) # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(thread) # export NUMEXPR_NUM_THREADS=6
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib.pyplot as plt

import gpflow
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, WhiteKernel
import GPy
import gpflow
import pyro
import torch
import pyro.contrib.gp as pyrogp
import pyro.distributions as dist
import tensorflow as tf
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.util import torch_backward, torch_item
from pyro.contrib.autoguide import AutoMultivariateNormal
from torch.distributions import constraints
from torch.nn import Parameter
from pyro.optim import Adam


# In[112]:


X = np.load("./data/X.npy")
y = np.loadtxt("./data/E_def2-tzvp.dat")


# In[202]:

class GPR_benchmark():
    def __init__(self, avg_mae, avg_training_time_cpu, avg_testing_time_cpu, 
                 avg_training_time_wall, avg_testing_time_wall, library, kernel, 
                 sample_size, sigma, length_scale, white_noise, optimize=False, 
                 iterations=100, jitter=1e-6, seed = 42):
        self.avg_mae = avg_mae
        self.avg_training_time_cpu = avg_training_time_cpu
        self.avg_testing_time_cpu = avg_testing_time_cpu
        self.avg_training_time_wall = avg_training_time_wall
        self.avg_testing_time_wall = avg_testing_time_wall
        self.library = library
        self.kernel = kernel
        self.sample_size = sample_size
        self.sigma = sigma
        self.length_scale = length_scale
        self.white_noise = white_noise
        self.optimize = optimize
        self.iterations = iterations
        self.jitter = jitter
        self.seed = seed


class GPR:
    def __init__(self, library='scikit', kernel='gaussian', sigma = 1.0, length_scale=1.0, white_noise=0.1, optimize = False, iterations = 100, jitter = 1e-6, seed = 42):
        self.jitter = jitter
        self.sigma = sigma
        self.library = library
        self.kernel = kernel
        self.length_scale = length_scale
        self.white_noise = white_noise
        self.iterations = iterations
        self.optimize = optimize
        self.is_fit = False
        self.avg_mae = 0
        self.avg_training_time_cpu = 0
        self.avg_testing_time_cpu = 0
        self.avg_training_time_wall = 0
        self.avg_testing_time_wall = 0
        self.seed = seed
    def fit(self, X, y):
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        self.X = X
        self.y = y
        start_time = time.time()
        if self.library == 'scikit':
            kernel_scikit = None
            if self.kernel == 'gaussian':
                kernel_scikit = self.sigma * RBF(length_scale=self.length_scale) + WhiteKernel(
                    noise_level=self.white_noise, noise_level_bounds=(1e-6, 1e1))
            elif self.kernel == 'laplacian':
                kernel_scikit = self.sigma * RationalQuadratic(length_scale=self.length_scale,
                                                          alpha=1.0) + WhiteKernel(noise_level=self.white_noise,
                                                                                   noise_level_bounds=(1e-6, 1e1))
            if self.optimize:
                self.model = GaussianProcessRegressor(kernel=kernel_scikit,
                                                      alpha = self.jitter,
                                                      n_restarts_optimizer = self.iterations)
            else:
                self.model = GaussianProcessRegressor(kernel=kernel_scikit,
                                                      alpha = self.jitter,
                                                      optimizer = None, n_restarts_optimizer = 0)
            self.model.fit(X, y)
        elif self.library == 'gpy':
            kernel_gpy = None
            y = np.expand_dims(y, axis=1)
            if self.kernel == 'gaussian':
                kernel_gpy = GPy.kern.RBF(input_dim=X.shape[1],
                                          variance=self.sigma,
                                          lengthscale=self.length_scale, ARD = True)
            elif self.kernel == 'laplacian':
                kernel_gpy = GPy.kern.Exponential(input_dim=X.shape[1],
                                                  variance=self.sigma,
                                                  lengthscale=self.length_scale)

            self.model = GPy.models.GPRegression(X, y, kernel_gpy, noise_var=self.white_noise)
            if self.optimize:
                self.model.optimize(max_iters=self.iterations)
            
        elif self.library == 'gpflow':
            kernel_gpflow = None
            y = np.expand_dims(y, axis=1)
            if self.kernel == 'gaussian':
                kernel_gpflow = gpflow.kernels.SquaredExponential(variance=self.sigma,
                                                                  lengthscales=self.length_scale)
            elif self.kernel == 'laplacian':
                kernel_gpflow = gpflow.kernels.Exponential(variance=self.sigma,
                                                           lengthscales=self.length_scale * 2)
            self.model = gpflow.models.GPR(data=(X, y), kernel=kernel_gpflow,
                                           likelihood = gpflow.likelihoods.Gaussian(variance= self.sigma))
            if self.optimize:
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(self.model.training_loss,
                                   self.model.trainable_variables,
                                   options=dict(maxiter=self.iterations))


        elif self.library == 'pyro':
            self.tensorX = torch.from_numpy(X).to(torch.device('cpu'))
            self.tensory = torch.from_numpy(y).to(torch.device('cpu'))
            kernel_pyro = None
            if self.kernel == 'gaussian':
                kernel_pyro = pyrogp.kernels.RBF(input_dim=X.shape[1],
                                                 variance=torch.tensor(self.sigma).to(torch.device('cpu')),
                                                 lengthscale=torch.tensor(self.length_scale).to(torch.device('cpu')))
            elif self.kernel == 'laplacian':
                kernel_pyro = pyrogp.kernels.Exponential(input_dim=X.shape[1],
                                                         variance=torch.tensor(self.sigma).to(torch.device('cpu')),
                                                         lengthscale=torch.tensor(self.length_scale).to(torch.device('cpu')))
            self.model = pyrogp.models.GPRegression(self.tensorX,
                                                    self.tensory,
                                                    kernel_pyro,
                                                    noise=self.tensorX.new_tensor(self.white_noise).to(torch.device('cpu')),
                                                    jitter=self.jitter)
            if self.optimize:
                print("Pyro Optimizer On")
                pyrogp.util.train(self.model, optimizer=None,
                                  loss_fn=None, retain_graph=None,
                                  num_steps=self.iterations)

                

        end_cpu = time.process_time()
        end_wall = time.perf_counter()
        self.training_time_wall = end_wall - start_wall
        self.training_time_cpu = end_cpu - start_cpu
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("Fit model first")
        start_time = time.time()
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        if self.library == 'scikit':
            y_pred = self.model.predict(X, return_std=False)
        elif self.library == 'gpy':
            y_pred, _ = self.model.predict(X)
        elif self.library == 'gpflow':
            y_pred, _ = self.model.predict_f(X)
        elif self.library == 'pyro':
            with torch.no_grad():
                y_pred, _ = self.model.forward(torch.from_numpy(X).to(torch.device('cpu')))
                y_pred = y_pred.detach().numpy()
#         elif self.library == 'pymc3':
#             y_pred = self.model.predict(X)
        end_cpu = time.process_time()
        end_wall = time.perf_counter()

        self.cpu_time = end_cpu - start_cpu
        self.testing_time_wall = end_wall - start_wall
        self.testing_time_cpu = end_cpu - start_cpu
        return y_pred
    
    def compute_mae(self, X, y):
        y_predicted = self.predict(X)
        self.mae = np.mean(np.abs(y_predicted - y))
        return self.mae



sigma = 1 #output variance scale
length_scale = 200 #length_scale
white_noise = 1e-6 #likelihood noise
optimize = False
iterations = 0
jitter = 1e-6 #can only be specified in scikit and pyro
gpr_array = np.empty((13, 4, 2), dtype=object) 
for l, kernel in enumerate(["laplacian", "gaussian"]):
    for k, lib in enumerate(["scikit", "gpy", "gpflow", "pyro" ]):
        mae_array = np.zeros((12,))
        training_wall_array = np.zeros((12,))
        testing_wall_array = np.zeros((12,))
        training_cpu_array = np.zeros((12,))
        testing_cpu_array = np.zeros((12,))
        for j in range(1, 11):
            for i in range(12):
                N = pow(2, i+1)
                if N > 10000:
                    break
                X_copy = X.copy()
                y_copy = y.copy()
                np.random.seed(j)
                np.random.shuffle(X_copy)
                np.random.seed(j)
                np.random.shuffle(y_copy)
                X_train = X_copy[:N]
                y_train = y_copy[:N]
                avg_mae = 0
                X_test = X_copy[9000:]
                y_test = y_copy[9000:]
                gpr_ins = GPR(library=lib,
                              kernel=kernel, 
                              sigma = sigma, 
                              length_scale=length_scale, 
                              white_noise=white_noise, 
                              optimize = optimize,  
                              iterations = iterations,
                              jitter = jitter,
                              seed = j)
                gpr_ins.fit(X_train, y_train)
                training_wall_array[i] += gpr_ins.training_time_wall
                training_cpu_array[i] += gpr_ins.training_time_cpu
                

                X_test_n = X_copy[:N]
                gpr_ins.predict(X_test_n)
                testing_wall_array[i] += gpr_ins.testing_time_wall
                testing_cpu_array[i] += gpr_ins.testing_time_cpu
                
                gpr_ins.compute_mae(X_test, y_test)
                mae_array[i] += gpr_ins.mae
                gpr_ins = None

                if j == 10:
                    gpr_array[i, k, l] = GPR_benchmark(avg_mae = mae_array[i] /10,
                                                       avg_training_time_wall = training_wall_array[i] /10, 
                                                       avg_testing_time_wall = testing_wall_array[i] /10,
                                                       avg_training_time_cpu = training_cpu_array[i] /10,
                                                       avg_testing_time_cpu = testing_cpu_array[i] /10,
                                                       library = lib,
                                                       kernel = kernel,
                                                       sample_size = N,
                                                       sigma = sigma,
                                                       length_scale = length_scale,
                                                       white_noise = white_noise,
                                                       optimize = optimize,
                                                       iterations = iterations,
                                                       jitter = jitter,
                                                       seed = 0)
                    
for l, kernel in enumerate(["laplacian", "gaussian"]):
    for k, lib in enumerate(["scikit", "gpy", "gpflow", "pyro" ]):
        mae_array = np.zeros((13,))
        training_wall_array = np.zeros((13,))
        testing_wall_array = np.zeros((13,))
        training_cpu_array = np.zeros((13,))
        testing_cpu_array = np.zeros((13,))
        for j in range(1, 11):
            for i in range(12, 13):
                N = pow(2, i+1)
                if N > 10000:
                    break
                X_copy = X.copy()
                y_copy = y.copy()
                np.random.seed(j)
                np.random.shuffle(X_copy)
                np.random.seed(j)
                np.random.shuffle(y_copy)
                X_train = X_copy[:N]
                y_train = y_copy[:N]
                avg_mae = 0
                X_test = X_copy[9000:] #Taking remaining 1000 samples
                y_test = y_copy[9000:]
                gpr_ins = GPR(library=lib,
                              kernel=kernel, 
                              sigma = sigma, 
                              length_scale=length_scale, 
                              white_noise=white_noise, 
                              optimize = optimize,  
                              iterations = iterations,
                              jitter = jitter,
                              seed = j)
                gpr_ins.fit(X_train, y_train)
                training_wall_array[i] += gpr_ins.training_time_wall
                training_cpu_array[i] += gpr_ins.training_time_cpu
                
                
                #Strictly for evaluation time comparisions. Bad for MAE calculations as training data is used for testing
                X_test_time = X_copy[:N]
                gpr_ins.predict(X_test_time)
                testing_wall_array[i] += gpr_ins.testing_time_wall
                testing_cpu_array[i] += gpr_ins.testing_time_cpu
                
                gpr_ins.compute_mae(X_test, y_test)
                mae_array[i] += gpr_ins.mae
                gpr_ins = None

                if j == 10:
                    gpr_array[i, k, l] = GPR_benchmark(avg_mae = mae_array[i] /10,
                                                       avg_training_time_wall = training_wall_array[i] /10, 
                                                       avg_testing_time_wall = testing_wall_array[i] /10,
                                                       avg_training_time_cpu = training_cpu_array[i] /10,
                                                       avg_testing_time_cpu = testing_cpu_array[i] /10,
                                                       library = lib,
                                                       kernel = kernel,
                                                       sample_size = N,
                                                       sigma = sigma,
                                                       length_scale = length_scale,
                                                       white_noise = white_noise,
                                                       optimize = optimize,
                                                       iterations = iterations,
                                                       jitter = jitter,
                                                       seed = 0)

outfile = "/home/ssunar/Thesis/data/output_data_new/GPR_thread_" + str(thread) + ".npy"
np.save(outfile, gpr_array)






