#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
thread = sys.argv[1]
os.environ["OMP_NUM_THREADS"] = str(thread) # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = str(thread) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(thread) # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(thread) # export NUMEXPR_NUM_THREADS=6
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import qml
from qml.math import cho_solve
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel


# In[2]:


X = np.load("./data/X.npy")
y = np.loadtxt("./data/E_def2-tzvp.dat")


class KRR_benchmark():
    def __init__(self, avg_mae, avg_training_time_cpu, avg_testing_time_cpu, 
                 avg_training_time_wall, avg_testing_time_wall, library, kernel, 
                 sample_size, sigma, alpha, seed = 42):
        self.avg_mae = avg_mae
        self.avg_training_time_cpu = avg_training_time_cpu
        self.avg_testing_time_cpu = avg_testing_time_cpu
        self.avg_training_time_wall = avg_training_time_wall
        self.avg_testing_time_wall = avg_testing_time_wall
        self.library = library
        self.kernel = kernel
        self.sample_size = sample_size
        self.sigma = sigma
        self.alpha = alpha
        self.seed = seed

# In[3]:


class KRR():
    def __init__(self, alpha=1e-8, library = "qml", sigma = 720, kernel='gaussian', seed = 42):
        self.alpha = alpha
        self.sigma = sigma
        self.kernel = kernel
        self.library = library
        self.is_fit = False
        self.avg_mae = 0
        self.avg_training_time_cpu = 0
        self.avg_testing_time_cpu = 0
        self.avg_training_time_wall = 0
        self.avg_testing_time_wall = 0
        self.seed = seed

    def fit(self, X, y):
        np.random.seed(self.seed)
        self.X_ = X
        self.y_ = y
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        if self.library == "qml":
            if self.kernel == 'gaussian':
                K = gaussian_kernel(X, X, self.sigma)    
            elif self.kernel == 'laplacian':
                K = laplacian_kernel(X, X, self.sigma)
            K[np.diag_indices_from(K)] += self.alpha 
            self.beta = cho_solve(K, y)
        elif self.library == "scikit-learn":
            if self.kernel == 'gaussian':
                self.krr = KernelRidge(kernel = "rbf", gamma = 1/self.sigma, alpha=self.alpha)
            elif self.kernel == 'laplacian':
                self.krr = KernelRidge(kernel = "laplacian", gamma = 1/self.sigma, alpha=self.alpha)
            self.krr.fit(X, y)
        else:
            print("Error library")
        end_cpu = time.process_time()
        end_wall = time.perf_counter()
        self.training_time_wall = end_wall - start_wall
        self.training_time_cpu = end_cpu - start_cpu
        
        self.is_fit = True


    def predict(self, X):
        if not self.is_fit:
            print("Fit model first")
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        if self.library == "qml":
            if self.kernel == 'gaussian':
                Ks = gaussian_kernel(X, self.X_, self.sigma)  
            elif self.kernel == 'laplacian':
                Ks = laplacian_kernel(X, self.X_, self.sigma)
            y_predicted = np.dot(Ks, self.beta)
        elif self.library == "scikit-learn":
             y_predicted = self.krr.predict(X)
        end_cpu = time.process_time()
        end_wall = time.perf_counter()

        self.cpu_time = end_cpu - start_cpu
        self.testing_time_wall = end_wall - start_wall
        self.testing_time_cpu = end_cpu - start_cpu
        
        return y_predicted
    
    def compute_mae(self, X, y):
        y_predicted = self.predict(X)
        self.mae = np.mean(np.abs(y_predicted - y))
        return self.mae


# In[4]:




sigma = 200
alpha = 1e-6

krr_array = np.empty((13, 2, 2), dtype=object) 
for l, kernel in enumerate(["laplacian", "gaussian"]):
    for k, lib in enumerate(["qml", "scikit-learn"]):
        mae_array = np.zeros((13,))
        training_wall_array = np.zeros((13,))
        testing_wall_array = np.zeros((13,))
        training_cpu_array = np.zeros((13,))
        testing_cpu_array = np.zeros((13,))
        for j in range(1, 11):
            for i in range(20):
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
                X_test = X_copy[9000:]#Taking remaining 1000 samples
                y_test = y_copy[9000:]
                krr_ins = KRR(alpha=alpha, library = lib, sigma = sigma, kernel=kernel, seed = j)
                krr_ins.fit(X_train, y_train)
                training_wall_array[i] += krr_ins.training_time_wall
                training_cpu_array[i] += krr_ins.training_time_cpu
                
                #Strictly for evaluation time comparisions. Bad for MAE calculations as training data is used for testing
                X_test_time = X_copy[:N] 
                krr_ins.predict(X_test_time)
                testing_wall_array[i] += krr_ins.testing_time_wall
                testing_cpu_array[i] += krr_ins.testing_time_cpu
                
                krr_ins.compute_mae(X_test, y_test)
                mae_array[i] += krr_ins.mae

                if j == 10:
                    krr_array[i, k, l] = KRR_benchmark(avg_mae = mae_array[i] /10,
                                                       avg_training_time_wall = training_wall_array[i] /10, 
                                                       avg_testing_time_wall = testing_wall_array[i] /10,
                                                       avg_training_time_cpu = training_cpu_array[i] /10,
                                                       avg_testing_time_cpu = testing_cpu_array[i] /10,
                                                       library = lib,
                                                       kernel = kernel,
                                                       sample_size = N,
                                                       sigma = sigma,
                                                       alpha = alpha,
                                                       seed = 0)


# In[5]:


outfile = "/home/ssunar/Thesis/data/output_data_new/KRR_thread_" + str(thread) + ".npy"
np.save(outfile, krr_array)






