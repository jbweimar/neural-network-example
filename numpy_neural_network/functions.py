#!/usr/bin/python3
 
"""functions.py: Various cost and activation functions """

import numpy as np

def cost_ce_prime(A, Y): # Cross Entropy
    return A - Y

def cost_mse(A, Y): # Mean Squared Error
    return (A - Y) / A * (1 - A)

class ActivationReLU:
    
    def f(self, Z): 
        return Z * (Z > 0)

    def f_prime(self, Z):
        return (Z > 0).astype(float)

class ActivationSigmoid:

    def f(self, Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def f_prime(self, Z):
        return self.f(Z) * (1 - self.f(Z))
