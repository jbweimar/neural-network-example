#!/usr/bin/python3
 
"""network.py: A feedforward neural network."""
 
import mnist_loader
import numpy as np
import random
 
class Network:
 
    def __init__(self,
        layers,
        sigmas,
        C_prime):
        self.layers = layers
        self.sigmas = sigmas
        self.C_prime = C_prime
        self.init_weights_and_biases()
 
    def init_weights_and_biases(self):
        self.Bs = [np.random.randn(l) / np.sqrt(l) for l in self.layers[1:]]
        dims = zip(self.layers[1:], self.layers[:-1])
        self.Ws = [np.random.randn(v, w) / np.sqrt(w) for v, w in dims]
 
    def _get_batches(self, data, size):
        indices = np.random.permutation(len(data[0]))
        xs = [data[0][i] for i in indices]
        ys = [data[1][i] for i in indices]
        return [(np.vstack(xs[k : k+size]), np.vstack(ys[k : k+size])) 
            for k in range(0, len(data[0]), size)]
 
    def _get_Zs_As(self, X):
        As = [X]
        Zs = []
        A = X
        l = 0
        rows = np.size(X, 0)
        for b, W in zip(self.Bs, self.Ws):
            B = np.tile(b, (rows, 1))
            Z = A @ W.transpose() + B # (FF1)
            A = self.sigmas[l].f(Z) # (FF2)
            As.append(A)
            Zs.append(Z)
            l -= 1
        return Zs, As
 
    def SGD_learn(self,
        training_data,
        test_data,
        batch_size,
        rate,
        epochs):
        self.init_weights_and_biases() # 1.
        for e in range(0, epochs): # 2.
            batches = self._get_batches(training_data, batch_size) # 2.1
            for X, Y in batches: # 2.2 & 2.2.1
                dBs, dWs = self.backprop(X, Y)
                self.Bs -= rate * dBs / batch_size
                self.Ws -= rate * dWs / batch_size
                
    def backprop(self, X, Y):
        Zs, As = self._get_Zs_As(X) # 2.2.2
        batch_size = np.size(X, 0)
        sp = self.sigmas[-1].f_prime(Zs[-1])
        Delta = self.C_prime(As[-1], Y) * sp # 2.2.3 (E3)
        dBs = np.empty_like(self.Bs) # 2.2.4
        dWs = np.empty_like(self.Ws)
        dBs[-1] = np.ones(batch_size).T @ Delta # (E1)
        dWs[-1] = Delta.T @ As[-2] # (E2)
        print(np.shape(dWs[-1]))
        for l in range(2, len(self.layers)): # 2.2.5
            sp = self.sigmas[-l].f_prime(Zs[-l])
            Delta = Delta @ self.Ws[-l+1] * sp # 2.2.5.1 (E3)
            dBs[-l] = np.ones(batch_size).T @ Delta # (E1)
            dWs[-l] = Delta.T @ As[-l-1].T # (E2)
            print(np.shape(As[-l-1].T))
            print(np.shape(Delta))
        return dBs, dWs