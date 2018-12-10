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
        self.bs = [np.random.randn(l) / np.sqrt(l) for l in self.layers[1:]]
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
        for b, W in zip(self.bs, self.Ws):
            B = np.tile(b, (rows, 1))
            Z = A @ W.transpose() + B # (FF1)
            A = self.sigmas[l].f(Z) # (FF2)
            As.append(A)
            Zs.append(Z)
            l -= 1
        return Zs, As
 
    def _feedforward(self, x):
        a = x
        l = 0
        for b, W in zip(self.bs, self.Ws):
            a = self.sigmas[l].f(W @ a + b)
            l += 1
        return a

    def display_accuracy(self, epoch, data):
        results_x = [np.argmax(_feedforward(x)) for x in data[0]]
        results_y = [np.argmax(y) for y in data[1]]
        correct = sum(int(results_x == results_y))
        print("")


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
                self.bs -= rate * dBs / batch_size # 2.2.6
                self.Ws -= rate * dWs / batch_size
                if test_data != None: 
                    display_accuracy(test_data) # 2.3
                
    def backprop(self, X, Y):
        Zs, As = self._get_Zs_As(X) # 2.2.2
        batch_size = np.size(X, 0)
        sp = self.sigmas[-1].f_prime(Zs[-1])
        Delta = self.C_prime(As[-1], Y) * sp # 2.2.3 (M3)
        dBs = np.empty_like(self.bs) # 2.2.4
        dWs = np.empty_like(self.Ws)
        dBs[-1] = np.ones(batch_size).T @ Delta # (M1)
        dWs[-1] = Delta.T @ As[-2] # (M2)
        print(np.shape(dWs[-1]))
        for l in range(2, len(self.layers)): # 2.2.5
            sp = self.sigmas[-l].f_prime(Zs[-l])
            Delta = Delta @ self.Ws[-l+1] * sp # 2.2.5.1 (M3)
            dBs[-l] = np.ones(batch_size).T @ Delta # 2.2.5.2 (M1)
            dWs[-l] = Delta.T @ As[-l-1].T # (M2)
        return dBs, dWs