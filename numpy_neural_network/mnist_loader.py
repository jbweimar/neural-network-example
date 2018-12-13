#!/usr/bin/python3

"""mnist_loader.py: Helper class to load the MNIST database."""

import gzip
import pickle
import numpy as np
import random
 
class MNISTLoader:
 
    def load(self):
        f = gzip.open('../data/mnist.pkl.gz', 'rb')
        training_data, v_data, test_data = pickle.load(f, encoding='latin1')
        self.training_data = self._one_hot_encode_ys(training_data)
        self.test_data = list(zip(test_data[0], test_data[1]))

    def _one_hot_encode_ys(self, data):
        return [data[0], [self._one_hot_encode(y) for y in data[1]]]

    def _one_hot_encode(self, y):
        e = np.zeros(10)
        e[y] = 1.0
        return e

