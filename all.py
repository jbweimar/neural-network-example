import pickle
import gzip
import numpy as np
import random

class MNISTLoader:

    def load(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        tr, va, te = pickle.load(f, encoding='latin1')
        self.training_data = self._correct_data(tr)
        self.test_data = self._correct_data(te)

    def _correct_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return [(x, self._one_hot(y)) for x, y in zip(data_x, data_y)]

    def _one_hot(self, y):
        e = np.zeros(10)
        e[y] = 1.0
        return e

data = MNISTLoader()
data.load()

digits = [np.argmax(digit[1]) for digit in data.training_data]
bincounts = np.bincount(digits)
print(bincounts)

import matplotlib.pyplot as plt
digits = [np.reshape(digit[0], (28, 28)) for digit in data.training_data[0:36]]
fig = plt.figure()
for i in range(0, 36):
    fig.add_subplot(6, 6, i + 1)
    plt.gray()
    plt.imshow(digits[i])
plt.show()

class Network: 

    def __init__(self, sizes):
        self.sizes = sizes
        dimensions = zip(self.sizes[1:], self.sizes[:-1])
        self.biases = [np.random.randn(x) for x in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in dimensions]

    def _feedforward(self, x):
        a = x
        for b, w in zip(self.biases, self.weights):
            a = self._sigmoid(w @ a + b)
        return a


    def _get_batches(self, data, size):
        random.shuffle(data)
        return [data[k : k+size] for k in range(0, len(data), size)]

    def SGD(self, data, batch_size, test_data):
        for batch in self._get_batches(data, batch_size):
            self._update_weights_biases(batch)
        print(self.get_percentage_correct(test_data))

    def _update_weights_biases(self, batch):
        dbs = [np.zeros(b.shape) for b in self.biases]
        dws = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in batch:
            db, dw = self._backprop(x, y)
            dbs = [a + b for a, b in zip(dbs, db)]
            dws = [a + b for a, b in zip(dws, dw)]
        zipped_b = zip(self.biases, dbs)
        zipped_w = zip(self.weights, dws)
        self.biases = [b - (3.0/len(batch)) * db for b, db in zipped_b]
        self.weights = [b - (3.0/len(batch)) * db for b, db in zipped_w]

    def _backprop(self, x, y):
        activations, zs = self._calculate_seeds(x)
        sp = self._sigmoid_prime(zs[-1])
        delta = self._cost_derivative(activations[-1], y) * sp
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.biases]
        db[-1] = delta
        dw[-1] = np.outer(delta, activations[-2].transpose())
        for k in range(2, len(self.sizes)):
            sp = self._sigmoid_prime(zs[-k])
            delta = (self.weights[-k + 1].transpose() @ delta) * sp
            db[-k] = delta
            dw[-k] = np.outer(delta, activations[-k - 1].transpose())
        return db, dw

    def _calculate_seeds(self, x):
        activations = [x]
        zs = []
        a = x
        for b, w in zip(self.biases, self.weights):
            z = w @ a + b
            zs.append(z)
            a = self._sigmoid(z)
            activations.append(a)
        return activations, zs

    def get_percentage_correct(self, test_data):
        result = [(np.argmax(self._feedforward(x)), np.argmax(y))\
        for x, y in test_data]
        nr_correct = sum(int(x == y) for x, y in result)
        return '{0:.2f}%'.format(100 * nr_correct / len(test_data))

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _cost_derivative(self, a, y):
        return a - y

network = Network([784, 30, 10])
network.SGD(
    data=data.training_data,
    batch_size=10,
    test_data=data.test_data)

