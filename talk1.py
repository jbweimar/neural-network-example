import urllib.request as urllib
import pickle 
import gzip
import numpy as np
import random

class MNISTData:
    
    def load(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        training_data, \
        validation_data, \
        test_data = pickle.load(f, encoding='latin1')
        training_data_x = list(training_data[0])
        training_data_y = list(training_data[1])
        zipped_training_data = zip(training_data_x, training_data_y)
        self.training_data = [(dx, dy) for dx, dy in zipped_training_data]
        test_data_x = list(test_data[0])
        test_data_y = list(test_data[1])
        zipped_test_data = zip(test_data_x, test_data_y)
        self.test_data = [(dx, dy) for dx, dy in zipped_test_data]
    
data = MNISTData()
data.load()

class Network:
        
    def __init__(self, sizes):
        self.sizes = sizes

    def initialize_weights(self):
        b_counts = self.sizes[1:]
        w_matrix_dims = zip(self.sizes[:-1], self.sizes[1:])
        self.b_vectors = [np.random.randn(v) for v in b_counts]
        self.w_matrices = [np.random.randn(w, v) for v, w in w_matrix_dims]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, x):
        a = x
        for b, w in zip(self.b_vectors, self.w_matrices):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def get_batches(self, size, data):
        np.random.shuffle(data)
        return [data[k:k+size] for k in range(0, len(data), size)]


    def SGD(self, epochs, data, batch_size, eta, cost_derivative, 
            test_data=None):
        self.initialize_weights()
        for j in range(epochs):
            for batch in self.get_batches(batch_size, data):
                self.update_b_w(batch, eta, cost_derivative)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(
                    j, self.evaluate(test_data), len(test_data)))
            else:
                print('Epoch {0} complete'.format(i))

    def update_b_w(self, batch, eta, cost_derivative):
        batch_db = [np.zeros(b.shape) for b in self.b_vectors]
        batch_dw = [np.zeros(w.shape) for w in self.w_matrices]
        for x, y in batch:
            single_db, single_dw = self.backpropagate(x, y, cost_derivative)
            batch_db = [bdb+sdb for bdb, sdb in zip(batch_db, single_db)]
            batch_dw = [bdw+sdw for bdw, sdw in zip(batch_dw, single_dw)]
        zipped_b = zip(self.b_vectors, batch_db)
        zipped_w = zip(self.w_matrices, batch_dw)
        self.b_vectors = [b - (eta*db/len(batch)) for b, db in zipped_b]
        self.w_matrices = [w - (eta*dw/len(batch)) for w, dw in zipped_w]

    def backpropagate(self, x, y, cost_derivative):
        db = [np.zeros(b.shape) for b in self.b_vectors]
        dw = [np.zeros(w.shape) for w in self.w_matrices]
        activations, zs = self.calculate_activations(x)
        zs_prime = self.sigmoid_prime(zs[-1])
        delta = cost_derivative(activations[-1], y) * zs_prime
        db[-1] = delta
        dw[-1] = np.outer(delta, activations[-2].transpose())
        for l in range(2, len(self.sizes)):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_matrices[-l + 1].transpose(), delta) * sp
            db[-l] = delta 
            dw[-l] = np.outer(delta, activations[-l - 1].transpose())
        return db, dw

    def calculate_activations(self, x):
        a = x
        activations = [x]
        zs = []
        for b, w in zip(self.b_vectors, self.w_matrices):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        return activations, zs

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) 
            for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

def MSE_derivative(output_a, y):
    return (output_a - y)

network = Network(sizes=[784, 30, 10])
network.SGD(
    epochs=30, 
    data=data.training_data, 
    batch_size=10,
    eta=3.0,
    cost_derivative=MSE_derivative,
    test_data=data.test_data)











