#!/usr/bin/python3
 
"""main.py: Main file where everything comes together. """

import mnist_loader
import network
import functions
import numpy as np

data = mnist_loader.MNISTLoader()
data.load()

relu = functions.ActivationReLU()
sigmoid = functions.ActivationSigmoid()

network = network.Network(
    layers=[784, 100, 10],
    sigmas=[relu, relu, sigmoid],
    C_prime=functions.cost_ce_prime)

network.SGD_learn(
    training_data=data.training_data,
    test_data=data.test_data,
    batch_size=10,
    rate=0.2,
    epochs=30)