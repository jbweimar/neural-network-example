import matplotlib.pyplot as plt
import mnist_loader
import numpy as np

data = mnist_loader.MNISTLoader()
data.load()

digits = data.training_data[0][0:9] # Grab the first 16 digits.
 
image_data = [np.reshape(digit, (28, 28)) for digit in digits]
fig = plt.figure()
for i in range(0, 9):
    fig.add_subplot(3, 3, i + 1)
    plt.imshow(image_data[i])
    plt.gray()
plt.show()