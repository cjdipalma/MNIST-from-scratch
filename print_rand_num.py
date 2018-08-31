# Prints a random number from the MNIST training dataset to the console.

from mnist import MNIST
import random

mndata = MNIST()

images, labels = mndata.load_training()

randindex = random.randrange(0, len(images))
print(mndata.display(images[randindex]))
print(labels[randindex])
##print("mess of raw data")
##print(images[randindex])
