# Prints a random number from the MNIST training dataset to the console.

from mnist import MNIST
import random

mn_data = MNIST('.\\images')
images, labels = mn_data.load_training()

randindex = random.randrange(0, len(images))

index = 0

print(mn_data.display(images[index]))
print(labels[index])
# print(len(images))
# print("mess of raw data")
print(images[index])
