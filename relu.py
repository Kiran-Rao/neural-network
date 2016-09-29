import numpy as np


def relu(z):
    return np.maximum(z, 0)

def relu_d(z):
    return np.sign(np.maximum(z, 0))


if __name__ == "__main__":
    print('ReLU Tests')
    testVector = np.matrix("5 1 0 -1 -2")

    print(testVector)
    print(relu(testVector))
    print(relu_d(testVector))
