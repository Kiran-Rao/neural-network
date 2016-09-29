import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_d(z):
    return np.exp(-z) / (np.square((np.exp(-z) + 1)))


if __name__ == "__main__":
    print('Sigmoid Tests')
    testVector = np.matrix("1 0 -1")

    for i in range(2, 5):
        epsilon = 10 ** (-i) # Testing 0.01 ... 0.0001
        print("Epsilon: ", epsilon)

        testDerivative = (sigmoid(testVector + epsilon) - sigmoid(testVector)) / epsilon
        realDerivative = sigmoid_d(testVector)
        print(realDerivative - testDerivative)
