import numpy as np
import sigmoid

class Neural_Net(object):
    def __init__(self):
        # Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        # self.batchSize = 10
        # self.learningRate = 0

        # Weights
        self.weights1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.weights2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Forward Propegation
        self.z2 = np.dot(X, self.weights1)
        self.a2 = sigmoid.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)
        yHat = sigmoid.sigmoid(self.z3)
        return yHat

    def backprop(self, X):
        # Back Propegation
        pass

    def cost(self, X, y):
        # Cost function based on mean square errors
        yHat = self.forward(X)
        return np.mean(np.square(y - yHat)) / 2

    def cost_d(self, X, y):
        # Partial derivative of cost fuction wrt weight 1 and weight 2
        self.yHat = self.forward(X)

        # Calculate partial derivative wrt weight 2
        delta3 = np.multiply( -(y - self.yHat), sigmoid.sigmoid_d(self.z3))
        dJdW2 = np.dot(np.transpose(self.a2), delta3)

        # Calculate partial derivative wrt weight 1
        delta2 = np.dot(delta3, np.transpose(self.weights2)) * sigmoid.sigmoid_d(self.z2)
        dJdW1 = np.dot(np.transpose(X), delta2)

        print('Partial Derivatives')
        print(dJdW1, dJdW2)
        return dJdW1, dJdW2

if __name__ == "__main__":
    nn = Neural_Net()
    print("Weights: ")
    print(nn.weights1)
    print(nn.weights2)
    x = [[1, 0.6], [0.6, 0.2], [0.4, 0.8]]
    y = [[0.6], [0.2], [0.8]]
    nn.cost(x, y)
    nn.cost_d(x, y)
