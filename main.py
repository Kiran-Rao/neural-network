import numpy as np
import sigmoid

class Neural_Net(object):
    def __init__(self):
        # Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.batchSize = 10
        self.learningRate = 0

        # Weights
        self.weights1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.weights2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Forward Propegation
        self.z2 = np.dot(X, self.weights1)
        self.a2 = sigmoid.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)
        return self.z3

    def backprop(self, X):
        # Back Propegation
        pass

if __name__ == "__main__":
    nn = Neural_Net()
    print ("Weights: ")
    print (nn.weights1)
    print (nn.weights2)
