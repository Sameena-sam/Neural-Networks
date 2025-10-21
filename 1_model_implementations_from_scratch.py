import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  

class perceptronScratch:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()

    def forward(self,x):
        linear_output = np.dot(x,self.weights) + self.bias
        return sigmoid(linear_output)
    
perceptron = perceptronScratch(input_dim = 4)
input_vector = np.array([-0.9, 0.2, 1.3, 3.4])
output = perceptron.forward(input_vector)
print(output)