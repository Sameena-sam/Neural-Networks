import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

class MLPScratch:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.random.randn(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.random.randn(output_dim)

    def forward(self,x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(x, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)

Multilayerperceptron = MLPScratch(input_dim=3,hidden_dim=2,output_dim=1)
vector = np.array([0.2,0.1,0.3])
output = Multilayerperceptron.forward(vector)
print(vector)
