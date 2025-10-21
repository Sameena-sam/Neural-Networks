import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of sigmoid for backpropagation

# Initialize inputs and expected output
X = np.array([[0.5, 0.8]])  # Example input with 2 features
y = np.array([[1]])  # Expected output

# Initialize weights and biases randomly
np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.rand(2, 2)  # 2 input -> 2 hidden neurons
weights_hidden_output = np.random.rand(2, 1)  # 2 hidden -> 1 output neuron
bias_hidden = np.random.rand(1, 2)  # Bias for hidden layer
bias_output = np.random.rand(1, 1)  # Bias for output neuron

# Learning rate
learning_rate = 0.1

# Forward pass
hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
output = sigmoid(output_layer_input)

# Compute error
error = y - output

# Backpropagation
output_error_term = error * sigmoid_derivative(output)
hidden_error_term = output_error_term.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

# Update weights and biases
weights_hidden_output += hidden_layer_output.T.dot(output_error_term) * learning_rate
weights_input_hidden += X.T.dot(hidden_error_term) * learning_rate
bias_output += np.sum(output_error_term, axis=0, keepdims=True) * learning_rate
bias_hidden += np.sum(hidden_error_term, axis=0, keepdims=True) * learning_rate

print("Updated Weights (Hidden to Output):", weights_hidden_output)
print("Updated Weights (Input to Hidden):", weights_input_hidden)
print("Updated Biases (Output Layer):", bias_output)
print("Updated Biases (Hidden Layer):", bias_hidden)
