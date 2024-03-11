import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize weights for the neural network
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.rand(input_size, hidden_size)
    W2 = np.random.rand(hidden_size, output_size)
    return W1, W2

# Forward propagation through the neural network
def forward_propagate(X, W1, W2):
    z1 = np.dot(X, W1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2)
    a2 = sigmoid(z2)
    return a1, a2

# Backpropagation to update weights based on the error
def backpropagate(X, y, a1, a2, W1, W2):
    dA2 = (a2 - y) * sigmoid_derivative(a2)
    dA1 = np.dot(dA2, W2.T) * sigmoid_derivative(a1)
    dW2 = np.dot(a1.T, dA2)
    dW1 = np.dot(X.T, dA1)
    return dW1, dW2

# Train the neural network using gradient descent
def train_neural_network(X, y, learning_rate, epochs):
    input_size = X.shape[1]
    hidden_size = 4
    output_size = y.shape[1]
    W1, W2 = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        a1, a2 = forward_propagate(X, W1, W2)
        dW1, dW2 = backpropagate(X, y, a1, a2, W1, W2)
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
    return W1, W2

# Test the neural network on given data
def test_neural_network(X, y, W1, W2):
    _, a2 = forward_propagate(X, W1, W2)
    predictions = np.round(a2)
    accuracy = np.mean(predictions == y)
    return accuracy

# Define input-output pairs for the AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])

# Train the neural network
learning_rate = 0.05
epochs = 1000
W1, W2 = train_neural_network(X, y_and, learning_rate, epochs)

# Test the neural network and calculate accuracy
accuracy = test_neural_network(X, y_and, W1, W2)
print(f'Accuracy for AND Gate: {accuracy:.4f}')

# Print the final weights
print("Final weights:")
print("W1 (Input to Hidden):")
print(W1)
print("W2 (Hidden to Output):")
print(W2)

# Predict the output
_, output = forward_propagate(X, W1, W2)
print("Predicted output:")
print(np.round(output))
