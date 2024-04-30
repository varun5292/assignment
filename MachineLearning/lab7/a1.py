import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input data for AND gate
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Output data for AND gate
output_data = np.array([[1], [1], [1], [1]])  # Output O1

# Set the seed for reproducibility
np.random.seed(42)

# Initialize weights and biases
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.05
epochs = 1000

# Initialize weights and biases for the connections
weights_input_to_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
biases_hidden = np.zeros((1, hidden_size))

weights_hidden_to_output = 2 * np.random.random((hidden_size, output_size)) - 1
biases_output = np.zeros((1, output_size))

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    input_layer = input_data
    hidden_layer_input = np.dot(input_layer, weights_input_to_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + biases_output
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the error
    error = output_data - output_layer_output

    # Backward pass
    d_output = error * sigmoid_derivative(output_layer_output)
    d_hidden = d_output.dot(weights_hidden_to_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_to_output += learning_rate * hidden_layer_output.T.dot(d_output)
    biases_output += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    weights_input_to_hidden += learning_rate * input_layer.T.dot(d_hidden)
    biases_hidden += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    # Check for convergence
    if np.mean(np.abs(error)) <= 0.002:
        print(f"Converged at epoch {epoch + 1}")
        break

# Print the final weights and biases
print("Final weights (Input to Hidden):")
print(weights_input_to_hidden)
print("Final biases (Hidden):")
print(biases_hidden)
print("Final weights (Hidden to Output):")
print(weights_hidden_to_output)
print("Final biases (Output):")
print(biases_output)

# Test the trained neural network
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden_layer_test = sigmoid(np.dot(test_input, weights_input_to_hidden) + biases_hidden)
predicted_output = sigmoid(np.dot(hidden_layer_test, weights_hidden_to_output) + biases_output)
print("Predicted output (Output):")
print(predicted_output)
