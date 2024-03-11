import numpy as np
import matplotlib.pyplot as plt

# Step function for perceptron output
def step_function(x):
    return 1 if x >= 0 else 0

# Function to calculate error given inputs, outputs, and weights
def calculate_error(X, Y, W):
    error = 0
    for i in range(len(X)):
        prediction = step_function(np.dot(X[i], W[1:]) + W[0])
        error += (Y[i] - prediction) ** 2
    return error

# Define XOR gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_xor = np.array([0, 1, 1, 0])

# Initial weights for perceptron
initial_weights = np.array([10, 0.2, -0.75])

# Training parameters
learning_rate = 0.05
epochs = 1000
convergence_error = 0.002

# Training perceptron for XOR gate using perceptron learning algorithm
W = initial_weights.copy()
errors_xor = []
for epoch in range(epochs):
    total_error = calculate_error(X, Y_xor, W)
    errors_xor.append(total_error)
    if total_error <= convergence_error:
        print("Converged after", epoch, "epochs")
        break
    for i in range(len(X)):
        prediction = step_function(np.dot(X[i], W[1:]) + W[0])
        W[1:] += learning_rate * (Y_xor[i] - prediction) * X[i]
        W[0] += learning_rate * (Y_xor[i] - prediction)

# Plot epochs vs error
plt.plot(range(1, epoch + 2), errors_xor)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs Error (XOR Gate)')
plt.show()

# Activation functions for perceptron
def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return max(0, x)

# Function to train perceptron for XOR gate with varying activation functions
def train_perceptron_xor(learning_rate, activation_function):
    W = initial_weights.copy()
    for epoch in range(epochs):
        total_error = calculate_error(X, Y_xor, W)
        if total_error <= convergence_error:
            return epoch
        for i in range(len(X)):
            prediction = activation_function(np.dot(X[i], W[1:]) + W[0])
            W[1:] += learning_rate * (Y_xor[i] - prediction) * X[i]
            W[0] += learning_rate * (Y_xor[i] - prediction)
    return epochs 

# Train perceptron with different activation functions and get epochs needed to converge
epochs_bipolar = train_perceptron_xor(learning_rate, bipolar_step_function)
epochs_sigmoid = train_perceptron_xor(learning_rate, sigmoid_function)
epochs_relu = train_perceptron_xor(learning_rate, relu_function)

# Print results
print("Number of epochs taken to converge (XOR Gate) - Bi-Polar Step:", epochs_bipolar)
print("Number of epochs taken to converge (XOR Gate) - Sigmoid:", epochs_sigmoid)
print("Number of epochs taken to converge (XOR Gate) - ReLU:", epochs_relu)

# Varying learning rates and plotting number of iterations needed to converge
learning_rates = np.arange(0.1, 1.1, 0.1)

def train_perceptron_varying_lr(learning_rate):
    W = initial_weights.copy()
    for epoch in range(epochs):
        total_error = calculate_error(X, Y_xor, W)
        if total_error <= convergence_error:
            return epoch
        for i in range(len(X)):
            prediction = step_function(np.dot(X[i], W[1:]) + W[0])
            W[1:] += learning_rate * (Y_xor[i] - prediction) * X[i]
            W[0] += learning_rate * (Y_xor[i] - prediction)
    return epochs 

# Collecting number of iterations for varying learning rates
iterations_xor = []
for learning_rate in learning_rates:
    iterations_xor.append(train_perceptron_varying_lr(learning_rate))

# Plotting learning rates vs number of iterations
plt.plot(learning_rates, iterations_xor, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs Learning Rate (XOR Gate)')
plt.grid(True)
plt.show()
