import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.learning_rate = learning_rate

    # Step function for binary classification
    def step_function(self, x):
        return np.where(x >= 0, 1, 0)

    # Predict function using the step function
    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.step_function(weighted_sum)

    # Training function
    def train(self, inputs, labels, epochs=1000):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = labels[i] - prediction
                total_error += error**2  # Square the error for sum-squared error
                self.weights[1:] += self.learning_rate * error * inputs[i]
                self.weights[0] += self.learning_rate * error

            # Calculate and store average error for the epoch
            average_error = total_error / len(inputs)
            errors.append(average_error)

            # Stop if convergence criterion is met
            if average_error <= 0.002:
                break

        return errors

# Initial weights
initial_weights = np.array([10, 0.2, -0.75])

# Learning rate
learning_rate = 0.05

# AND gate inputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND gate labels
labels = np.array([0, 0, 0, 1])

# Initialize Perceptron
perceptron = Perceptron(initial_weights, learning_rate)

# Train Perceptron
errors = perceptron.train(inputs, labels)

# Plot epochs vs errors
epochs = np.arange(1, len(errors) + 1)  # Adjust for indexing
plt.plot(epochs, errors)
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("Learning Process")
plt.grid(True)
plt.show()

# Test the trained Perceptron
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range(len(test_inputs)):
    prediction = perceptron.predict(test_inputs[i])
    print(f"Input: {test_inputs[i]} Predicted Output: {prediction}")
