import numpy as np
import matplotlib.pyplot as plt

# Step function for perceptron output
def step(x):
    return 1 if x >= 0 else 0

# Train a perceptron using the perceptron learning algorithm
def train_perceptron(inputs, outputs, learning_rate, epochs=1000):
    # Initialize weights
    w0 = 10
    w1 = 0.2
    w2 = -0.75

    errors = []
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            # Compute the weighted sum and predict the output
            weighted_sum = w0 + np.dot(inputs[i], [w1, w2])
            predicted_output = step(weighted_sum)
            
            # Update the weights based on the perceptron learning rule
            error = outputs[i] - predicted_output
            total_error += error**2

            w0 += learning_rate * error
            w1 += learning_rate * error * inputs[i][0]
            w2 += learning_rate * error * inputs[i][1]

        # Calculate average error for the epoch
        average_error = total_error / len(inputs)
        errors.append(average_error)

        # Check for convergence (average error threshold)
        if average_error <= 0.002:
            return w0, w1, w2, epoch + 1, errors

    return w0, w1, w2, epochs, errors

# Sample inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])

# Set learning rate
learning_rate = 0.05

# Train the perceptron
w0, w1, w2, converged_epoch, errors = train_perceptron(inputs, outputs, learning_rate)

# Display results
print("A1 - Learning Rate:", learning_rate)
print(f"Final weights: w0: {w0:.4f}, w1: {w1:.4f}, w2: {w2:.4f}")
if converged_epoch < 1000:
    print(f"Converged in {converged_epoch} epochs")
else:
    print("Convergence not reached within 1000 epochs")

# Plot the learning process
plt.plot(range(1, converged_epoch + 1), errors)
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("Learning Process (A1)")
plt.grid(True)
plt.show()

# Test different learning rates and plot convergence epochs
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
convergence_epochs = []
for lr in learning_rates:
    w0, w1, w2, converged_epoch, _ = train_perceptron(inputs, outputs, lr)
    convergence_epochs.append(converged_epoch if converged_epoch < 1000 else 1000)

# Plot the convergence epochs vs. learning rates
plt.plot(learning_rates, convergence_epochs)
plt.xlabel("Learning Rate")
plt.ylabel("Epochs until Convergence (or 1000)")
plt.title("Convergence Epochs vs. Learning Rates (A3)")
plt.grid(True)
plt.show()
