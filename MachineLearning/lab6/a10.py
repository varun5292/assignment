from sklearn.neural_network import MLPClassifier
import numpy as np

# Define the AND gate truth table
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Create an MLPClassifier for the AND gate
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)

# Train the model on the AND gate truth table
mlp_and.fit(X_and, y_and)

# Test the model on the AND gate truth table
test_data_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output_and = mlp_and.predict(test_data_and)

# Print the predicted output for the AND gate
print("Predicted output for AND gate:")
print(predicted_output_and)

# Define the XOR gate truth table
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create an MLPClassifier for the XOR gate
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)

# Train the model on the XOR gate truth table
mlp_xor.fit(X_xor, y_xor)

# Test the model on the XOR gate truth table
test_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output_xor = mlp_xor.predict(test_data_xor)

# Print the predicted output for the XOR gate
print("Predicted output for XOR gate:")
print(predicted_output_xor)
