import numpy as np
import pandas as pd

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights randomly
def initialize_weights(input_size):
    return np.random.randn(input_size, 1)

# Predict using the sigmoid activation function
def predict(inputs, weights):
    return sigmoid(np.dot(inputs, weights))

# Train the perceptron using gradient descent
def train_perceptron(inputs, labels, learning_rate, epochs):
    input_size = inputs.shape[1]
    weights = initialize_weights(input_size)
    converged = False

    for epoch in range(epochs):
        predictions = predict(inputs, weights)
        error = labels - predictions
        weights += learning_rate * np.dot(inputs.T, error) / len(labels)
        
        # Check convergence
        if np.linalg.norm(error) < 0.001:
            converged = True
            break

    return weights, converged, epoch + 1

# Train the model using the pseudo-inverse method
def train_pseudo_inverse(inputs, labels):
    inputs_with_bias = np.hstack((np.ones((inputs.shape[0], 1)), inputs))
    weights = np.dot(np.linalg.pinv(inputs_with_bias), labels)
    return weights

# Calculate accuracy by comparing predicted and actual labels
def calculate_accuracy(predictions, labels):
    correct = np.sum(predictions == labels)
    total = len(labels)
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    # Sample customer data
    customer_data = {
        'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
        'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
        'Milk_Packets': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
        'Payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198],
        'High_Value_Tx': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No']
    }

    customer_df = pd.DataFrame(customer_data)

    # Prepare inputs and labels
    inputs = np.array(customer_df.iloc[:, :-1])
    labels = np.array(customer_df['High_Value_Tx'].map({'Yes': 1, 'No': 0})).reshape(-1, 1)

    learning_rate = 0.01
    epochs = 1000

    # Train with perceptron
    weights_perceptron, converged, epochs_perceptron = train_perceptron(inputs, labels, learning_rate, epochs)
    classified_transactions_perceptron = ['Yes' if p >= 0.5 else 'No' for p in predict(inputs, weights_perceptron)]

    # Train with pseudo-inverse
    weights_pseudo_inverse = train_pseudo_inverse(inputs, labels)
    inputs_with_bias = np.hstack((np.ones((inputs.shape[0], 1)), inputs))
    predictions_pseudo_inverse = predict(inputs_with_bias, weights_pseudo_inverse)
    classified_transactions_pseudo_inverse = ['Yes' if p >= 0.5 else 'No' for p in predictions_pseudo_inverse]

    # Display results
    print("Weights after training with perceptron:")
    print(weights_perceptron)
    
    print("Weights obtained with matrix pseudo-inverse:")
    print(weights_pseudo_inverse)

    print("Number of epochs needed to converge with perceptron:", epochs_perceptron)

    print("Classified Transactions (Perceptron):", classified_transactions_perceptron)
    print("Classified Transactions (Pseudo-inverse):", classified_transactions_pseudo_inverse)

    # Calculate accuracy for perceptron
    accuracy_perceptron = calculate_accuracy(classified_transactions_perceptron, customer_df['High_Value_Tx'])

    # Calculate accuracy for pseudo-inverse
    accuracy_pseudo_inverse = calculate_accuracy(classified_transactions_pseudo_inverse, customer_df['High_Value_Tx'])

    print("Accuracy of Perceptron:", accuracy_perceptron)
    print("Accuracy of Pseudo-inverse:", accuracy_pseudo_inverse)
