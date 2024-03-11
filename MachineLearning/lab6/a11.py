import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data_path = "C:/Users/mvy48/OneDrive/Desktop/vscodeprograms/ml_labsessions/lab4/DCT_malayalam_char 1.xlsx"
dataset = pd.read_excel(data_path)

# Separate features (X) and labels (y)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and configure the Multi-Layer Perceptron (MLP) classifier
model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',  # Rectified Linear Unit (ReLU) activation function
    solver='adam',  # Optimizer for weight optimization
    max_iter=1000,  # Maximum number of iterations for optimization
    random_state=42  # Set random seed for reproducibility
)

# Train the MLP classifier on the training data
model.fit(X_train, y_train)

# Predict labels on the test data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
