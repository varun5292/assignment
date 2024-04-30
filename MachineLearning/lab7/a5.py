import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load data from Excel
data = pd.read_excel(r"C:\\Users\\nvsur\\OneDrive\\Documents\\MachineLearning\\lab7\\Lab Session1 Data.xlsx")

# Separate features and target variable
features = data.iloc[:, 0:383]
target = data.iloc[:, 384]
target = target.astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for MLPClassifier
param_grid = {
    'hidden_layer_sizes': [(100,), (200,), (300,)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

# Instantiate MLPClassifier
mlp_classifier = MLPClassifier(max_iter=1000, tol=1e-4)  # Increase max_iter and set tolerance to a lower value

# Instantiate RandomizedSearchCV
random_search = RandomizedSearchCV(mlp_classifier, param_distributions=param_grid, n_iter=10, cv=3, random_state=42)

# Suppress ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # Fit RandomizedSearchCV to the training data
    random_search.fit(X_train_scaled, y_train)

# Extract the best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Evaluate the model on the test data
y_pred = random_search.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
