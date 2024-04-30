import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning

# Define the AND gate truth table
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Define the XOR gate truth table
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create a function to perform hyperparameter tuning using RandomizedSearchCV
def tune_hyperparameters(X, y):
    # Define parameter grid for MLPClassifier
    param_grid = {
        'hidden_layer_sizes': [(2,), (4,), (8,), (16,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }

    # Instantiate MLPClassifier
    mlp = MLPClassifier(max_iter=1000, random_state=42)

    # Instantiate RandomizedSearchCV with 10 iterations in cross-validation
    random_search = RandomizedSearchCV(mlp, param_distributions=param_grid, n_iter=10, cv=2, random_state=42)

    # Suppress ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # Fit RandomizedSearchCV to the training data
        random_search.fit(X, y)

    # Extract the best parameters
    best_params = random_search.best_params_

    return best_params

# Tune hyperparameters for AND gate
best_params_and = tune_hyperparameters(X_and, y_and)
print("Best Parameters for AND gate:", best_params_and)

# Tune hyperparameters for XOR gate
best_params_xor = tune_hyperparameters(X_xor, y_xor)
print("Best Parameters for XOR gate:", best_params_xor)
