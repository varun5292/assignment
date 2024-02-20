import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import traceback

# A1: Evaluate intraclass spread and interclass distances
def evaluate_class_distances(class1_data, class2_data):
    centroid1 = np.mean(class1_data, axis=0)
    centroid2 = np.mean(class2_data, axis=0)
    spread1 = np.std(class1_data, axis=0)
    spread2 = np.std(class2_data, axis=0)
    distance = np.linalg.norm(centroid1 - centroid2)
    return centroid1, centroid2, spread1, spread2, distance

# A2: Plot histogram for a feature
def plot_histogram(feature_data):
    plt.hist(feature_data, bins=10, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Feature')
    plt.xlabel('Feature Values')
    plt.ylabel('Frequency')
    plt.show()
    mean = np.mean(feature_data)
    variance = np.var(feature_data)
    return mean, variance

# A3: Calculate Minkowski distance
def calculate_minkowski_distance(vec1, vec2, r_values):
    distances = []
    for r in r_values:
        distance = np.linalg.norm(vec1 - vec2, ord=r)
        distances.append(distance)
    return distances

# A4: Split dataset into train and test sets
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# A5: Train kNN classifier
def train_knn(X_train, y_train):
    neigh = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    neigh.fit(X_train, y_train)
    return neigh

# A6: Test kNN classifier
def test_knn(neigh, X_test, y_test):
    accuracy = neigh.score(X_test, y_test)
    return accuracy

# A7: Predict using kNN classifier
def predict_knn(neigh, X_test):
    predictions = neigh.predict(X_test)
    return predictions

# A8: Plot accuracy for different k values
def plot_accuracy(X_train, X_test, y_train, y_test):
    k_values = list(range(1, 12))
    train_accuracy = []
    test_accuracy = []
    for k in k_values:
        neigh = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        neigh.fit(X_train, y_train)
        train_accuracy.append(neigh.score(X_train, y_train))
        test_accuracy.append(neigh.score(X_test, y_test))
    
    plt.plot(k_values, train_accuracy, label='Train Accuracy')
    plt.plot(k_values, test_accuracy, label='Test Accuracy')
    plt.title('Accuracy vs. k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# A9: Evaluate confusion matrix and performance metrics
def evaluate_performance(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return cm, accuracy, precision, recall, f1


# Load data from Excel files
file_paths = ["C:\\Users\\mvy48\\Downloads\\DCT_malayalam_char 1.xlsx"]

try:
    combined_df = pd.concat([pd.read_excel(file_path) for file_path in file_paths], ignore_index=True)
except Exception as e:
    print("An error occurred while reading the Excel file:")
    print(traceback.format_exc())
    combined_df = None

if combined_df is not None:
    # Print out DataFrame information for debugging
    print("DataFrame Information:")
    print(combined_df.info())
    print("\nDataFrame Columns:")
    print(combined_df.columns)

    # Check if 'LABEL' column exists
    if 'LABEL' in combined_df.columns:
        # Extract features and labels
        X = combined_df.drop(columns=['LABEL']).values
        y = combined_df['LABEL'].values

        # A1: Evaluate intraclass spread and interclass distances
        class0_data = X[y == 0]
        class1_data = X[y == 1]
        centroid0, centroid1, spread0, spread1, distance = evaluate_class_distances(class0_data, class1_data)
        print("Class 0 Centroid:", centroid0)
        print("Class 1 Centroid:", centroid1)
        print("Class 0 Spread:", spread0)
        print("Class 1 Spread:", spread1)
        print("Interclass Distance:", distance)

        # A2: Plot histogram for a feature
        feature_data = X[:, 0]  # Using the first feature for demonstration
        mean, variance = plot_histogram(feature_data)
        print("Mean:", mean)
        print("Variance:", variance)

        # A3: Calculate Minkowski distance
        vec1 = X[0]
        vec2 = X[1]
        r_values = list(range(1, 11))
        distances = calculate_minkowski_distance(vec1, vec2, r_values)
        plt.plot(r_values, distances)
        plt.title('Minkowski Distance')
        plt.xlabel('r')
        plt.ylabel('Distance')
        plt.show()

        # A4: Split dataset into train and test sets
        X_train, X_test, y_train, y_test = split_dataset(X, y)

        # A5: Train kNN classifier
        neigh = train_knn(X_train, y_train)

        # A6: Test kNN classifier
        accuracy = test_knn(neigh, X_test, y_test)
        print("Test Accuracy:", accuracy)

        # A7: Predict using kNN classifier
        predictions = predict_knn(neigh, X_test)
        print("Predictions:", predictions)

        # A8: Plot accuracy for different k values
        plot_accuracy(X_train, X_test, y_train, y_test)

        # A9: Evaluate confusion matrix and performance metrics
        cm, accuracy, precision, recall, f1 = evaluate_performance(y_test, predictions)
        print("Confusion Matrix:\n", cm)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
    else:
        print("The 'LABEL' column does not exist in the DataFrame.")
else:
    print("Failed to load data from the Excel file.")