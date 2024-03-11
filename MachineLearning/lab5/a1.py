import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Load dataset
dataset_path = "C:\\Users\\mvy48\\OneDrive\\Desktop\\vscodeprograms\\ml_labsessions\\lab4\\DCT_malayalam_char 1.xlsx"
df = pd.read_excel(dataset_path)

# Extract features and labels
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# Define class labels
class_label_1 = 3353
class_label_2 = 3378

# Select indices for class labels
class_1_indices = np.where(labels == class_label_1)[0]
class_2_indices = np.where(labels == class_label_2)[0]

# Combine selected indices
selected_indices = np.concatenate((class_1_indices, class_2_indices))

# Create selected features and labels
selected_features = features[selected_indices]
selected_labels = labels[selected_indices]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, selected_labels, test_size=0.3, random_state=42)

# Code for kNN classification
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

# Evaluate kNN classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(selected_labels)))
plt.xticks(tick_marks, np.unique(selected_labels), rotation=45)
plt.yticks(tick_marks, np.unique(selected_labels))
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Additional kNN classifier evaluation metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

plt.show()

# Regression evaluation on Purchase data
file_path = r"C:\Users\mvy48\OneDrive\Desktop\vscodeprograms\ml_labsessions\Lab_Session1_Data.xlsx"
df_purchase = pd.read_excel(file_path, sheet_name="Purchase data")

actual_prices_column_name = 'Milk'
predicted_prices_column_name = 'Milk'

actual_prices = df_purchase[actual_prices_column_name].values
predicted_prices = df_purchase[predicted_prices_column_name].values

df_cleaned = df_purchase.dropna(subset=[actual_prices_column_name, predicted_prices_column_name])

actual_prices = df_cleaned[actual_prices_column_name].values
predicted_prices = df_cleaned[predicted_prices_column_name].values

mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

if len(actual_prices) >= 2:
    r2 = r2_score(actual_prices, predicted_prices)
    print(f'R-squared (R2) score: {r2}')
else:
    print('Insufficient samples to calculate R-squared.')

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# Scatter plot of random training data
np.random.seed(0)
X_rand = np.random.randint(1, 11, size=(20, 2))
classes_rand = np.random.randint(0, 2, size=20)
class0_points = X_rand[classes_rand == 0]
class1_points = X_rand[classes_rand == 1]

plt.figure(figsize=(8, 6))
plt.scatter(class0_points[:, 0], class0_points[:, 1], color='blue', label='Class 0')
plt.scatter(class1_points[:, 0], class1_points[:, 1], color='red', label='Class 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Training Data')
plt.legend()
plt.grid(True)
plt.show()

# kNN classification and visualization
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_values, y_values)
test_data = np.c_[xx.ravel(), yy.ravel()]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_rand, classes_rand)
predicted_classes = knn.predict(test_data)

class0_test = test_data[predicted_classes == 0]
class1_test = test_data[predicted_classes == 1]

plt.figure(figsize=(8, 6))
plt.scatter(class0_test[:, 0], class0_test[:, 1], color='blue', alpha=0.1, label='Predicted Class 0')
plt.scatter(class1_test[:, 0], class1_test[:, 1], color='red', alpha=0.1, label='Predicted Class 1')
plt.scatter(class0_points[:, 0], class0_points[:, 1], color='blue', label='Class 0', marker='x')
plt.scatter(class1_points[:, 0], class1_points[:, 1], color='red', label='Class 1', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Test Data with Predicted Classes')
plt.legend()
plt.grid(True)
plt.show()

# kNN classification with different k values and visualization
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_values, y_values)
test_data = np.c_[xx.ravel(), yy.ravel()]

k_values = [1, 3, 5, 7, 9]

plt.figure(figsize=(15, 10))

for i, k in enumerate(k_values, start=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_rand, classes_rand)
    predicted_classes = knn.predict(test_data)
    
    class0_test = test_data[predicted_classes == 0]
    class1_test = test_data[predicted_classes == 1]

    plt.subplot(2, 3, i)
    plt.scatter(class0_test[:, 0], class0_test[:, 1], color='blue', alpha=0.1, label='Predicted Class 0')
    plt.scatter(class1_test[:, 0], class1_test[:, 1], color='red', alpha=0.1, label='Predicted Class 1')
    plt.scatter(class0_points[:, 0], class0_points[:, 1], color='blue', label='Class 0', marker='x')
    plt.scatter(class1_points[:, 0], class1_points[:, 1], color='red', label='Class 1', marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'kNN Classification (k={k})')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Scatter plot of classes 3353 and 3378
class_3353 = df[df['LABEL'] == 3353]
class_3378 = df[df['LABEL'] == 3378]

plt.scatter(class_3353[0], class_3353[1], color='red', label='Class 3353')
plt.scatter(class_3378[0], class_3378[1], color='blue', label='Class 3378')

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Scatter Plot of Classes 3353 and 3378')
plt.legend()
plt.show()

# kNN classification on specific classes with scatter plot
filtered_df = df[df['LABEL'].isin([3353, 3378])]
X_specific = filtered_df[[0, 1]].values
y_specific = filtered_df['LABEL'].values

X_train_specific, X_test_specific, y_train_specific, y_test_specific = train_test_split(X_specific, y_specific, test_size=0.3)

knn_classifier_specific = KNeighborsClassifier(n_neighbors=3)
knn_classifier_specific.fit(X_train_specific, y_train_specific)

predicted_classes_specific = knn_classifier_specific.predict(X_test_specific)

for i, class_label in enumerate(predicted_classes_specific):
    if class_label == 3353:
        plt.scatter(X_test_specific[i, 0], X_test_specific[i, 1], color='blue', label='Predicted Class 3353')
    elif class_label == 3378:
        plt.scatter(X_test_specific[i, 0], X_test_specific[i, 1], color='red', label='Predicted Class 3378')

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Scatter Plot of Test Data with Predicted Classes')
plt.show()

# kNN classification with different k values and scatter plot
k_values_specific = [1, 3, 5, 7, 9]
for k_specific in k_values_specific:
    knn_classifier_specific = KNeighborsClassifier(n_neighbors=k_specific)
    knn_classifier_specific.fit(X_train_specific, y_train_specific)

    # Predict the classes for test data
    predicted_classes_specific = knn_classifier_specific.predict(X_test_specific)

    # Plot the test points with predicted classes
    for i, class_label in enumerate(predicted_classes_specific):
        if class_label == 3353:
            plt.scatter(X_test_specific[i, 0], X_test_specific[i, 1], color='blue', label='Predicted Class 3353')
        elif class_label == 3378:
            plt.scatter(X_test_specific[i, 0], X_test_specific[i, 1], color='red', label='Predicted Class 3378')

    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title(f'Scatter Plot of Test Data with Predicted Classes (k={k_specific})')
    plt.show()

# kNN classifier with grid search for best k value
param_grid_specific = {'n_neighbors': [1, 3, 5, 7, 9]}

# Initialize kNN classifier
knn_classifier_grid = KNeighborsClassifier()

grid_search_specific = GridSearchCV(knn_classifier_grid, param_grid_specific, cv=5)
grid_search_specific.fit(X_train_specific, y_train_specific)

print("Best parameters:", grid_search_specific.best_params_)

best_knn_classifier_specific = grid_search_specific.best_estimator_

predicted_classes_grid = best_knn_classifier_specific.predict(X_test_specific)

for i, class_label in enumerate(predicted_classes_grid):
    if class_label == 3353:
        plt.scatter(X_test_specific[i, 0], X_test_specific[i, 1], color='blue', label='Predicted Class 3353')
    elif class_label == 3378:
        plt.scatter(X_test_specific[i, 0], X_test_specific[i, 1], color='red', label='Predicted Class 3378')

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title(f'Scatter Plot of Test Data with Predicted Classes (Best k={grid_search_specific.best_params_["n_neighbors"]})')
plt.show()
