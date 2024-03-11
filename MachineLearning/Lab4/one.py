import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import minkowski
# Load dataset and calculate mean, spread, and distance between classes
dataset_path = "C:\\Users\\mvy48\\OneDrive\\Desktop\\vscodeprograms\\ml_labsessions\\lab4\\DCT_malayalam_char 1.xlsx"
dataset = pd.read_excel(dataset_path)
unique_classes = np.unique(dataset.iloc[:, -1].values)
print("Unique Class Labels:", unique_classes)

class_label_1 = unique_classes[0]
class_label_2 = unique_classes[1]

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

class_means = []
class_spreads = []

# Calculate mean and spread for each class
for cls in [class_label_1, class_label_2]:
    class_data = features[labels == cls]
    class_mean = np.mean(class_data, axis=0)
    class_spread = np.std(class_data, axis=0)
    class_means.append(class_mean)
    class_spreads.append(class_spread)

# Calculate Euclidean distance between class means
distance_between_classes = np.linalg.norm(class_means[0] - class_means[1])
print("Mean (centroid) of Class", class_label_1, ":", class_means[0])
print("Mean (centroid) of Class", class_label_2, ":", class_means[1])
print("Spread (standard deviation) of Class", class_label_1, ":", class_spreads[0])
print("Spread (standard deviation) of Class", class_label_2, ":", class_spreads[1])
print("Distance between mean vectors of Class", class_label_1, "and Class", class_label_2, ":", distance_between_classes)

# Plot histogram and calculate mean/variance for a selected feature
selected_feature = features[:, 0]
plt.hist(selected_feature, bins=10)
plt.title("Histogram of Selected Feature")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
feature_mean = np.mean(selected_feature)
feature_variance = np.var(selected_feature)
print("Mean of selected feature:", feature_mean)
print("Variance of selected feature:", feature_variance)

# Calculate Minkowski distance for different values of r and plot
def calculate_minkowski_distance(feature_vector1, feature_vector2, r):
    return minkowski(feature_vector1, feature_vector2, r)

r_values = list(range(1, 11))
distances = []

for r in r_values:
    distance = calculate_minkowski_distance(features[1], features[2], r)  # Using features[0] and features[1]
    distances.append(distance)

plt.plot(r_values, distances, marker='o')
plt.title("Minkowski Distance vs. r")
plt.xlabel("r")
plt.ylabel("Minkowski Distance")
plt.xticks(r_values)
plt.grid(True)
plt.show()

# Load dataset and extract features/labels for specific classes
dataset_path = "C:\\Users\\mvy48\\OneDrive\\Desktop\\vscodeprograms\\ml_labsessions\\lab4\\DCT_malayalam_char 1.xlsx"
dataset = pd.read_excel(dataset_path)
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values
class_label_1 = 3353
class_label_2 = 3378
class_1_indices = np.where(labels == class_label_1)[0]
class_2_indices = np.where(labels == class_label_2)[0]
selected_indices = np.concatenate((class_1_indices, class_2_indices))
selected_features = features[selected_indices]
selected_labels = labels[selected_indices]
X_train, X_test, y_train, y_test = train_test_split(selected_features, selected_labels, test_size=0.3)

# Train kNN classifier, calculate accuracy, and make predictions
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
accuracy = neigh.score(X_test, y_test)
print("Accuracy of kNN classifier:", accuracy)
predictions = neigh.predict(X_test)
print("Predicted classes for the test vectors:")
print(predictions)

# Compare kNN and 1-NN classifiers for different values of k
accuracy_kNN = []
accuracy_NN = []

for k in range(1, 12):
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(X_train, y_train)
    accuracy_kNN.append(kNN_classifier.score(X_test, y_test))
    
    NN_classifier = KNeighborsClassifier(n_neighbors=1)
    NN_classifier.fit(X_train, y_train)
    accuracy_NN.append(NN_classifier.score(X_test, y_test))

plt.plot(range(1, 12), accuracy_kNN, label='kNN (k=3)')
plt.plot(range(1, 12), accuracy_NN, label='NN (k=1)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy of kNN vs NN for Different Values of k')
plt.legend()
plt.show()

# Evaluate kNN classifier using confusion matrix and classification report
train_predictions = neigh.predict(X_train)
test_predictions = neigh.predict(X_test)

train_confusion_matrix = confusion_matrix(y_train, train_predictions)
test_confusion_matrix = confusion_matrix(y_test, test_predictions)

print("Confusion Matrix for Training Data:")
print(train_confusion_matrix)
print("\nConfusion Matrix for Test Data:")
print(test_confusion_matrix)

train_classification_report = classification_report(y_train, train_predictions)
test_classification_report = classification_report(y_test, test_predictions)

print("\nClassification Report for Training Data:")
print(train_classification_report)
print("\nClassification Report for Test Data:")
print(test_classification_report)
