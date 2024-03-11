from collections import Counter

def euclidean_distance(vector1, vector2):
    """Calculate and return the Euclidean distance between two vectors."""
    return sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)) ** 0.5

def manhattan_distance(vector1, vector2):
    """Calculate and return the Manhattan distance between two vectors."""
    return sum(abs(v1 - v2) for v1, v2 in zip(vector1, vector2))

def k_nearest_neighbors(training_data, test_instance, k=3):
    """Predict the label for a test instance using k-nearest neighbors algorithm."""
    distances = [(euclidean_distance(test_instance, training_instance[0]), training_instance[1]) for training_instance in training_data]
    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_labels = [label for _, label in sorted_distances[:k]]
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label

def label_encoding(categories):
    """Encode categorical data with integer labels and return the mapping."""
    encoded_mapping = {}
    for index, category in enumerate(categories):
        encoded_mapping[category] = index
    return encoded_mapping

def get_unique_labels(data):
    """Get the unique labels present in the given data."""
    return list(set(data))

def one_hot_encoding(data, unique_labels):
    """Performing one-hot encoding for the given data based on unique labels."""
    one_hot_matrix = []
    for value in data:
        one_hot_vector = []
        for label in unique_labels:
            # Set 1 if the value matches the label, 0 otherwise
            one_hot_vector.append(1 if value == label else 0)
        one_hot_matrix.append(one_hot_vector)
    return one_hot_matrix

def categorical_to_numeric_one_hot(data):
    """Convert categorical data to a numeric one-hot encoded matrix."""
    unique_labels = get_unique_labels(data)
    one_hot_matrix = one_hot_encoding(data, unique_labels)
    return one_hot_matrix

if __name__ == "__main__":
    # Get user input for vectors
    vector1_str = input("Enter vector 1 (comma-separated values): ")
    vector2_str = input("Enter vector 2 (comma-separated values): ")

    # Convert input strings to lists of integers
    vector1 = [int(x) for x in vector1_str.split(',')]
    vector2 = [int(x) for x in vector2_str.split(',')]

    # Display input vectors
    print("Vector 1:", vector1)
    print("Vector 2:", vector2)

    # Calculate distances
    euclidean_dist = euclidean_distance(vector1, vector2)
    manhattan_dist = manhattan_distance(vector1, vector2)

    # Display distances
    print("Euclidean Distance:", euclidean_dist)
    print("Manhattan Distance:", manhattan_dist)

    # Collect training data
    training_data = []
    num_train_instances = int(input("Enter the number of training instances: "))

    for _ in range(num_train_instances):
        features = [float(x) for x in input("Enter features (comma-separated values): ").split(',')]
        label = input("Enter the label for this instance: ")
        training_data.append((features, label))

    # Collect test instance and k value
    test_instance = [float(x) for x in input("Enter test instance features (comma-separated values): ").split(',')]
    k_value = int(input("Enter the value of k: "))

    # Predict and display the result
    predicted_label = k_nearest_neighbors(training_data, test_instance, k=k_value)
    print(f"The predicted label for the test instance is: {predicted_label}")

    # Get user input for categorical data
    categorical_data_str = input("Enter categorical data (comma-separated values): ")
    categorical_data = categorical_data_str.split(',')

    # Perform label encoding
    label_encoded_mapping = label_encoding(categorical_data)

    # Display original and label-encoded data
    print("Original Categorical Data:", categorical_data)
    print("Label Encoded Mapping:", label_encoded_mapping)

    # Convert and display the result
    one_hot_matrix = categorical_to_numeric_one_hot(categorical_data)
    print(f"Original Categorical Data: {categorical_data}")
    print("One-Hot Encoded Matrix:")
    for row in one_hot_matrix:
        print(row)
