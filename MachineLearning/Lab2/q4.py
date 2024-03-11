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

# Get user input for categorical data
categorical_data = input("Enter categorical data (comma-separated values): ").split(',')

# Convert and display the result
one_hot_matrix = categorical_to_numeric_one_hot(categorical_data)
print(f"Original Categorical Data: {categorical_data}")
print("One-Hot Encoded Matrix:")
for row in one_hot_matrix:
    print(row)
