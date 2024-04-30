import numpy as np
from math import log2

def calculate_entropy(labels):
    """Calculate the entropy of a set of labels."""
    # Count the occurrences of each label
    label_counts = {}
    total_samples = len(labels)
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Calculate entropy using the formula: -p_i * log2(p_i)
    entropy = 0
    for count in label_counts.values():
        probability = count / total_samples
        entropy -= probability * log2(probability)
    return entropy

def calculate_information_gain(feature_values, labels):
    """Calculate the information gain for a feature."""
    total_samples = len(labels)
    parent_entropy = calculate_entropy(labels)
    unique_values = set(feature_values)
    weighted_entropy = 0
    
    # Calculate the entropy of each subset based on unique feature values
    for value in unique_values:
        subset_indices = [i for i, val in enumerate(feature_values) if val == value]
        subset_labels = [labels[i] for i in subset_indices]
        subset_entropy = calculate_entropy(subset_labels)
        subset_weight = len(subset_indices) / total_samples
        weighted_entropy += subset_weight * subset_entropy
    
    # Calculate information gain as the difference between parent entropy and weighted entropy
    information_gain = parent_entropy - weighted_entropy
    return information_gain

def select_root_node(features, labels):
    """Select the root node using Information Gain."""
    num_features = len(features[0])
    information_gains = []
    
    # Calculate information gain for each feature and select the one with maximum gain
    for i in range(num_features):
        feature_values = [sample[i] for sample in features]
        information_gain = calculate_information_gain(feature_values, labels)
        information_gains.append(information_gain)
    
    # Select the index of the feature with maximum information gain
    root_node_index = np.argmax(information_gains)
    return root_node_index

# Example data
features = [[1, 0], [1, 1], [0, 0], [0, 1]]
labels = [1, 1, 0, 0]

# Select the root node
root_node_index = select_root_node(features, labels)
print("Index of the selected root node:", root_node_index)
