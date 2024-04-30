import numpy as np

class Node:
    def __init__(self, feature=None, split=None, left=None, right=None, leaf=None):
        """Initialize a tree node."""
        self.feature = feature    # Feature index for splitting
        self.split = split       # Split value for continuous features
        self.left = left         # Left child node
        self.right = right       # Right child node
        self.leaf = leaf         # Leaf node value (class label if leaf)
    
    def print_tree(self, depth=0):
        """Print the tree structure recursively."""
        indent = "  " * depth
        if self.leaf is not None:
            print(indent + "Leaf: Class", self.leaf)
        else:
            print(indent + "Feature", self.feature, "<=", self.split)
            if self.left:
                self.left.print_tree(depth + 1)
            if self.right:
                self.right.print_tree(depth + 1)

def entropy(labels):
    """Calculate entropy of labels."""
    counts = {label: labels.count(label) for label in labels}  # Count occurrences of each label
    total = len(labels)                                       # Total number of labels
    entropy = sum(-count/total * np.log2(count/total) for count in counts.values())  # Calculate entropy
    return entropy

def best_split(features, labels):
    """Find best feature and split value."""
    best_gain = 0        # Initialize best information gain
    best_feature = None  # Initialize best feature index
    best_split_value = None   # Initialize best split value
    for i in range(len(features[0])):  # Iterate over each feature
        values = [sample[i] for sample in features]  # Get feature values
        unique_values = sorted(set(values))   # Get unique feature values
        for j in range(len(unique_values) - 1):  # Iterate over unique feature values for possible splits
            split_value = (unique_values[j] + unique_values[j+1]) / 2  # Calculate split value as average of adjacent unique values
           
            left_labels = [label for k, label in enumerate(labels) if features[k][i] <= split_value]  # Labels in left split
            right_labels = [label for k, label in enumerate(labels) if features[k][i] > split_value]   # Labels in right split
            # Calculate information gain
            gain = entropy(labels) - (len(left_labels)/len(labels)) * entropy(left_labels) - (len(right_labels)/len(labels)) * entropy(right_labels)
            # Update best gain, feature index, and split value if current gain is higher
            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_split_value = split_value
    return best_feature, best_split_value

def build_tree(features, labels, max_depth=None):
    """Build decision tree recursively."""
    # Base case: if maximum depth is reached or all labels are the same
    if max_depth is not None and (max_depth == 0 or len(set(labels)) == 1):
        return Node(leaf=max(labels, key=labels.count))  # Create leaf node with most common label
    feature, split_value = best_split(features, labels)  # Find best feature and split value
    # If no best split is found (e.g., due to categorical features with the same values), create leaf node
    if feature is None:
        return Node(leaf=max(labels, key=labels.count))
    # Partition data based on best split
    left_features = [sample for sample in features if sample[feature] <= split_value]  # Features in left split
    left_labels = [label for i, label in enumerate(labels) if features[i][feature] <= split_value]  # Labels in left split
    right_features = [sample for sample in features if sample[feature] > split_value]  # Features in right split
    right_labels = [label for i, label in enumerate(labels) if features[i][feature] > split_value]  # Labels in right split
    # Recursively build left and right subtrees
    left = build_tree(left_features, left_labels, max_depth - 1 if max_depth is not None else None)
    right = build_tree(right_features, right_labels, max_depth - 1 if max_depth is not None else None)
    return Node(feature=feature, split=split_value, left=left, right=right)  # Create internal node with best split

features = [[5, 1], [3, 1], [8, 2], [2, 2], [6, 3], [7, 3]]  
labels = [0, 0, 1, 1, 1, 0] 
tree = build_tree(features, labels)  
tree.print_tree()
