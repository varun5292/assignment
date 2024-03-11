def label_encoding(categories):
    """Encode categorical data with integer labels and return the mapping."""
    encoded_mapping = {}
    for index, category in enumerate(categories):
        encoded_mapping[category] = index
    return encoded_mapping

if __name__ == "__main__":
    # Get user input for categorical data
    categorical_data_str = input("Enter categorical data (comma-separated values): ")
    categorical_data = categorical_data_str.split(',')

    # Perform label encoding
    label_encoded_mapping = label_encoding(categorical_data)

    # Display original and label-encoded data
    print("Original Categorical Data:", categorical_data)
    print("Label Encoded Mapping:", label_encoded_mapping)
