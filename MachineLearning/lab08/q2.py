import numpy as np
from collections import Counter

def equal_width_binning(feature_values, num_bins):
    """Perform equal width binning for continuous-valued features."""
    # Calculate the bin width based on the range of feature values
    min_value = min(feature_values)
    max_value = max(feature_values)
    bin_width = (max_value - min_value) / num_bins
    
    # Assign each feature value to the corresponding bin
    binned_values = [int((value - min_value) // bin_width) for value in feature_values]
    return binned_values

def frequency_binning(feature_values, num_bins):
    """Perform frequency binning for continuous-valued features."""
    # Count the occurrences of each feature value
    value_counts = Counter(feature_values)
    
    # Calculate bin counts based on the frequency of feature values
    bin_counts = {value: i // (len(value_counts) // num_bins) for i, value in enumerate(value_counts.keys())}
    
    # Assign each feature value to the corresponding bin
    binned_values = [bin_counts[value] for value in feature_values]
    return binned_values

def bin_continuous_feature(feature_values, num_bins, binning_type='equal_width'):
    """Bin continuous-valued features using the specified binning type."""
    if binning_type == 'equal_width':
        return equal_width_binning(feature_values, num_bins)
    elif binning_type == 'frequency':
        return frequency_binning(feature_values, num_bins)
    else:
        raise ValueError("Invalid binning type. Choose either 'equal_width' or 'frequency'.")

feature_values = [2.5, 3.5, 5.0, 6.5, 7.2, 8.0]
num_bins = 3
binning_type = 'equal_width'
binned_values = bin_continuous_feature(feature_values, num_bins, binning_type)
print("Binned Values using Equal Width Binning:", binned_values)
