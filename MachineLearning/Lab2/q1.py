def euclidean_distance(vector1, vector2):
    """Calculate and return the Euclidean distance between two vectors."""
    # Ensure vectors have the same dimensionality
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality")

    squared_distance = 0
    for i in range(len(vector1)):
        squared_distance += (vector1[i] - vector2[i]) ** 2
    return squared_distance ** 0.5

def manhattan_distance(vector1, vector2):
    """Calculate and return the Manhattan distance between two vectors."""
    # Ensure vectors have the same dimensionality
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality")

    distance = 0
    for i in range(len(vector1)):
        distance += abs(vector1[i] - vector2[i])
    return distance

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
