def matrix_multiplication(matrix, exponent):
    """ Perform matrix multiplication with an exponent. """
    # Check if the input matrix is square
    if not is_square(matrix):
        raise ValueError("Input matrix must be square")

    # Initialize the result matrix with zeros
    result = [[0] * len(matrix) for _ in range(len(matrix))]

    # Perform matrix multiplication
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            for k in range(len(matrix)):
                # Multiply each element and raise to the specified exponent
                result[i][j] += matrix[i][k] * matrix[k][j] ** exponent

    return result

def is_square(matrix):
    """Check if a matrix is square."""
    return all(len(row) == len(matrix) for row in matrix)

def matrix_inputing():
    """Take user input to create a square matrix."""
    size = int(input("Enter the size for the square matrix: "))
    # Create a square matrix based on user input
    matrix = [[int(val) for val in input(f"Enter values for row {i + 1} ").split()] for i in range(size)]
    return matrix

def power_matrix():
    """Take user input for the exponent to raise each element of the matrix to."""
    return int(input("Enter a number to multiply the matrix: "))

if __name__ == "__main__":
    try:
        # Get user input for the matrix and exponent
        matrixA = matrix_inputing()
        exponent_m = power_matrix()

        # Perform matrix multiplication and display the result
        result_matrix = matrix_multiplication(matrixA, exponent_m)
        print(f"Resultant Matrix:\n{result_matrix}")

    except ValueError as e:
        # Handle ValueError if the input matrix is not square
        print(f"Error: {e}")
