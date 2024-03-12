# Function to find pairs in a list whose sum is equal to a given value
def find_pairs(input_list, target_sum):
    """
    Finds pairs in a list whose sum is equal to the target value.

    Parameters:
    - input_list (list): The input list of integers.
    - target_sum (int): The target sum value.

    Returns:
    - list: A list of pairs whose sum is equal to the target value.
    """
    pairs = []
    length = len(input_list)

    for i in range(length):
        for j in range(i, length):
            if (input_list[i] + input_list[j]) == target_sum:
                pairs.append((input_list[i], input_list[j]))

    return pairs


# Function to calculate the highest occurring character in a string
def find_highest_occurrence(input_str):
    """
    Finds the highest occurring character in a string.

    Parameters:
    - input_str (str): The input string.

    Returns:
    - tuple: A tuple containing the highest occurring character and its count.
    """
    char_count = {}

    for char in input_str:
        if char.isalpha():
            char = char.lower()
            char_count[char] = char_count.get(char, 0) + 1

    max_count = 0
    max_char = None

    for char, count in char_count.items():
        if count > max_count:
            max_count = count
            max_char = char

    return max_char, max_count


# Function to calculate the range of a list of real numbers
def calculate_range(real_numbers):
    """
    Calculates the range of a list of real numbers.

    Parameters:
    - real_numbers (list): The list of real numbers.

    Returns:
    - float or str: The range of the list if possible, else a string indicating impossibility.
    """
    if len(real_numbers) < 3:
        return "Range determination not possible"

    min_number = min(real_numbers)
    max_number = max(real_numbers)

    return max_number - min_number


# Function to calculate the power of a matrix
def power_of_matrix(matrix, power):
    """
    Calculates the power of a matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.
    - power (int): The desired power.

    Returns:
    - numpy.ndarray: The result of raising the matrix to the specified power.
    """
    import numpy as np

    return np.linalg.matrix_power(matrix, power)


# Main program
input_list = [2, 7, 4, 1, 3, 6]
sum_value = 10
result_pairs = find_pairs(input_list, sum_value)
print("Pairs whose sum is {}:".format(sum_value))
for pair in result_pairs:
    print(pair)
result_count = len(result_pairs)
print("Number of pairs:", result_count)

print("\nQuestion number 4\n")
input_str = input("Enter a string: ")
highest_char, highest_count = find_highest_occurrence(input_str)
if highest_char:
    print("Highest occurring character:", highest_char)
    print("Occurrence count:", highest_count)
else:
    print("No alphabets found in the input string.")

print("\nQuestion number 2\n")
input_list = [5, 3, 8, 1, 0, 4]
result = calculate_range(input_list)
print("Range:", result)

print("\nQuestion number 3\n")
import numpy as np
n = int(input(" Enter the matrix dimension:"))
matrix = np.random.randint(0, 10, (n, n))
power = int(input("Enter the power:"))
result_matrix = power_of_matrix(matrix, power)
print(result_matrix)
