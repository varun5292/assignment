#1
def find(list, length, summ):
    pairs = []
    for i in range(length):
        for j in range(i, length):
            if (list[i] + list[j]) == summ:
                pairs.append((list[i], list[j]))
    return pairs

list = [2, 7, 4, 1, 3, 6]
sum_value = 10
result_pairs = find(list, len(list), sum_value)
print("Pairs whose sum is {}:".format(sum_value))
for pair in result_pairs:
    print(pair)
result_count = len(result_pairs)
print("Number of pairs:", result_count)

#4
print("\nQuestion number 4\n")
def occurrence(str):
    char_count = {}
    for char in str:
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

str = input("Enter a string: ")
highest_char, highest_count = occurrence(str)
if highest_char:
    print("Highest occurring character:", highest_char)
    print("Occurrence count:", highest_count)
else:
    print("No alphabets found in the input string.")

#2
print("\nQuestion number 2\n")
def calculate_range(real_numbers):
    if len(real_numbers) < 3:
        return "Range determination not possible"
    
    min_number = min(real_numbers)
    max_number = max(real_numbers)
    
    return max_number - min_number
input_list = [5, 3, 8, 1, 0, 4]
result = calculate_range(input_list)
print("Range:", result)

#3
print("\nQuestion number 3\n")
import numpy as np
def power_of_matrix(matrix, m):
    return np.linalg.matrix_power(matrix, m)
 
n = int(input(" Enter the matrix dimension:"))
matrix = np.random.randint(0,10,(n,n))
m = int(input("Enter the power:"))
result = power_of_matrix(matrix, m)
print(result)