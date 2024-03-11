def find_pairs_with_sum(numbers, sum):
    """Find pairs of numbers in the given list that add up to the specified sum."""
    # Create a set to keep track of visited numbers
    visited = set() 
    count = 0

    # Iterate through each number in the list
    for number in numbers:
        # Calculate the difference needed to reach the target sum
        difference = sum - number

        # Check if the difference is in the visited set
        if difference in visited:
            # Increment the count and print the pairs
            count += 1
            print(f"Pairs that give sum as {sum}: ({number}, {difference})")

        # Add the current number to the visited set
        visited.add(number)

    # Return the count of pairs
    return count

if __name__ == "__main__":
    # Example usage with a list of numbers and a target sum
    numbers = [2, 7, 4, 1, 3, 6]
    target_sum = 10
    result = find_pairs_with_sum(numbers, target_sum)
    print(f"Number of pairs with sum {target_sum}: {result}")
