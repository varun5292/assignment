def range_of_list(numbers):
    """Calculate the range of a list, which is the difference between the maximum and minimum values."""
    if len(numbers) < 3:
        return "Range determination not possible"
    
    return max(numbers) - min(numbers)

def main():
    # List with more than two elements
    list1 = [5, 3, 8, 1, 0, 4]
    result1 = range_of_list(list1)
    print(f"Range of list 1: {result1}")

    # List with less than three elements
    list2 = [1, 2]
    result2 = range_of_list(list2)
    print(f"Range of list 2: {result2}")

if __name__ == "__main__":
    main()
