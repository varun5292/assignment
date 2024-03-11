def count_char(input_str):
    """Count the occurrences of alphabetic characters in a given string."""
    counts = {}

    # Iterate through each character in the input string
    for char in input_str:
        if char.isalpha():
            char_lower = char.lower()
            # Update the counts dictionary for each alphabetic character
            counts[char_lower] = counts.get(char_lower, 0) + 1

    # Check if there are any counts
    if counts:
        # Find the maximum count value
        max_count = max(counts.values())
        # Get the character(s) with the highest count
        highest_count_chars = [char for char, count in counts.items() if count == max_count]

        return highest_count_chars, max_count
    else:
        # Return None if there are no counts
        return None, 0

if __name__ == "__main__":
    # Get user input for the string
    input_string = input("Enter a string: ")
    # Call the count_char function to get the results
    highest_count_chars, max_count = count_char(input_string)

    # Display the results
    if highest_count_chars:
        print(f"Character(s) that are/is repeated the most: {', '.join(highest_count_chars)}")
        print(f"Occurrence count: {max_count}")
    else:
        print("Error")
