import os

def parse_prettytable(file_path):
    """Parses a PrettyTable formatted string from a .txt file and extracts the Relative Error column."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # Find the start of the table (header separator line starts with '+')
    table_start = next(i for i, line in enumerate(lines) if line.startswith('+'))
    # Skip the header and the separator line
    data_lines = lines[table_start + 2 : -1]  # Exclude the last border line
    
    # Extract the relative error values from each row
    relative_errors = []
    for line in data_lines:
        columns = line.split('|')
        if len(columns) < 5:  # Ignore malformed rows
            continue
        relative_error_str = columns[-2].strip()  # Second last column is "Relative Error"
        relative_errors.append(float(relative_error_str))
    
    return relative_errors

def find_folder_with_min_relative_error(base_directory):
    min_sum = float('inf')
    min_folder = None

    for root, dirs, files in os.walk(base_directory):
        if "table.txt" in files:
            file_path = os.path.join(root, "table.txt")
            try:
                relative_errors = parse_prettytable(file_path)
                relative_error_sum = sum(relative_errors)
                if relative_error_sum < min_sum:
                    min_sum = relative_error_sum
                    min_folder = root
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return min_folder, min_sum

# Example usage
if __name__ == "__main__":
    base_directory = os.getcwd()  # Replace with your root directory
    folder, relative_error_sum = find_folder_with_min_relative_error(base_directory)
    if folder:
        relative_path = folder.split("/Project", 1)[-1].lstrip("/")
        print(f"The folder with the smallest relative error sum is: {relative_path}")
        print(f"The smallest sum of relative errors is: {relative_error_sum:.4f}")
        sum_rel = (40.8 + 46.8 + 32.7 + 36.1 + 12.9 + 435.5 + 55.3 + 5.2 + 5.4) / 100
        print(f"From Bagterp the sum of relative errors to compare to is: {sum_rel:.4f}")
        improvement = (sum_rel - relative_error_sum) / sum_rel * 100
        if improvement < 0:
            print(f"Which means we are {-improvement:.2f}% worse than Bagterp" )
        else:
            print(f"Which means we are {improvement:.2f}% better than Bagterp" )
    else:
        print("No valid table.txt files were found.")
