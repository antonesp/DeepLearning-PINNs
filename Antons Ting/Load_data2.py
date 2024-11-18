import pandas as pd
import csv
import os
import sys

def custom_csv_parser(file_path):
    # Open the file and parse it as a CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        
        headers = next(reader)  # Read header row for the keys
        
        # Initialize a dictionary with keys from headers and empty lists for values
        data = {key: [] for key in headers}

        # Populate the dictionary with values
        for row in reader:
            for key, value in zip(headers, row):
                data[key].append(float(value))  # Convert values to float

    return data