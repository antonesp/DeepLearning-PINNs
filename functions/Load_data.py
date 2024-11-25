#%%
import pandas as pd

# Define a custom function to parse the CSV file
def custom_csv_parser(file_path, Training=True):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize dictionaries to store different sections
    data = {}
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        if not line:  # Skip empty lines
            continue
        
        if line.startswith('Meal_size') or line.startswith('Steady_insulin') or line.startswith('Bolus'):
            key, value, _ = line.split(',')
            data[key] = float(value)
        elif line.startswith('D1') or line.startswith('D2'):
            parts = line.split(',')
            key = parts[0]
            values = [float(x) for x in parts[2:] if x]  # Filter out empty strings
            data[key] = values
        elif line.startswith('I_sc') or line.startswith('I_p') or line.startswith('I_eff') or line.startswith('G') or line.startswith('G_sc'):
            parts = line.split(',')
            key = parts[0]
            values = [float(x) for x in parts[1:] if x]  # Filter out empty strings
            data[key] = values
        elif line.startswith('tau1') or line.startswith('tau2') or line.startswith('Ci') or line.startswith('p2') or line.startswith('Si') or line.startswith('GEZI') or line.startswith('EGP0') or line.startswith('Vg') or line.startswith('taum') or line.startswith('tausc'):
            if not Training:
                if line.startswith('Si'):
                    pass
                else:
                    key, value, _ = line.split(',')
                    data[key] = float(value)
            else:
                key, value, _ = line.split(',')
                data[key] = float(value)

    if Training:
        print('Training data loaded successfully!')
    else:
        print('Test data loaded successfully!')
    return data

# file_path = '../Patient.csv'
# # Load the training data
# train_data = custom_csv_parser(file_path = '../Patient.csv')
# # Load the test data
# test_data = custom_csv_parser(file_path = '../Patient.csv', Training=False)

import csv


def custom_csv_parser2(file_path):
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
import numpy as np

def data_split(data, seed = 42, train_frac = 0.15):
    # Set the random seed
    np.random.seed(seed)
    step_time = 0.1
    ts = [step_time*i for i in range(len(data['D1']))]
    # Split dataset into training set and test set
    train_size = int(len(data['D1']) * train_frac)
    train_idx = np.random.choice(len(data['D1']), train_size, replace=False).astype(int)
    test_idx = np.array([i for i in range(len(data['D1'])) if i not in train_idx]).astype(int)
    # Sort the indices
    train_idx = np.sort(train_idx).astype(int)
    test_idx = np.sort(test_idx).astype(int)

    # Split
    X_train, X_test = {}, {}
    for key in data.keys():
        data[key] = np.array(data[key])
        X_train[key] = data[key][train_idx]
        X_test[key] = data[key][test_idx]
    ts_train = np.array(ts)[train_idx]
    ts_test = np.array(ts)[test_idx]

    return X_train, X_test, ts_train, ts_test, ts