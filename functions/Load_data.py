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
