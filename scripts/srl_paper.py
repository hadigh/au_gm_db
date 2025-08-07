import pandas as pd
import matplotlib.pyplot as plt


# --- 1. Load the data from a CSV file ---
df = pd.read_csv('../outputs/flatfile.csv')

# --- 2. Filter the DataFrame based on the criteria ---
# This filter ensures we are analyzing the same subset of data.
df = df[(df['record_id'].str.endswith('Z')) & (df['magnitude'] >= 3.75) & (df['distance'] <= 1500)]


# --- 3. Calculate and Print Key Statistics ---

# Calculate the number of total records after filtering
total_records = len(df)

# Calculate the number of unique earthquakes by counting the 'earthquake_id'
num_unique_earthquakes = df['earthquake_id'].nunique()

# Extract the network code from 'record_id' and count the unique entries
# The network code is the part before the first period (e.g., 'AU' or 'S1')
df['network'] = df['record_id'].str.split('.').str[0]
num_unique_networks = df['network'].nunique()

# Extract the station code from 'record_id' and count the unique entries
# The station code is the first two parts of the 'record_id' (e.g., 'AU.ARMA' or 'S1.AUAYR')
df['station'] = df['record_id'].apply(lambda x: '.'.join(x.split('.')[:2]))
num_unique_stations = df['station'].nunique()


# Print the calculated statistics
print("--- Earthquake Data Summary ---")
print(f"Number of records after filtering: {total_records}")
print(f"Number of unique earthquakes: {num_unique_earthquakes}")
print(f"Number of unique networks: {num_unique_networks}")
print(f"Number of unique stations: {num_unique_stations}")
print("\n--- Descriptive Statistics for Numerical Columns ---")
# Use the .describe() method to get a summary of the numerical data
print(df[['hpc', 'distance', 'magnitude', 'depth']].describe())