import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 0. define modules
# Define a function to assign NEHRP site class based on Vs30
def assign_nehrp_site_class(vs30):
    if vs30 > 1500:
        return 'A'
    elif 760 < vs30 <= 1500:
        return 'B'
    elif 360 < vs30 <= 760:
        return 'C'
    elif 180 < vs30 <= 360:
        return 'D'
    else:
        return 'E'

# --- 1. Load the data from a CSV file ---
df_pass = pd.read_csv('../outputs/flatfile.csv')
df_fail = pd.read_csv('../outputs/failed_flatfile.csv')
df_vs = pd.read_csv('../outputs/merged_au_wf_lst.csv')

df_fail_all_channels = pd.read_csv('../outputs/failed_flatfile.csv') #required for fail reaspn plot (see section 8)

# --- 2. add new columns, and prepare additional df
df_pass['label'] = 'Passed'
df_fail['label'] = 'Failed'
df_pass['network'] = df_pass['record_id'].str.split('.').str[0]
df_fail['network'] = df_fail['record_id'].str.split('.').str[0]
df_pass['station'] = df_pass['record_id'].apply(lambda x: '.'.join(x.split('.')[:2]))
df_fail['station'] = df_fail['record_id'].apply(lambda x: '.'.join(x.split('.')[:2]))
df = pd.concat([df_fail,df_pass])

df_vs['event_origin_cleaned'] = df_vs['event_origin'].str.replace(r'[-:\s]', '', regex=True)
df_vs['net_sta'] = df_vs['REV_NET'] + '.' + df_vs['STA']
df_pass['earthquake_id_str'] = df_pass['earthquake_id'].astype(str)
df_pass = df_pass.reset_index()
df_merged = pd.merge(
    df_pass,
    df_vs[['event_origin_cleaned', 'net_sta', 'VS', 'VS_FLAG', 'ASSCM_SC', 'KAYEN_SC',
            'ASSCM_VS', 'KAYEN_VS', 'USGS_VS', 'HVSR_VS', 'PWAVE_VS', 'ARRAY_VS']],
    left_on=['earthquake_id_str', 'station'],
    right_on=['event_origin_cleaned', 'net_sta'],
    how='left'
)
df_merged.drop(columns=['event_origin_cleaned', 'net_sta'], inplace=True)
df_merged = df_merged.drop_duplicates(subset=['record_id', 'earthquake_id']).reset_index(drop=True)
# Apply the function to the 'VS' column
df_merged['NEHRP_Site_Class'] = df_merged['VS'].apply(assign_nehrp_site_class)


# breakpoint()

# --- 3. Filter the DataFrame based on the criteria ---
# This filter ensures we are analyzing the same subset of data.
df_pass = df_pass[(df_pass['record_id'].str.endswith('Z')) & (df_pass['magnitude'] >= 3.75) & (df_pass['distance'] <= 1500)]
df_fail = df_fail[(df_fail['record_id'].str.endswith('Z')) & (df_fail['magnitude'] >= 3.75) & (df_fail['distance'] <= 1500)]
df = df[(df['record_id'].str.endswith('Z')) & (df['magnitude'] >= 3.75) & (df['distance'] <= 1500)]
df_merged = df_merged[(df_merged['record_id'].str.endswith('Z')) & (df_merged['magnitude'] >= 3.75) & (df_merged['distance'] <= 1500)]

# --- 4. Calculate and Print Key Statistics ---

# Calculate the number of total records after filtering
total_records_all = len(df_pass) + len(df_fail)
total_records_final = len(df_pass)

# Calculate the number of unique earthquakes by counting the 'earthquake_id'
num_earthquakes_all = len(pd.concat([df_pass['earthquake_id'], df_fail['earthquake_id']]).drop_duplicates().reset_index(drop=True))
num_earthquakes_final = df_pass['earthquake_id'].nunique()

# Extract the network code from 'record_id' and count the unique entries
# The network code is the part before the first period (e.g., 'AU' or 'S1')
num_networks_all = len(pd.concat([df_pass['network'], df_fail['network']]).drop_duplicates().reset_index(drop=True))
num_networks_final = df_pass['network'].nunique()
  
# Extract the start time of the database
starttime_db_all = pd.concat([df_pass['earthquake_id'], df_fail['earthquake_id']]).drop_duplicates().reset_index(drop=True).sort_values().iloc[0]
starttime_db_final = df_pass['earthquake_id'].sort_values().iloc[0]

# Extract number of stations
num_stations_all = len(pd.concat([df_pass['station'], df_fail['station']]).drop_duplicates().reset_index(drop=True))
num_stations_final = df_pass['station'].nunique()

fail_percentage = ((total_records_all - total_records_final) / total_records_all) * 100
# Print the calculated statistics
print("--- Earthquake Data Summary (passed and failed)---")
print(f"Database start time (passed and failed): {starttime_db_all}")
print(f"Number of records (passed and failed): {total_records_all}")
print(f"Number of earthquakes (passed and failed): {num_earthquakes_all}")
print(f"Number of unique networks (passed and failed): {num_networks_all}")
print(f"Number of unique stations (passed and failed): {num_stations_all}")

print("--- Earthquake Data Summary (passed) ---")
print(f"Database start time (passed): {starttime_db_final}")
print(f"Number of records (passed): {total_records_final}")
print(f"Number of earthquakes (passed): {num_earthquakes_final}")
print(f"Number of unique networks (passed): {num_networks_final}")
print(f"Number of unique stations (passed): {num_stations_final}")
print(f" Rejected percentage: {fail_percentage}")
# --- 5. Calculate and Print colocated/duplicated % ---

df['station_id'] = df.iloc[:, 2].str.split('.', n=2).str[:2].str.join('.')
df['event_id'] = df.iloc[:, 1]
total_entries = len(df)
entry_counts = df.groupby(['event_id', 'station_id']).size().reset_index(name='count')
duplicate_groups = entry_counts[entry_counts['count'] > 1]
total_duplicate_entries = duplicate_groups['count'].sum()
percentage = (total_duplicate_entries / total_entries) * 100
print("--- Colocated stations Summary (passed) ---")
print(f"Total number of entries in the DataFrame: {total_entries}")
print(f"Total number of entries with duplicate (event_id, station_id) pairs: {total_duplicate_entries}")
print(f"Percentage of duplicate entries: {percentage:.2f}%")

# --- 6. Distance-Magnitude plot % ---

g = sns.jointplot(data=df, x="distance", y="magnitude", hue="label", kind='scatter', s=50, alpha=0.5, palette=['red', 'blue'], height=6)


g.ax_joint.set_xscale('log')
g.ax_joint.grid(True, linestyle='--', alpha=0.6) 
g.savefig("../outputs/SRL_mag-dist.png", dpi=600)

# --- 7. Network, and instrument plots % ---
df_pass['instrument_type'] = df_pass['record_id'].str.split('.').str[3].str[:-1]
all_networks = sorted(df_pass['network'].unique())
all_instruments = sorted(df_pass['instrument_type'].unique())
total_records = len(df_pass)

# --- Subplot 1: Total records by network as a percentage ---
# Get the counts for each network, reindex to include all networks, fill NaNs with 0
# and convert to percentages.
network_counts = df_pass['network'].value_counts().reindex(all_networks, fill_value=0)
total_records = len(df_pass)
network_percentages = (network_counts / total_records) * 100

# --- Subplot 2: Records with distance < 100 km by network as a percentage ---
# Filter the DataFrame for records with distance less than 50 km
df_filtered = df_pass[df_pass['distance'] < 50]
# Count the number of records for each network, reindex, and convert to percentages.
network_counts_filtered = df_filtered['network'].value_counts().reindex(all_networks, fill_value=0)
total_filtered_records = len(df_filtered)
network_percentages_filtered = (network_counts_filtered / total_filtered_records) * 100

# --- Subplot 3: Total records by instrument type as a percentage ---
instrument_counts = df_pass['instrument_type'].value_counts().reindex(all_instruments, fill_value=0)
instrument_percentages = (instrument_counts / total_records) * 100

# --- Subplot 4: Records with distance < 100 km by instrument type as a percentage ---
instrument_counts_filtered = df_filtered['instrument_type'].value_counts().reindex(all_instruments, fill_value=0)
instrument_percentages_filtered = (instrument_counts_filtered / total_filtered_records) * 100

# --- subplot 5:
methods = ['ASSCM_VS', 'KAYEN_VS', 'USGS_VS', 'HVSR_VS', 'PWAVE_VS', 'ARRAY_VS']
method_counts = df_merged[methods].count()
method_percentages = (method_counts / total_records) * 100

# --- subplot 6:
class_counts = df_merged['NEHRP_Site_Class'].value_counts().reindex(['A', 'B', 'C', 'D', 'E'], fill_value=0)
class_percentages = (class_counts / class_counts.sum()) * 100

# Create a figure with two subplots side by side and shared x-axis
# Create the figure and axes
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
data_series = [
    network_percentages,
    network_percentages_filtered,
    instrument_percentages,
    instrument_percentages_filtered,
    method_percentages,
    class_percentages
]
xlabels = [
    'Network Code',
    'Network Code',
    'Instrument Code',
    'Instrument Code',
    'Vs30 Measurement Method',
    'NEHRP Site Class'
]

# Plot each subplot
for i, ax in enumerate(axes.flat):
    series = data_series[i]
    ax.bar(series.index, series.values, color='gray', edgecolor='black')
    ax.set_xlabel(xlabels[i])
    ax.set_ylabel('Percentage (%)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.text(0.007, 0.98, labels[i], transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.setp(ax.get_xticklabels(), rotation=45 if i == 4 else (90 if i in [0, 1] else 0))

plt.tight_layout()
plt.savefig("../outputs/SRL_net_inst_vs.png", dpi=600)
# --- 8. plot fail reason histogram % ---
df_fail_all_channels = df_fail_all_channels[(df_fail_all_channels['record_id'].str.endswith('Z')) & (df_fail_all_channels['magnitude'] >= 3.75) & (df_fail_all_channels['distance'] <= 1500)]
# Filter out rows where fail_reason is 'Passed'
# Define mapping rules
mapping = {
    "Colocated with BH instrument.": "Colocated",
    "Colocated with BN instrument.": "Colocated",
    "Colocated with HH instrument.": "Colocated",
    "Colocated with HN instrument.": "Colocated",
    "No instruments match entries in the colocated instrument preference list for this station.": "Colocated",
    "Failed SNR check.": "SNR_check",
    "SNR not greater than required threshold.": "SNR_check",
    "SNR not met within the required bandwidth.": "SNR_check",
    "Failed clipping check.": "Clipping check",
    "Failed noise window duration check.": "Noise window check",
    "Minimum sample rate of 20.0 not exceeded.": "Sampling rate check",
    "Zero crossing rate too low.": "Zero crossing check",
    "auto_fchp did not find an acceptable f_hp.": "Auto high-pass corner check"
}

# Filter out unwanted fail reasons
exclude_reasons = ['Passed', 'Raw data type is not integer.']
df_fail_all_channels = df_fail_all_channels[~df_fail_all_channels['fail_reason'].isin(exclude_reasons)].copy()
# Derive instrument_id by removing the last character from record_id
df_fail_all_channels['instrument_id'] = df_fail_all_channels['record_id'].str[:-1]
# Drop duplicates based on earthquake_id and instrument_id
df_fail_all_channels = df_fail_all_channels.drop_duplicates(subset=['earthquake_id', 'instrument_id'])

# Map fail reasons to simplified categories
df_fail_all_channels['mapped_fail_reason'] = df_fail_all_channels['fail_reason'].map(mapping)

# Plot histogram
plt.figure(figsize=(6, 6))
df_fail_all_channels['mapped_fail_reason'].value_counts().plot(kind='bar', color='gray', edgecolor='black')
plt.title('Frequency of Fail Reason Categories')
plt.xlabel('Fail Reason Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/SRL_fail_hist.png", dpi=600)
plt.show()

# --- 9. plot mag-distance for clippied! % ---
dc = df_fail_all_channels[df_fail_all_channels['fail_reason']=="Failed clipping check."]

dc.plot(kind='scatter', x='distance', y='magnitude')

# --- 10. correlation heatmap
selected_columns = ['hpc', 'magnitude', 'distance', 'VS']

correlation_matrix = df_merged[selected_columns].corr()

# Plot the correlation heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap between fhpc and Selected Features')
plt.savefig("../outputs/SRL_corr_mtrx.png", dpi=600)
plt.show()

# --- 11. random forest
# Create a new binary feature 'record_type' based on the channel code's first letter
# The channel code is the last three characters of the record_id
# 1 if the code starts with 'B' or 'H' (Broadband), 0 if it starts with 'S' or 'E' (Short Period)
df = df_merged
df['record_type'] = df['record_id'].str[-3].apply(
    lambda x: 1 if x in ['B', 'H'] else (0 if x in ['S', 'E'] else None)
)

# Filter the whole row if record_type is not 0 or 1
df = df.dropna(subset=['record_type'])

# Convert the record_type to integer type for multiplication
df['record_type'] = df['record_type'].astype(int)

# Define the target variable (y)
y = df['hpc']

# --- NEW: Create Interaction Features ---
# These features are a product of two existing features, allowing the model
# to capture how their effects might depend on each other.
df['magnitude_distance_interaction'] = df['magnitude'] * df['distance']
df['magnitude_record_type_interaction'] = df['magnitude'] * df['record_type']


# Select all the features for the model, including the new interaction terms.
X = df[['magnitude', 'distance', 'record_type', 'VS',
        'magnitude_distance_interaction', 'magnitude_record_type_interaction']]


# --- 3. Train a Random Forest Regressor Model ---

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Random Forest Regressor model
# We're using a random_state for reproducibility
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# --- 4. Evaluate the Model's Performance ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# --- 5. Print Results and Feature Importance ---

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print("\nFeature Importance:")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))


# --- 6. Visualize the Results ---
# The visualizations are saved to files, which can then be viewed.

# Plot 1: Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.sort_values(ascending=False), y=feature_importances.index)
plt.title('Feature Importance from Random Forest Model (with interactions)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance_with_interactions.png')  # Save the plot
plt.close()

# Plot 2: Predicted vs. Actual Values Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction') # Add a diagonal line for reference
plt.title('Predicted vs. Actual hpc values (with interactions)')
plt.xlabel('Actual hpc')
plt.ylabel('Predicted hpc')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual_with_interactions.png') # Save the plot
plt.close()




############### redundant!!!
# fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=False)
# # fig.suptitle('Seismic Data Analysis by Network and Instrument Type', fontsize=16)

# # Plotting on the first subplot (ax1)
# ax1 = axes[0,0]
# ax1.bar(network_percentages.index, network_percentages.values, color='skyblue')
# ax1.set_title('Total Records by Network')
# ax1.set_xlabel('Network ID')
# ax1.set_ylabel('Percentage of Total Records (%)')
# ax1.grid(axis='y', linestyle='--', alpha=0.7)
# ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
# plt.setp(ax1.get_xticklabels(), rotation=90)
# # Plotting on the second subplot (ax2)
# ax2 = axes[0,1]
# ax2.bar(network_percentages_filtered.index, network_percentages_filtered.values, color='salmon')
# ax2.set_title('Records with Distance < 50 km')
# ax2.set_xlabel('Network ID')
# ax2.set_ylabel('Percentage of Total Records (%)')
# ax2.grid(axis='y', linestyle='--', alpha=0.7)
# ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=12, verticalalignment='top')
# plt.setp(ax2.get_xticklabels(), rotation=90)
# # Plotting on the third subplot (Bottom-Left)
# ax3 = axes[1, 0]
# ax3.bar(instrument_percentages.index, instrument_percentages.values, color='lightgreen')
# ax3.set_title('Total Records by Instrument Type')
# ax3.set_xlabel('Instrument Type')
# ax3.set_ylabel('Percentage of Total Records (%)')
# ax3.grid(axis='y', linestyle='--', alpha=0.7)
# ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=12, verticalalignment='top')
# plt.setp(ax3.get_xticklabels(), rotation=90)

# # Plotting on the fourth subplot (Bottom-Right)
# ax4 = axes[1, 1]
# ax4.bar(instrument_percentages_filtered.index, instrument_percentages_filtered.values, color='coral')
# ax4.set_title('Records with Distance < 50 km')
# ax4.set_xlabel('Instrument Type')
# ax4.set_ylabel('Percentage of Total Records (%)')
# ax4.grid(axis='y', linestyle='--', alpha=0.7)
# ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=12, verticalalignment='top')
# plt.setp(ax4.get_xticklabels(), rotation=90)


# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap
# plt.savefig("../outputs/SRL_network_instrument.png", dpi=600)

# # --- 7. Vs plots % ---
# total_records = len(df_pass)

# # --- Second Figure: Contribution of each measurement method ---
# # Define the list of columns representing the measurement methods
# methods = ['ASSCM_SC', 'KAYEN_SC', 'ASSCM_VS', 'KAYEN_VS', 'USGS_VS', 'HVSR_VS', 'PWAVE_VS', 'ARRAY_VS']

# # Calculate the number of non-NaN values for each method
# method_counts = df_merged[methods].count()

# # Calculate the percentage of total records for which each method has a value
# method_percentages = (method_counts / total_records) * 100

# # Calculate percentage of each class
# class_counts = df_merged['NEHRP_Site_Class'].value_counts().reindex(['A', 'B', 'C', 'D', 'E'], fill_value=0)
# class_percentages = (class_counts / class_counts.sum()) * 100


# # Create the new figure and a single subplot for the histogram
# fig, ax1 = plt.subplots(figsize=(6, 6))
# ax1.bar(method_percentages.index, method_percentages.values, color='purple')
# ax1.set_title('Contribution of V_s Methods (% of Total Records)', fontsize=14)
# ax1.set_xlabel('Measurement Method')
# ax1.set_ylabel('Percentage of Records with Data (%)')
# ax1.grid(axis='y', linestyle='--', alpha=0.7)
# plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# plt.tight_layout()



# plt.figure(figsize=(8, 5))
# class_percentages.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title('NEHRP Site Class Distribution (%)')
# plt.ylabel('Percentage')
# plt.xlabel('Site Class')
# plt.xticks(rotation=0)
# plt.ylim(0, 100)
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()