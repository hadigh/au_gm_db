import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# To ensure plots are displayed in the output
# Note: This is not necessary in a jupyter notebook environment.
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Load the data from a CSV file ---
df = pd.read_csv('../outputs/flatfile.csv')

# --- 2. Data Preprocessing and Feature Engineering ---
# Filter to keep only records where record_id ends with 'Z'
# df = df[df['record_id'].str.endswith('Z')]
df = df[(df['record_id'].str.endswith('Z')) & (df['magnitude'] >= 3.75) & (df['distance'] <= 1500)]
# Create a new binary feature 'record_type' based on the channel code's first letter
# The channel code is the last three characters of the record_id
# 1 if the code starts with 'B' or 'H' (Broadband), 0 if it starts with 'S' or 'E' (Short Period)
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
X = df[['magnitude', 'distance', 'record_type',
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


selected_columns = ['hpc', 'magnitude', 'distance']

correlation_matrix = df[selected_columns].corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap between fhpc and Selected Features')
plt.show()
