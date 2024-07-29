import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your CSV file
file_path = '/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/MCDM/compensatory_/Python/Data.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Extract the matrix, assuming the first column is alternative names
matrix = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').dropna()

# Convert to numpy array
X = matrix.values

# Step 2: Normalize the matrix
def normalize_matrix(matrix):
    return matrix / np.sqrt((matrix ** 2).sum(axis=0))

R = normalize_matrix(X)

# Define the weights
weights = np.array([0.157991489, 0.252633783, 0.101047207, 0.212350458, 0.101047207, 0.038959503, 0.157991489, 0.141205938, 0.128086399, 0.182658054, 0.079713421])

# Step 3: Apply weights to the criteria
V = R * weights

# Step 4: Calculate the sums of weighted normalized values
S = np.sum(V, axis=1)

# Step 5: Rank the alternatives
rankings = np.argsort(-S)  # Sort in descending order

# Prepare results
alternatives = data.iloc[:, 0].values  # Use the first column for alternative names
results = pd.DataFrame({
    'Alternative': alternatives,
    'S': S,
    'Rank': rankings + 1
})

print(results)

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(alternatives, S, color='skyblue')
plt.xlabel('Alternatives')
plt.ylabel('S values')
plt.title('COPRAS Method Results')
plt.show()

# Save results to Excel
results.to_excel('copras_results.xlsx', index=False)