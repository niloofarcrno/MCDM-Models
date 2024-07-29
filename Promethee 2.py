import numpy as np
import pandas as pd
import os

# Function to normalize the decision matrix
def normalize_matrix(data):
    return data / np.sqrt((data**2).sum(axis=0))

# Function to calculate preference index
def preference_index(a, b, p, q):
    diff = a - b
    if diff <= q:
        return 0
    elif q < diff < p:
        return (diff - q) / (p - q)
    else:
        return 1

# Function to calculate the preference matrix for all criteria
def preference_matrix(data, weights, p, q):
    num_alternatives, num_criteria = data.shape
    pref_matrices = np.zeros((num_alternatives, num_alternatives, num_criteria))

    for k in range(num_criteria):
        for i in range(num_alternatives):
            for j in range(num_alternatives):
                if i != j:
                    pref_matrices[i, j, k] = preference_index(data[i, k], data[j, k], p[k], q[k])
    
    return pref_matrices

# Function to calculate the aggregated preference matrix
def aggregated_preference_matrix(pref_matrices, weights):
    num_alternatives = pref_matrices.shape[0]
    aggregated_pref_matrix = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                aggregated_pref_matrix[i, j] = np.sum(pref_matrices[i, j, :] * weights)
    
    return aggregated_pref_matrix

# Function to calculate the positive, negative, and net flows
def calculate_flows(aggregated_pref_matrix):
    positive_flow = np.sum(aggregated_pref_matrix, axis=1) / (aggregated_pref_matrix.shape[0] - 1)
    negative_flow = np.sum(aggregated_pref_matrix, axis=0) / (aggregated_pref_matrix.shape[0] - 1)
    net_flow = positive_flow - negative_flow
    return positive_flow, negative_flow, net_flow

# Main function to execute the PROMETHEE II process
def promethee_ii(data, weights, p, q):
    norm_data = normalize_matrix(data)
    pref_matrices = preference_matrix(norm_data, weights, p, q)
    aggregated_pref_matrix = aggregated_preference_matrix(pref_matrices, weights)
    positive_flow, negative_flow, net_flow = calculate_flows(aggregated_pref_matrix)
    return positive_flow, negative_flow, net_flow

# Example Usage
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/MCDM/compensatory_/Python")
df = pd.read_csv('Data.csv')
data = df.iloc[:, 1:].values  # Assuming first column is alternative names


# Example weights and thresholds (these should be defined based on your specific use case)
weights = np.array([0.157991489, 0.252633783, 0.101047207, 0.212350458, 0.101047207, 0.038959503, 0.157991489, 0.141205938, 0.128086399, 0.182658054, 0.079713421])

p = np.array([0.3] * data.shape[1])  # Preference thresholds for each criterion
q = np.array([0.1] * data.shape[1])  # Indifference thresholds for each criterion

positive_flow, negative_flow, net_flow = promethee_ii(data, weights, p, q)

# Create a DataFrame for the rankings
rankings_df = pd.DataFrame({
    'Alternative': df.iloc[:, 0],
    'Positive Flow (φ+)': positive_flow,
    'Negative Flow (φ-)': negative_flow,
    'Net Flow (φ)': net_flow
})

# Sort the DataFrame by the net flow in descending order
rankings_df = rankings_df.sort_values(by='Net Flow (φ)', ascending=False)
rankings_df['Rank'] = range(1, len(rankings_df) + 1)

# Display the results
print("Final Rankings:\n", rankings_df)

# Save the results to a new Excel file
with pd.ExcelWriter('promethee2_results.xlsx') as writer:
    rankings_df.to_excel(writer, sheet_name='Rankings', index=False)

print("Results saved to promethee2_results.xlsx")