import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to normalize the decision matrix using sum square method
def normalize_matrix(data):
    norm_data = data / np.sqrt(np.sum(data**2, axis=0))
    return norm_data

# Function to calculate preference index using Type IV Level Criterion
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

# Function to calculate the interval values for PROMETHEE III
def calculate_intervals(net_flow, alpha=0.5):
    phi_bar = net_flow
    sigma_phi = np.std(net_flow)
    
    X = phi_bar - alpha * sigma_phi
    Y = phi_bar + alpha * sigma_phi

    return X, Y

# Function to determine the final ranking based on intervals
def determine_final_ranking(X, Y):
    num_alternatives = len(X)
    rankings = np.zeros(num_alternatives, dtype=int)

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                if X[i] > Y[j]:
                    rankings[i] += 1
                elif X[i] <= Y[j] and X[j] <= Y[i]:
                    rankings[i] += 0.5
                    rankings[j] += 0.5

    return rankings

# Main function to execute the PROMETHEE III process
def promethee_iii(data, weights, p, q, alpha=0.5):
    norm_data = normalize_matrix(data)
    pref_matrices = preference_matrix(norm_data, weights, p, q)
    aggregated_pref_matrix = aggregated_preference_matrix(pref_matrices, weights)
    positive_flow, negative_flow, net_flow = calculate_flows(aggregated_pref_matrix)
    X, Y = calculate_intervals(net_flow, alpha)
    final_ranking = determine_final_ranking(X, Y)
    return positive_flow, negative_flow, net_flow, X, Y, final_ranking

# Example usage
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/MCDM/compensatory_/Python")
df = pd.read_csv('Data.csv')
data = df.iloc[:, 1:].values  # Assuming first column is alternative names

# Example weights and thresholds (these should be defined based on your specific use case)
weights = np.array([0.157991489, 0.252633783, 0.101047207, 0.212350458, 0.101047207, 0.038959503, 0.157991489, 0.141205938, 0.128086399, 0.182658054, 0.079713421])

def calculate_thresholds(data):
    std_devs = np.std(data, axis=0)
    q = (2 / 3) * std_devs   # Indifference threshold
    p = std_devs             # Preference threshold
    
    return q, p
p,q=calculate_thresholds(data)

positive_flow, negative_flow, net_flow, X, Y, final_ranking = promethee_iii(data, weights, p, q, alpha=0.5)

# Create a DataFrame for the rankings
rankings_df = pd.DataFrame({
    'Alternative': df.iloc[:, 0],
    'Positive Flow (φ+)': positive_flow,
    'Negative Flow (φ-)': negative_flow,
    'Net Flow (φ)': net_flow,
    'X (Lower Bound)': X,
    'Y (Upper Bound)': Y,
    'Final Ranking Score': final_ranking
})

# Sort the DataFrame by the final ranking score in descending order
rankings_df = rankings_df.sort_values(by='Final Ranking Score', ascending=False)
rankings_df['Rank'] = range(1, len(rankings_df) + 1)

# Display the results
print("Final Rankings:\n", rankings_df)

# Save the results to a new Excel file
with pd.ExcelWriter('promethee3_results.xlsx') as writer:
    rankings_df.to_excel(writer, sheet_name='Rankings', index=False)

print("Results saved to promethee3_results.xlsx")
