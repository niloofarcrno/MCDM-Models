import numpy as np
import pandas as pd
import os

def normalize_matrix(data):
    return data / np.sqrt((data**2).sum(axis=0))

def concordance_matrix(data, weights, q, p, v):
    num_alternatives, num_criteria = data.shape
    concordance = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                concordance_sum = 0
                for k in range(num_criteria):
                    if data[i, k] >= data[j, k] + p[k]:
                        concordance_sum += weights[k]
                    elif data[j, k] - p[k] < data[i, k] < data[j, k] + p[k]:
                        concordance_sum += weights[k] * ((data[i, k] - (data[j, k] - q[k])) / (p[k] - q[k]))
                concordance[i, j] = concordance_sum / np.sum(weights)

    return concordance

def discordance_matrix(data, q, p, v):
    num_alternatives, num_criteria = data.shape
    discordance = np.zeros((num_alternatives, num_alternatives, num_criteria))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                for k in range(num_criteria):
                    if data[i, k] < data[j, k]:
                        if data[j, k] - data[i, k] > v[k]:
                            discordance[i, j, k] = 1
                        elif data[j, k] - data[i, k] > p[k]:
                            discordance[i, j, k] = (data[j, k] - data[i, k] - p[k]) / (v[k] - p[k])
                        else:
                            discordance[i, j, k] = 0

    return discordance

def credibility_matrix(concordance, discordance):
    num_alternatives = concordance.shape[0]
    credibility = np.zeros((num_alternatives, num_alternatives))

    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                d_max = np.max(discordance[i, j])
                if d_max <= concordance[i, j]:
                    credibility[i, j] = concordance[i, j]
                else:
                    J = [k for k in range(discordance.shape[2]) if discordance[i, j, k] > concordance[i, j]]
                    if not J:
                        credibility[i, j] = concordance[i, j]
                    else:
                        prod = np.prod([(1 - discordance[i, j, k]) / (1 - concordance[i, j]) for k in J])
                        credibility[i, j] = concordance[i, j] * prod

    return credibility

def calculate_thresholds(data):
    std_devs = np.std(data, axis=0)
    q = (2 / 3) * std_devs   # Indifference threshold
    p = std_devs             # Preference threshold
    v = 3 * std_devs         # Veto threshold
    return q, p, v

def determine_ranking(credibility):
    phi_plus = np.sum(credibility, axis=1)
    phi_minus = np.sum(credibility, axis=0)
    phi_net = phi_plus - phi_minus

    return phi_plus, phi_minus, phi_net

# Main function to execute the ELECTRE III process
def electre_iii(data, weights):
    norm_data = normalize_matrix(data)
    weighted_data = norm_data * weights
    q, p, v = calculate_thresholds(data)
    concordance = concordance_matrix(weighted_data, weights, q, p, v)
    discordance = discordance_matrix(weighted_data, q, p, v)
    credibility = credibility_matrix(concordance, discordance)
    phi_plus, phi_minus, phi_net = determine_ranking(credibility)

    return phi_plus, phi_minus, phi_net, concordance, discordance, credibility

# Example Usage
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/MCDM/compensatory_/Python")
df = pd.read_csv('Data.csv')
data = df.iloc[:, 1:].values  # Assuming first column is alternative names
weights = np.array([0.157991489, 0.252633783, 0.101047207, 0.212350458, 0.101047207, 0.038959503, 0.157991489, 0.141205938, 0.128086399, 0.182658054, 0.079713421])

phi_plus, phi_minus, phi_net, concordance, discordance, credibility = electre_iii(data, weights)

rankings_df = pd.DataFrame({
    'Alternative': df.iloc[:, 0],
    'Concordance Degree (φ+)': phi_plus,
    'Discordance Degree (φ-)': phi_minus,
    'Net Credibility Degree (φ)': phi_net
})

rankings_df = rankings_df.sort_values(by='Net Credibility Degree (φ)', ascending=False)
rankings_df['Rank'] = range(1, len(rankings_df) + 1)

# Display the results
print("Concordance Matrix:\n", concordance)
print("Discordance Matrix:\n", discordance)
print("Credibility Matrix:\n", credibility)
print("Final Rankings:\n", rankings_df)

# Create DataFrame to store results
alt_names = df.iloc[:, 0].values
concordance_df = pd.DataFrame(concordance, index=alt_names, columns=alt_names)
discordance_df = pd.DataFrame(discordance.max(axis=2), index=alt_names, columns=alt_names)  # Use max discordance value
credibility_df = pd.DataFrame(credibility, index=alt_names, columns=alt_names)

with pd.ExcelWriter('electre3_results.xlsx') as writer:
    concordance_df.to_excel(writer, sheet_name='Concordance')
    discordance_df.to_excel(writer, sheet_name='Discordance')
    credibility_df.to_excel(writer, sheet_name='Credibility')
    rankings_df.to_excel(writer, sheet_name='Rankings', index=False)

print("Results saved to electre3_results.xlsx")