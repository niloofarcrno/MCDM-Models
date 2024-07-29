import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def topsis(data, weights):
    # Step 1: Normalize the decision matrix
    norm_data = data / np.sqrt((data**2).sum(axis=0))
    
    # Step 2: Multiply the normalized decision matrix by the weights
    weighted_data = norm_data * weights
    
    # Step 3: Determine the positive ideal and negative ideal solutions
    ideal_solution = np.max(weighted_data, axis=0)
    anti_ideal_solution = np.min(weighted_data, axis=0)
    
    # Step 4: Calculate the Euclidean distance from the ideal and anti-ideal solutions
    distance_to_ideal = np.sqrt(((weighted_data - ideal_solution)**2).sum(axis=1))
    distance_to_anti_ideal = np.sqrt(((weighted_data - anti_ideal_solution)**2).sum(axis=1))
    
    # Step 5: Calculate the relative closeness to the ideal solution
    relative_closeness = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)
    
    # Step 6: Rank the alternatives (highest relative closeness gets rank 1)
    ranking = np.argsort(relative_closeness)  # Ascending order
    
    return relative_closeness, ranking

# Set working directory
os.chdir("/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD_UBC/Proposal/Methodology and Results/MCDM/compensatory_/Python")

# Load the dataset
df = pd.read_csv('Data.csv')
 
# Assuming the CSV file has columns for criteria and rows for alternatives
data = df.iloc[:, 1:].values  # Assuming first column is alternative names
weights = np.array([0.157991489, 0.252633783, 0.101047207, 0.212350458, 0.101047207, 0.038959503, 0.157991489, 0.141205938, 0.128086399, 0.182658054, 0.079713421])  # Adjust weights as needed

relative_closeness, ranking = topsis(data, weights)

# Adding results to the DataFrame
df['Relative Closeness'] = relative_closeness
df['Rank'] = ranking + 1  # +1 to start ranking from 1

# Plotting the results
plt.figure(figsize=(10, 6))
plt.bar(df.iloc[:, 0], df['Relative Closeness'], color='blue')
plt.xlabel('Alternatives')
plt.ylabel('Relative Closeness')
plt.title('TOPSIS Ranking')
plt.xticks(rotation=45)
plt.tight_layout()

for i, rank in enumerate(df['Rank']):
    plt.text(i, df['Relative Closeness'][i], f'Rank {rank}', ha='center', va='bottom')

plt.show()

# Save the results to a new Excel file
df.to_excel('topsis_results.xlsx', index=False)

print(df)