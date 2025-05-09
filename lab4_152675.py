# MCDM Analysis of Students' Late Submissions - Fun Version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymcdm.methods import TOPSIS, SPOTIS, VIKOR, PROMETHEE_II
from pymcdm.normalizations import minmax_normalization

# Define decision matrix and weights
alternatives = ['Student A', 'Student B', 'Student C', 'Student W']
decision_matrix = np.array([
    [5, 3, 8, 6],     # Student A: moderately late, medium dramatyzm, good quality, decent contribution
    [15, 7, 4, 7],    # Student B: quite late, high dramatyzm, average quality, good contribution
    [10, 5, 6, 5],    # Student C: late, some dramatyzm, decent quality, average contribution
    [30, 10, 5, 10]   # Student W: very late, very dramatic, moderate quality, top contribution
])
weights = np.array([0.4, 0.3, 0.2, 0.1])
criteria_types = [-1, -1, 1, 1]  # Minimize lateness and dramatyzm, maximize quality and contribution

# Normalize decision matrix
normalized_matrix = minmax_normalization(decision_matrix, criteria_types)

# Initialize MCDM methods
topsis = TOPSIS()
spotis = SPOTIS(np.stack((np.min(decision_matrix, axis=0), np.max(decision_matrix, axis=0)), axis=1))
vikor = VIKOR()
promethee = PROMETHEE_II('usual')

# Calculate scores for each method
topsis_scores = topsis(normalized_matrix, weights, criteria_types)
spotis_scores = spotis(decision_matrix, weights, criteria_types)
vikor_scores = vikor(decision_matrix, weights, criteria_types)
promethee_scores = promethee(decision_matrix, weights, criteria_types)

# Collect results in a DataFrame
results = pd.DataFrame({
    'Alternative': alternatives,
    'TOPSIS': topsis_scores,
    'SPOTIS': spotis_scores,
    'VIKOR': vikor_scores,
    'PROMETHEE': promethee_scores
})
results['TOPSIS Rank'] = results['TOPSIS'].rank(ascending=False)
results['SPOTIS Rank'] = results['SPOTIS'].rank(ascending=True)
results['VIKOR Rank'] = results['VIKOR'].rank(ascending=True)
results['PROMETHEE Rank'] = results['PROMETHEE'].rank(ascending=False)

print('MCDM Analysis Results of Student Submissions:')
print(results)

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Scores plot
axs[0].bar(alternatives, results['TOPSIS'], label='TOPSIS', alpha=0.7)
axs[0].bar(alternatives, results['SPOTIS'], label='SPOTIS', alpha=0.7)
axs[0].bar(alternatives, results['VIKOR'], label='VIKOR', alpha=0.7)
axs[0].bar(alternatives, results['PROMETHEE'], label='PROMETHEE', alpha=0.7)
axs[0].set_title('Scores of Late Submission Justifications')
axs[0].set_ylabel('Score')
axs[0].legend()

# Ranking plot
axs[1].bar(alternatives, results['TOPSIS Rank'], label='TOPSIS Rank')
axs[1].bar(alternatives, results['SPOTIS Rank'], label='SPOTIS Rank')
axs[1].bar(alternatives, results['VIKOR Rank'], label='VIKOR Rank')
axs[1].bar(alternatives, results['PROMETHEE Rank'], label='PROMETHEE Rank')
axs[1].set_title('Ranking Comparison of Student Submissions')
axs[1].set_ylabel('Rank')
axs[1].invert_yaxis()
axs[1].legend()

plt.tight_layout()
plt.show()
