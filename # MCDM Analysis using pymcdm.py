# MCDM Analysis using pymcdm

# 1. Import required libraries

import numpy as np
import pandas as pd
from pymcdm.methods import TOPSIS, SPOTIS
from pymcdm.normalizations import minmax_normalization

# 2. Prepare decision matrix and weights
alternatives = ['A1', 'A2', 'A3', 'A4']
criteria = ['Cost', 'Benefit', 'Time', 'Risk']
decision_matrix = np.array([
    [250, 0.8, 12, 0.3],
    [300, 0.6, 10, 0.5],
    [200, 0.9, 15, 0.4],
    [220, 0.7, 14, 0.2]
])
weights = np.array([0.3, 0.4, 0.2, 0.1])
criteria_types = [False, True, False, False]  # False for cost/min, True for benefit/max

# 3. Normalize decision matrix
normalized_matrix = minmax_normalization(decision_matrix, criteria_types)

# 4. Apply MCDM methods
# TOPSIS
topsis = TOPSIS()
topsis_scores = topsis(normalized_matrix, weights, criteria_types)

# SPOTIS
spotis = SPOTIS()
spotis_scores = spotis(normalized_matrix, weights, criteria_types)

# 5. Display results
results = pd.DataFrame({
    'Alternative': alternatives,
    'TOPSIS Score': topsis_scores,
    'SPOTIS Score': spotis_scores
})
results['TOPSIS Rank'] = results['TOPSIS Score'].rank(ascending=False)
results['SPOTIS Rank'] = results['SPOTIS Score'].rank(ascending=True)

print('Ranking of Alternatives:')
print(results)

# 6. Save the results to a file
results.to_csv('mcdm_results.csv', index=False)
