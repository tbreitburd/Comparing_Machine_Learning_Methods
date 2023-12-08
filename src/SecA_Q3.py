import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

# Load the data
C = pd.read_csv('data/C_MissingFeatures.csv')
C_no_nan = C.dropna()
C_no_nan = C_no_nan.drop(columns=['Unnamed: 0', 'classification'])


#--------------------------------------------------------------------------------
# (a) Summarise the missing data
#--------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part a--------------")
print('----------------------------------')

# Identify which features or data are missing, and how many are missing
C_missing_vals = C[C.isnull().any(axis=1)]
C_missing_vals = C_missing_vals.drop(columns=['Unnamed: 0', 'classification'])

print('There are 5 rows with missing values, samples: ', C_missing_vals.index.values.tolist())
print('----------------------------------')

# Get the features that are missing
C_missing_features = C_missing_vals.columns[C_missing_vals.isnull().any()]

print('The features that are missing in those rows are: ', C_missing_features.tolist())
print('----------------------------------')

#--------------------------------------------------------------------------------
# (c) Use a model-based inputation method to predict the missing values
#--------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part c--------------")
print('----------------------------------')


# We can use the same method as before, though slightly different,
# to predict the missing values, a kNN imputation approach
# Adapted from: https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/

# Define a kNN imputer, with the number of neighbours set to 28, the optimal value found in question 2
imputer = KNNImputer(n_neighbors=28, weights='distance', metric='nan_euclidean')
imputer.fit(C_no_nan)

# Impute the missing values
C_imputed = imputer.transform(C_missing_vals)

# Put the imputed values back into the dataframe
C_imputed = pd.DataFrame(C_imputed, index=C_missing_vals.index)
C_imputed.columns = C_missing_vals.columns

# Replace the missing values with the imputed values
C_new = C.fillna(C_imputed)
# Compare these new imputed values with the original values (from B)
C_missing_features = C_missing_features.tolist()

missing_index = C_missing_vals.index.values.tolist()
Original = []
Imputed = []

for i in range(len(missing_index)):
    Original.append(B[C_missing_features[i]][missing_index[i]])
    Imputed.append(C_new[C_missing_features[i]][missing_index[i]])

print('Original values: ', Original)
print('Imputed values: ', Imputed)

#--------------------------------------------------------------------------------
# (d) Implement a standardisation method to detect outliers
#--------------------------------------------------------------------------------

print('----------------------------------')
print("--------------Part d--------------")
print('----------------------------------')

# Using a Z-score standardisation method, we can detect outliers
# by looking at the values that are more than 3 standard deviations away from the mean
# We can then list those outliers 

C_d = C_new.drop(columns=['Unnamed: 0', 'classification'])

z_score = stats.zscore(C_d.iloc[:, 1:-1])
Any_outliers = np.abs(z_score) > 3
Outlier_count = Any_outliers.sum().sum()

print('Number of outliers (out of 204,000 data points): ', Outlier_count)
print('----------------------------------')

print('The outliers are: ', C_d[Any_outliers.any(axis=1)].index.values.tolist())

