import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import Plot_funcs as pf
from warnings import simplefilter

# Ignore sklearn's FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

# Load the data
C = pd.read_csv('data/C_MissingFeatures.csv')
C_no_nan = C.dropna()
C_no_nan = C_no_nan.drop(columns=['Unnamed: 0', 'classification'])

# Load the B dataset to compare the imputed values with the original values in (c)
B = pd.read_csv('data/B_Relabelled.csv')
#--------------------------------------------------------------------------------
# (a) Summarise the missing data
#--------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part a--------------")
print('----------------------------------')

# Identify which features or data are missing, and how many are missing
C_missing_vals = C[C.isnull().any(axis=1)]
C_missing_vals = C_missing_vals.drop(columns=['Unnamed: 0', 'classification'])

print('There are {} rows with missing values, samples: '.format(len(C_missing_vals)) , C_missing_vals.index.values.tolist())
print('----------------------------------')

# Get the features that are missing
for i in C_missing_vals.index.values.tolist():
    C_missing_features = C_missing_vals.columns[C_missing_vals.isnull().any()]

    print('The features that are missing in row {} are: '.format(i), C_missing_features.tolist())
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
imputer = KNNImputer(n_neighbors=10, weights='distance', metric='nan_euclidean')
imputer.fit(C_no_nan)

# Impute the missing values
C_imputed = imputer.transform(C_missing_vals)

# Put the imputed values back into the dataframe
C_imputed = pd.DataFrame(C_imputed, index=C_missing_vals.index)
C_imputed.columns = C_missing_vals.columns

# Replace the missing values with the imputed values
C_new = C.fillna(C_imputed)
# Compare these new imputed values with the original values (from B)

# Compare original and new distributions

pf.A_Q3c(C_new, C_no_nan, 1)



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

scaler = StandardScaler()
z_score = scaler.fit_transform(C_d)

# Apply the threshold of 3 standard deviations

Any_outliers = np.abs(z_score) > 3
Outlier_count = Any_outliers.sum().sum()

print('Number of outliers (out of 204,000 data points): ', Outlier_count)
print('----------------------------------')

print('The outliers are in features: ',  C_d[Any_outliers.any(axis=1)].index.values)

#--------------------------------------------------------------------------------
# (e) Deal with the outliers
#--------------------------------------------------------------------------------

# Set the outliers to NaN
C_e_nan = C_d
C_e_nan[Any_outliers] = np.nan

# Impute the missing values using the same method as before
imputer = KNNImputer(n_neighbors=10, weights='distance', metric='nan_euclidean')
imputer.fit(C_d)

# Impute the missing values
C_imputed = imputer.transform(C_e_nan)

# Put the imputed values back into the dataframe
C_imputed = pd.DataFrame(C_imputed, index=C_e_nan.index)
C_imputed.columns = C_e_nan.columns


C_new_e = C_e_nan.fillna(C_imputed)

# Get how many outliers remain
scaler = StandardScaler()
scaler.fit(C_d)
z_score = scaler.transform(C_new_e)
# Apply the threshold of 3 standard deviations
Any_outliers = np.abs(z_score) > 3
Outlier_count = Any_outliers.sum().sum()
print('Number of outliers left after correction(out of 204,000 data points): ', Outlier_count)

# Compare the distributions of the original and new data
pf.A_Q3c(C_new_e, C_d, 2)