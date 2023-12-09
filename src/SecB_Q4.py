import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load the data
Baseline = pd.read_csv('Data/ADS_baselineDataset.csv')
Baseline = Baseline.drop(columns=['Unnamed: 0'])


# ------------------------------------------------------------------------------
# (b) Pre-processing the data
# ------------------------------------------------------------------------------
print('------------------------')
print('--------part (b)--------')
print('------------------------')

# Get the frequency of each class
print(Baseline['type'].value_counts())

# Get the features that have 0 variance
Variances = Baseline.var(axis=0)
zero_var_features = Variances[Variances == 0].index.tolist()
print(
    'There are {} features with 0 variance, which will be dropped'.format(len(zero_var_features))
)

Baseline = Baseline.drop(columns=zero_var_features)
Variances = Variances.drop(zero_var_features)

# Get the features with near 0 variance
Variances = Variances.sort_values(ascending=False)
print(
    'The mean variance of all featuress is {}'.format(np.mean(Variances))
    )
Low_Var = Variances[Variances < 0.001]
print(
    'There are {} features with near 0 variance (<0.001), which will be dropped'.format(len(Low_Var))
)

Baseline = Baseline.drop(columns=Low_Var.index.tolist())

# Any missing values?
if Baseline.isnull().values.any():
    print('Missing values in the data were found and removed.')
    Baseline = Baseline.dropna()
else:
    print('No missing values were found.')

# Any outliers?
# Wait to figure out 3(d,e)

# Any duplicated rows?
Baseline_no_labels = Baseline.drop(columns=['type'])
duplicates = Baseline_no_labels[Baseline_no_labels.duplicated(keep=False)]

print('There are {} duplicated rows'.format(len(duplicates)))

# Any highly correlated features?
# Get a correlation matrix and get the indices with high correlation
corMat = Baseline.corr()
np.fill_diagonal(corMat.values, 0)
corMat = np.abs(corMat)
high_cor = np.where(corMat > 0.9)

# Get the unique pairs of highly correlated features
high_cor_features = []
for i in range(len(high_cor[0])):
    if high_cor[0][i] < high_cor[1][i]:
        high_cor_features.append([high_cor[0][i], high_cor[1][i]])

feature_names = Baseline.columns.values.tolist()
high_cor_features = [[feature_names[i], feature_names[j]] for i, j in high_cor_features]
# Print the pairs of highly correlated features
print('The following pairs of features are highly correlated:')
for pair in high_cor_features:
    print(pair)
print('We will drop one of each pair')

# Drop one of the features in each pair
for pair in high_cor_features:
    Baseline = Baseline.drop(columns=[pair[1]])

# ------------------------------------------------------------------------------
# (c) Train the random forest model
# ------------------------------------------------------------------------------
print('------------------------')
print('--------part (c)--------')
print('------------------------')
    
# We have our preprocessed data, so we can now train a random forest classifier
# on the data   
Labels = Baseline['type']
# Split the data into training and test sets
Baseline_no_labels_train, Baseline_no_labels_test, Labels_train, Labels_test = train_test_split(Baseline_no_labels, Labels, test_size=0.2, random_state=42)


# Training the Model using GridSearchCV
Model = RandomForestClassifier(random_state=42)
Model.fit(Baseline_no_labels_train, Labels_train)

# Summarise the output of the model
n_estimators = Model.n_estimators

print("Number of Trees:", n_estimators)
classes = Model.classes_
print("Classes:", classes)

# Evaluate the test set classification error
score = Model.score(Baseline_no_labels_test, Labels_test)
print("Test set classification error:", 1 - score)

# ------------------------------------------------------------------------------
# (d) Optimise the model for the best number of trees
# ------------------------------------------------------------------------------
print('------------------------')
print('--------part (d)--------')
print('------------------------')

# We will use the out-of-bag error to optimise the number of trees
n_estimators = []
error_rate = []

Model_1 = RandomForestClassifier(warm_start=True, oob_score=True, random_state=42)

for n in np.linspace(15, 300, 100, dtype=int):
    Model_1.set_params(n_estimators=n)
    Model_1.fit(Baseline_no_labels_train, Labels_train)
    oob_error = 1 - Model_1.oob_score_
    error_rate.append(oob_error)
    n_estimators.append(n)

# Plot the error rate against the number of trees
plt.figure(figsize=(6, 5))
plt.plot(n_estimators, error_rate)
plt.xlabel('Number of trees')
plt.ylabel('OOB error rate')
plt.title('OOB error rate vs number of trees')
plt.grid()
plt.tight_layout()
plt.savefig('Plots/B_Q4d.png')
plt.show()

# From the plot the optimal number of trees is around 150
# Train the model with 150 trees

Model = RandomForestClassifier(n_estimators=150, random_state=42)
Model.fit(Baseline_no_labels_train, Labels_train)

# ------------------------------------------------------------------------------
# (e) Feature importance
# ------------------------------------------------------------------------------
print('------------------------')
print('--------part (e)--------')
print('------------------------')

# Get the feature importances
features = Baseline_no_labels.columns.values.tolist()
feature_importance = Model.feature_importances_

feature_importances = pd.Series(feature_importance, index=features)

# Print out the feature importances
fig, ax = plt.subplots(figsize=(5, 12))
feature_importances.nlargest(50).plot(kind='barh', ax=ax)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# Retrain the model with the top 20 features
top_20_features = feature_importances.nlargest(20).index.tolist()
Baseline_top_20 = Baseline_no_labels[top_20_features]

# Split the data into training and test sets
Baseline_top_20_train, Baseline_top_20_test, Labels_train, Labels_test = train_test_split(Baseline_top_20, Labels, test_size=0.2, random_state=42)

# Training the Model using GridSearchCV
Model_top_20 = RandomForestClassifier(n_estimators= 130)
Model_top_20.fit(Baseline_top_20_train, Labels_train)

# Get test set classification accuracy
print('The test set classification error for the model trained on the top 20 most important features are:')
print(1 - Model_top_20.score(Baseline_top_20_test, Labels_test))

# ------------------------------------------------------------------------------
# (f) Re-do (b), (c) and (e) for another supervised learning classifier
# ------------------------------------------------------------------------------
print('------------------------')
print('--------part (f)--------')
print('------------------------')



