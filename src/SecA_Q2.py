import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedKFold                      
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np



# Load the data
B = pd.read_csv('data/B_Relabelled.csv')

# For now, we drop the missing values
B_no_nan = B.dropna()
#B_no_nan.reset_index(drop=True, inplace=True)

#--------------------------------------------------------------------------------
# (a) Summarise the frequency of the labels
#--------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part a--------------")
print('----------------------------------')
# Get the frequency of the labels
print(B_no_nan['classification'].value_counts())


#--------------------------------------------------------------------------------
# (b) Identify duplicates and address them
#--------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part b--------------")
print('----------------------------------')

# Find the duplicates
B_no_labels = B_no_nan.drop(columns=['Unnamed: 0', 'classification'])
duplicates = B_no_labels[B_no_labels.duplicated(keep=False)]

# Get the labels
Labels = B_no_nan['classification']

print('Number of duplicates: ', len(duplicates))
print("----------------------------------")
print('The duplicates are samples: ' + str(duplicates.index.values))
print("----------------------------------")

# One way to address the duplicates is to use kNN to train a classifier
# on the data without the duplicates, and then use the classifier to predict
# the labels of the duplicates. We can then use the predicted labels to check
# if they are the same as the original labels, and if not, we can drop those 
# duplicates as they are likely to be mislabelled.

# We will use the data without the duplicates to train the classifier
B_no_duplicates = B_no_labels.drop_duplicates(keep = False)

# But we first scale the data
# We fit the scaler on the data without the duplicates, for consistency
scaler = StandardScaler()
scaler.fit(B_no_duplicates)

# Scale the data
B_no_duplicates = pd.DataFrame(scaler.transform(B_no_duplicates), columns=B_no_duplicates.columns, index=B_no_duplicates.index)
duplicates = pd.DataFrame(scaler.transform(duplicates), columns=duplicates.columns, index=duplicates.index)

# Train the classifier on the data without the duplicates
# Split the data into training and test sets
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

tuneParam = {'knn__n_neighbors': list(np.logspace(np.log10(5), np.log10(200), 20).astype(int))}

# Pipeline for Preprocessing and Model Training
pipeline = Pipeline([
    ('yeojohnson', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Training the Model using GridSearchCV
grid = GridSearchCV(pipeline, param_grid=tuneParam, cv=cv, n_jobs=-1, return_train_score=True)
grid.fit(B_no_duplicates, Labels[B_no_duplicates.index.values])

# Get the best model
best_model = grid.best_estimator_

# Predict the labels of the duplicates
prediction = best_model.predict(duplicates)
duplicates['classification'] = prediction

# Check which duplicated rows have the same labels as the predicted ones
# and drop those that are not the same
dropped_duplicates = []

for i in duplicates.index.values:
    if duplicates.loc[i]['classification'] != Labels[i]:
        dropped_duplicates.append(i)
        B_no_nan = B_no_nan.drop(i)
        B_no_labels = B_no_labels.drop(i)


print(str(len(dropped_duplicates)) + ' dropped mislabelled duplicates, samples: ', dropped_duplicates)
print("----------------------------------")

# Check that the duplicates have been dropped
new_duplicates = B_no_labels[B_no_labels.duplicated(keep=False)]
print('Number of remaining duplicates: ', len(new_duplicates))
print("----------------------------------")

# The remaining duplicates are all correctly labelled so we can keep one of each
# and drop the rest
last_duplicates = B_no_labels[B_no_labels.duplicated(keep='first')]
B_no_nan = B_no_nan.drop_duplicates(keep='first')
B_no_labels = B_no_labels.drop_duplicates(keep='first')

print('Keeping only one of each duplicate.')
print("Dropping rows: ", last_duplicates.index.values)
print("----------------------------------")

# Check that the duplicates have been dropped completely
new_duplicates = B_no_labels[B_no_labels.duplicated(keep=False)]
print('Number of remaining duplicates: ', len(new_duplicates))
print("----------------------------------")

# Compare new summary of labels to the original
print('Compare this to the summary of labels: \n', B_no_nan['classification'].value_counts())









#--------------------------------------------------------------------------------
# (d) Identify and address missing values
#--------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part d--------------")
print('----------------------------------')

# Identify missing values
B_missing_vals = B[B.isnull().any(axis=1)]

print('Number of missing values: ', len(B_missing_vals))
print("----------------------------------")

# We will use the same method as in part (b) to address the missing values
# We will use the data without the missing values to train the classifier
# and then predict labels for the missing values
B_mv_no_labels = B_missing_vals.drop(columns=['Unnamed: 0', 'classification'])
B_no_labels = B_no_nan.drop(columns=['Unnamed: 0', 'classification'])


# We fit the scaler on the data without duplicates, since we now addressed them
scaler = StandardScaler()
scaler.fit(B_no_labels)
B_mv_no_labels = pd.DataFrame(scaler.transform(B_mv_no_labels), columns=B_mv_no_labels.columns, index=B_mv_no_labels.index)
B_no_labels = pd.DataFrame(scaler.transform(B_no_labels), columns=B_no_labels.columns, index=B_no_labels.index)



# -----------------------
# KNN
# -----------------------

# We use the same cross-validation as in part (b)
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# Pipeline for Preprocessing and Model Training
param_tuning_KNN = {'knn__n_neighbors': list(np.logspace(np.log10(5), np.log10(150), 20).astype(int))}
pipeline_KNN = Pipeline([
    ('yeojohnson', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Training the Model using GridSearchCV
grid_KNN = GridSearchCV(pipeline_KNN, param_grid=param_tuning_KNN, cv=cv, return_train_score=True)
grid_KNN.fit(B_no_labels, Labels[B_no_labels.index.values])

# Get the best model
best_model_KNN = grid_KNN.best_estimator_

# Predict the labels of the samples whose labels are missing
prediction_KNN = best_model_KNN.predict(B_mv_no_labels)


# Assess what k value was optimal
results = pd.DataFrame(grid_KNN.cv_results_)

# Create a summary table of the results
k_values = results['param_knn__n_neighbors']
mean_test_scores = results['mean_test_score']
std_test_scores = results['std_test_score']

summary_table = pd.DataFrame({
    'k_value': k_values,
    'mean_accuracy': mean_test_scores,
    'std_accuracy': std_test_scores
})

print('Summary table of k values (useful for q3): \n')
print(summary_table)
print("----------------------------------")


# Train the classifier on the data without the missing values (and no duplicates)
# But we need to stratify the data to ensure that the labels are represented
# in the training set in the same proportion as in the original data
n_splits = 2
test_size = 0.3
stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

# Startify sample the data
for train_index, test_index in stratified_split.split(B_no_labels, B_no_nan['classification']):
    B_train = B_no_labels.iloc[train_index]
    Label_train = B_no_nan['classification'].iloc[train_index]
    B_test = B_no_labels.iloc[test_index]
    Label_test = B_no_nan['classification'].iloc[test_index]


# -----------------------
# Multinomial Logistic Regression
# -----------------------
# Training the Model using CV
cv_MLR = LogisticRegressionCV(cv=5, multi_class='multinomial', random_state=42).fit(B_train, Label_train)

# Get the accuracy of the model
accuracy = cv_MLR.score(B_test, Label_test)
print('Accuracy of MLR: ', accuracy)   

# Predict the labels of the samples whose labels are missing
prediction_MLR = cv_MLR.predict(B_mv_no_labels)

# Add the predicted labels to the dataframe
B_mv_no_labels['classification'] = prediction_MLR

# And now add those to the original dataframe
B_no_nan = pd.concat([B_no_nan, B_mv_no_labels])

# Finally, we compare the summary of labels to the original
print('Compare this to the summary of labels: \n', B_no_nan['classification'].value_counts())

