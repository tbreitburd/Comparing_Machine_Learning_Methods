import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
import Plot_funcs as pf
from warnings import simplefilter

# Ignore sklearn's FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

# We use the pre-processing pipeline from the previous question to preprocess the data
# Load the data
Baseline = pd.read_csv('Data/ADS_baselineDataset.csv')
Baseline = Baseline.drop(columns=['Unnamed: 0'])

# Get the data without the labels
Baseline_no_labels = Baseline.drop(columns=['type'])
Labels = Baseline['type']


# ---------------- Preprocessing ------------------

# Any missing values?
if Baseline.isnull().values.any():
    print('Missing values in the data were found')
else:
    print('No missing values were found.')

# Get the features that have 0 variance
Variances = Baseline_no_labels.var(axis=0)
zero_var_features = Variances[Variances == 0].index.tolist()
print(
    'There are {} features with 0 variance, which will be dropped'.format(len(zero_var_features))
)
print(
    'Those rows are: ', zero_var_features
)

Baseline_no_labels = Baseline_no_labels.drop(columns=zero_var_features)
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
print(
    'Those rows are: ', Low_Var
    )

Baseline_no_labels = Baseline_no_labels.drop(columns=Low_Var.index.tolist())

# Any duplicated rows?
duplicates = Baseline_no_labels[Baseline_no_labels.duplicated(keep=False)]

print('There are {} duplicated rows'.format(len(duplicates)))

# Any highly correlated features?
# Get a correlation matrix and get the indices with high correlation
corMat = Baseline_no_labels.corr()
np.fill_diagonal(corMat.values, 0)
corMat = np.abs(corMat)
high_cor = np.where(corMat > 0.9)

# Get the unique pairs of highly correlated features
high_cor_features = []
for i in range(len(high_cor[0])):
    if high_cor[0][i] < high_cor[1][i]:
        high_cor_features.append([high_cor[0][i], high_cor[1][i]])

feature_names = Baseline_no_labels.columns.values.tolist()
high_cor_features = [[feature_names[i], feature_names[j]] for i, j in high_cor_features]
# Print the pairs of highly correlated features
print('The following pairs of features are highly correlated (> 0.9):')
for pair in high_cor_features:
    print(pair)
print('We will drop one of each pair')

# Drop one of the features in each pair
for pair in high_cor_features:
    Baseline_no_labels = Baseline_no_labels.drop(columns=[pair[1]])


# Outliers
# Using the standardisation appproach used in Q3(c)

scaler = StandardScaler()
z_score = scaler.fit_transform(Baseline_no_labels)

# Apply the threshold of 3 standard deviations

Any_outliers = np.abs(z_score) > 3
Outlier_count = Any_outliers.sum().sum()
print('Number of outliers (out of 204,000 data points): ', Outlier_count)
print('----------------------------------')
print('The outliers are in features: ',  Baseline_no_labels[Any_outliers.any(axis=1)].index.values)

# Replace the outliers with NaNs
Baseline_NaNs = Baseline_no_labels
Baseline_NaNs[Any_outliers] = np.nan

# Impute the missing values using the same method as before
imputer = KNNImputer(n_neighbors=10, weights='distance', metric='nan_euclidean')
imputer.fit(Baseline_no_labels)

# Impute the missing values
Baseline_imputed = imputer.transform(Baseline_NaNs)

# Put the imputed values back into the dataframe
Baseline_imputed = pd.DataFrame(Baseline_imputed, index=Baseline_NaNs.index)
Baseline_imputed.columns = Baseline_NaNs.columns
Baseline_no_labels = Baseline_NaNs.fillna(Baseline_imputed)

# Recalculate the number of outliers left now.
scaler = StandardScaler()
z_score = scaler.fit_transform(Baseline_no_labels)
Any_outliers = np.abs(z_score) > 3
Outlier_count = Any_outliers.sum().sum()
print('Number of outliers (out of 204,000 data points): ', Outlier_count)
print('----------------------------------')
print('The outliers are in features: ',  Baseline_no_labels[Any_outliers.any(axis=1)].index.values)

# Perform PCA on the data
pca = PCA(n_components=4)
pca.fit(Baseline_no_labels)
Baseline_no_labels_pca = pca.transform(Baseline_no_labels)
Baseline_no_labels_pca = pd.DataFrame(Baseline_no_labels_pca, columns=['PC1', 'PC2', 'PC3','PC4'])

# Scale the data
scaler = MinMaxScaler()
scaler.fit(Baseline_no_labels)
Baseline_no_labels_scaled = scaler.transform(Baseline_no_labels)

# Get the frequency of each class
print('The frequency of each class is:')
print(Baseline['type'].value_counts())

# ---------------------------------------------------------------------------------------------
# (a) Apply 2 different clustering methods
# ---------------------------------------------------------------------------------------------
print('----------------------------------------------')
print('----------------- part (a) -------------------')
print('----------------------------------------------')

# Get the silhouette scores for different cluster numbers
silhouette_scores = []
for k in range(2, 8):
    Model = KMeans(n_clusters=k, n_init = 10, random_state=42).fit(Baseline_no_labels)
    cluster_labels = Model.labels_
    silhouette_scores.append(silhouette_score(Baseline_no_labels, cluster_labels))

# Plot the silhouette scores
pf.B_Q5a_silhouette(silhouette_scores)


# -------- K-means clustering --------
# Fit the model
model = KMeans(n_clusters=3, n_init = 10, random_state=42)
model.fit(Baseline_no_labels)

# Get the predictions
KM_clustering_total = model.predict(Baseline_no_labels)

# Get those predictions in a dataframe
KM_clusters_total = pd.DataFrame(KM_clustering_total, columns=['Cluster'])

# -------- GMM clustering --------

GMM = GaussianMixture(n_components=3, random_state=42).fit(Baseline_no_labels_pca)
GMM_clustering_total = GMM.predict(Baseline_no_labels_pca)

# Get those predictions in a dataframe
GMM_clusters_total = pd.DataFrame(GMM_clustering_total, columns=['Cluster'])

# Get the label frequencies for both models
print('K-means clustering label frequencies:')
print(KM_clusters_total['Cluster'].value_counts())

print('GMM clustering label frequencies:')
print(GMM_clusters_total['Cluster'].value_counts())

# Print out a contingency table
contingency_table = pd.crosstab(KM_clusters_total['Cluster'], GMM_clusters_total['Cluster'], rownames=['K-means'], colnames=['GMM'])
print('Contingency table for the two models, for all features:')
print(contingency_table)


# ---------------------------------------------------------------------------------------------
# (b) Train a classifier on those clusterings
# ---------------------------------------------------------------------------------------------
print('----------------------------------------------')
print('----------------- part (b) -------------------')
print('----------------------------------------------')

# Using the random forest classifier from Q4, we get the feature importances
Model_KM = RandomForestClassifier(n_estimators=150, random_state=42)
Model_KM.fit(Baseline_no_labels, KM_clustering_total)

# Get the feature importances
KM_features = Baseline_no_labels.columns.values.tolist()
KM_feature_importance = Model_KM.feature_importances_

KM_feature_importances = pd.Series(KM_feature_importance, index=KM_features)

pf.B_Q4e(KM_feature_importances, 3)


# Cluster again only using the subset of features
top_40_features_KM = KM_feature_importances.nlargest(40).index.tolist()
Baseline_top_40_KM = Baseline_no_labels[top_40_features_KM]

# -------- K-means clustering --------
# Fit the model
model = KMeans(n_clusters=3, n_init = 10, random_state=42)
model.fit(Baseline_top_40_KM)

# Get the predictions
KM_clustering_t40 = model.predict(Baseline_top_40_KM)

# Get those predictions in a dataframe
KM_clusters_t40 = pd.DataFrame(KM_clustering_t40, columns=['Cluster'])


# Using the random forest classifier from Q4, we get the feature importances
Model_GMM = RandomForestClassifier(n_estimators=150, random_state=42)
Model_GMM.fit(Baseline_no_labels, GMM_clustering_total)

# Get the feature importances
GMM_features = Baseline_no_labels.columns.values.tolist()
GMM_feature_importance = Model_GMM.feature_importances_

GMM_feature_importances = pd.Series(GMM_feature_importance, index=GMM_features)

pf.B_Q4e(GMM_feature_importances, 4)


# Cluster again only using the subset of features
top_40_features_GMM = GMM_feature_importances.nlargest(40).index.tolist()
Baseline_top_40_GMM = Baseline_no_labels[top_40_features_GMM]

# -------- GMM clustering --------

#PCA transform the data
pca = PCA(n_components=4)
pca.fit(Baseline_top_40_GMM)
Baseline_top_40_pca_GMM = pca.transform(Baseline_top_40_GMM)
Baseline_top_40_pca_GMM = pd.DataFrame(Baseline_top_40_pca_GMM, columns=['PC1', 'PC2', 'PC3','PC4'])

GMM = GaussianMixture(n_components=3, random_state=42).fit(Baseline_top_40_pca_GMM)
GMM_clustering_t40 = GMM.predict(Baseline_top_40_pca_GMM)

# Get those predictions in a dataframe
GMM_clusters_t40 = pd.DataFrame(GMM_clustering_t40, columns=['Cluster'])

# Get the label frequencies for both models
print('K-means clustering label frequencies:')
print(KM_clusters_t40['Cluster'].value_counts())

print('GMM clustering label frequencies:')
print(GMM_clusters_t40['Cluster'].value_counts())

# Print out a contingency table
contingency_table = pd.crosstab(KM_clusters_t40['Cluster'], GMM_clusters_t40['Cluster'], rownames=['K-means'], colnames=['GMM'])
print('Contingency table for the two models, for the top 40 features:')
print(contingency_table)

contingency_table = pd.crosstab(KM_clusters_t40['Cluster'], KM_clusters_total['Cluster'], rownames=['Top 40'], colnames=['Total'])
print('Contingency table comparing top 40 and total for K-means:')
print(contingency_table)

contingency_table = pd.crosstab(GMM_clusters_t40['Cluster'], GMM_clusters_total['Cluster'], rownames=['Top 40'], colnames=['Total'])
print('Contingency table comparing top 40 and total for GMM:')
print(contingency_table)

# ---------------------------------------------------------------------------------------------
# (c) Plot the data
# ---------------------------------------------------------------------------------------------
print('----------------------------------------------')
print('----------------- part (c) -------------------')
print('----------------------------------------------')

# Plot the data wrt cluster membership
pf.B_Q5c_1(Baseline_no_labels_pca, KM_clustering_t40, GMM_clustering_t40, 'B_Q5c_top40_clusters')
pf.B_Q5c_1(Baseline_no_labels_pca, KM_clustering_total, GMM_clustering_total, 'B_Q5c_total_clusters')
# Get most important feature
top_fea_KM = KM_feature_importances.nlargest(1).index.tolist()[0]
Baseline_top_fea_KM = np.array(Baseline_no_labels[top_fea_KM])
print('The most important feature is: ', top_fea_KM)

top_fea_GMM = GMM_feature_importances.nlargest(1).index.tolist()[0]
Baseline_top_fea_GMM =np.array(Baseline_no_labels[top_fea_GMM])
print('The most important feature is: ', top_fea_GMM)

# Plot the data wrt most important feature
pf.B_Q5c_2(Baseline_no_labels_pca, Baseline_top_fea_GMM, Baseline_top_fea_GMM, 'B_Q5c_2')

# Get the 2nd most important feature

top_2fea_KM = KM_feature_importances.nlargest(2).index.tolist()[1]
Baseline_top_2fea_KM = np.array(Baseline_no_labels[top_2fea_KM])
print('The 2nd most important features are: ', top_2fea_KM)

top_2fea_GMM = GMM_feature_importances.nlargest(2).index.tolist()[1]
Baseline_top_2fea_GMM = np.array(Baseline_no_labels[top_2fea_GMM])
print('The 2nd most important features are: ', top_2fea_GMM)

pf.B_Q5c_2(Baseline_no_labels_pca, Baseline_top_2fea_KM, Baseline_top_2fea_GMM, 'B_Q5c_3')