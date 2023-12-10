import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS

# We use the pre-processing pipeline from the previous question to preprocess the data
# Load the data
Baseline = pd.read_csv('Data/ADS_baselineDataset.csv')
Baseline = Baseline.drop(columns=['Unnamed: 0'])

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


# ---------------------------------------------------------------------------------------------
# (a) Apply 2 different clustering methods
# ---------------------------------------------------------------------------------------------
print('----------------------------------------------')
print('----------------- part (a) -------------------')
print('----------------------------------------------')
# Get the data without the labels
Baseline_no_labels = Baseline.drop(columns=['type'])

# Scale the data, using a robust scaler
scaler = RobustScaler()
scaler.fit(Baseline_no_labels)
Baseline_no_labels = pd.DataFrame(scaler.transform(Baseline_no_labels), columns=Baseline_no_labels.columns, index=Baseline_no_labels.index)


# K-means

# Get the silhouette scores for different cluster numbers
silhouette_scores = []
for k in range(2, 8):
    Model = KMeans(n_clusters=k,n_init = 10).fit(Baseline_no_labels)
    cluster_labels = Model.labels_
    silhouette_scores.append(silhouette_score(Baseline_no_labels, cluster_labels))

# Plot the silhouette scores
plt.figure(figsize=(6, 5))
plt.plot(range(2, 8), silhouette_scores, marker='o')
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for different cluster numbers')
plt.tight_layout()
plt.savefig('Plots/B_Q5a_silhouette.png')
plt.show()

# We choose a cluster number of 3
optimal_k = 3
KM_Model = KMeans(n_clusters=3, init ="k-means++", random_state=42).fit(Baseline_no_labels)
KM_cluster_labels = KM_Model.labels_

# Get some specific K-means cluster outputs
# Get the cluster centers
KM_cluster_centers = KM_Model.cluster_centers_
print('The K-means cluster centers are:')
print(KM_cluster_centers)
print('----------------------------------------------')


# Get cluster labels and their frequency
print('The K-means cluster labels and their frequencies are:')
print(pd.Series(KM_cluster_labels).value_counts())
print('----------------------------------------------')

# Get the inertia
print('The K-means inertia is:')
print(KM_Model.inertia_)
print('----------------------------------------------')


# Spectral clustering
Spectral_clust = SpectralClustering(n_clusters=3, affinity= 'nearest_neighbors', n_neighbors= 40, random_state=42).fit(Baseline_no_labels)

