import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
from warnings import simplefilter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import Plot_funcs as pf

# Ignore sklearn's FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

# Set the random seed
np.random.seed(42)

# Load the data
A = pd.read_csv('data/A_NoiseAdded.csv')

# Print out some basic info on the dataset
print('----------------------------------')
print("--------------Part a--------------") 
print('----------------------------------')
print('Shape of dataset A: ', A.shape)
print('Short A dataset description: \n', A.describe())


# Check for missing values
if A.isnull().values.any():
    print('Missing values in the data were found and removed.')
    A = A.dropna()

# ------------------------------------------------------------------------------
# (a) Density plots for the first 20 features
# ------------------------------------------------------------------------------

# Get the first 20 features
A_20 = A.iloc[:, 1:21]

# Plot the density plots
# We noticed that some features were similar, 
# so we grouped them together in the same plot (i.e. the features are not in order)

pf.A_Q1a(A_20)

# ------------------------------------------------------------------------------
# (b) Apply PCA to visualise the features in 2D
# ------------------------------------------------------------------------------    

# This time we use the full dataset, but only the features,
# so we drop the classification column 
# and the sample index column

A_fea = A.drop(['classification', 'Unnamed: 0'], axis=1)

#Then scaling the data:
scaler = StandardScaler()
scaler.fit(A_fea)
A_fea = pd.DataFrame(scaler.transform(A_fea), columns=A_fea.columns, index=A_fea.index)


# Apply PCA
pca = PCA(n_components=2)
pca_fit = pca.fit(A_fea)
A_pca = pca_fit.transform(A_fea)

# Create a dataframe with the 2 PCA features
A_pca_df = pd.DataFrame(A_pca, columns=['PC1', 'PC2'])

# Check length of the PCA data, should be same as the original data
if len(A_pca_df) != len(A_fea):
    raise RuntimeError('PCA data length does not match original data length.')


# Plot the scatter plot of the PCA visualised data
# with density contours overlaid

pf.A_Q1b(A_pca_df)


# ------------------------------------------------------------------------------
# (c) Default k-means clustering
# ------------------------------------------------------------------------------

# Partiton the data into 2 training sets of equal size

A_1, A_2 = train_test_split(A_fea, test_size=0.5, random_state=42)

# Apply k-means clustering on each of the training sets

# Train set 1
Model1 = KMeans(random_state=42).fit(A_1) # From https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
train1_cluster = Model1.labels_

# Train set 2
Model2 = KMeans(random_state=42).fit(A_2) 
train2_cluster = Model2.labels_


#Map the unused set to the clusters
clustering_1 = Model1.predict(A_2)
clustering_2 = Model2.predict(A_1)

# Get a contigency table of the clusters
contingency_table1 = pd.crosstab(clustering_2, train1_cluster, rownames=['Model 2'], colnames=['Model 1']) 
contingency_table2 = pd.crosstab(clustering_1, train2_cluster, rownames=['Model 1'], colnames=['Model 2']) 

print('----------------------------------')
print("--------------Part c--------------")
print('----------------------------------')
print("Contingency table of the clusters for A1:\n", contingency_table1)
print("Contingency table of the clusters for A2:\n", contingency_table2)


# Plot the scatter plot of the PCA visualised data
pf.cluster_plot(A_pca_df, A_1, A_1, train1_cluster, clustering_2, 8, 'A_Q1c_8clusters_A1')

pf.cluster_plot(A_pca_df, A_2, A_2, clustering_1, train2_cluster, 8, 'A_Q1c_8clusters_A2')


# ------------------------------------------------------------------------------
# (d) Silhouette score + clustering for optimal cluster number
# ------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part d--------------")
print('----------------------------------')


# Get the silhouette scores for different cluster numbers
silhouette_scores = []
for k in range(2, 8):
    Model = KMeans(n_clusters=k, random_state=42).fit(A_fea)
    cluster_labels = Model.labels_
    silhouette_scores.append(silhouette_score(A_fea, cluster_labels))

# Plot the silhouette scores
pf.A_Q1d_silhouette(silhouette_scores)

# Partiton the data into 2 training sets of equal size

A_1, A_2 = train_test_split(A_fea, test_size=0.5, random_state=42)

# Apply k-means clustering on each of the training sets

# Train set 1
Model1 = KMeans(n_clusters = 2, random_state=42).fit(A_1) # From https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
train1_cluster = Model1.labels_

# Train set 2
Model2 = KMeans(n_clusters = 2, random_state=42).fit(A_2) 
train2_cluster = Model2.labels_


#Map the unused set to the clusters
clustering_1 = Model1.predict(A_2)
clustering_2 = Model2.predict(A_1)

# Plot the scatter plot of the PCA visualised data
# with cluster color coding
pf.cluster_plot(A_pca_df, A_1, A_1, train1_cluster, clustering_2, 2, 'A_Q1d_2clusters_A1')
pf.cluster_plot(A_pca_df, A_2, A_2, train2_cluster, clustering_1, 2, 'A_Q1d_2clusters_A2')


# Print a contingency table
contingency_table = pd.crosstab(clustering_1, train2_cluster, rownames=['Model 1'], colnames=['Model 2'])
print("Contingency table of the clusters for the 2nd split:\n", contingency_table)

# ------------------------------------------------------------------------------
# (e) Identify the clusters within the PCA figure
# ------------------------------------------------------------------------------

# See the above plots for the clusters