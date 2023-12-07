import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from warnings import simplefilter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Ignore sklearn's FutureWarnings
simplefilter(action='ignore', category=FutureWarning)


# Load the data
A = pd.read_csv('data/A_NoiseAdded.csv')

# Check for missing values
if A.isnull().values.any():
    print('Missing values in the data were found and removed.')
    A = A.dropna()

# Check for outliers, using Z-score and a threshold of 3, as standard


# ------------------------------------------------------------------------------
# (a) Density plots for the first 20 features
# ------------------------------------------------------------------------------

# Get the first 20 features
A_20 = A.iloc[:, 1:21]

# Plot the density plots
# We noticed that some features were similar, 
# so we grouped them together in the same plot (i.e. the features are not in order)

fig, axes = plt.subplots(2,3, figsize=(15, 10))


sns.kdeplot(A_20['Fea11'], ax=axes[0,0], label='Fea11', legend=True)
sns.kdeplot(A_20['Fea14'], ax=axes[0,0], label='Fea14', legend=True)
sns.kdeplot(A_20['Fea19'], ax=axes[0,0], label='Fea19', legend=True)
axes[0,0].set_title('Density plots of Fea11-14-19')
axes[0,0].set_ylabel('Density')
axes[0,0].set_xlabel('Feature value')
axes[0,0].set_xlim(-2,8)
axes[0,0].legend()

sns.kdeplot(A_20['Fea13'], ax=axes[0,1], label='Fea13', legend=True)
sns.kdeplot(A_20['Fea2'], ax=axes[0,1], label='Fea2', legend=True)
sns.kdeplot(A_20['Fea17'], ax=axes[0,1], label='Fea17', legend=True)
axes[0,1].set_title('Density plots of Fea2-13-17')
axes[0,1].set_xlabel('Feature value')
axes[0,1].set_ylabel(None)
axes[0,1].set_xlim(-2,8)
axes[0,1].legend()

sns.kdeplot(A_20['Fea18'], ax=axes[0,2], label='Fea18', legend=True)
sns.kdeplot(A_20['Fea20'], ax=axes[0,2], label='Fea20', legend=True)
sns.kdeplot(A_20['Fea5'], ax=axes[0,2], label='Fea5', legend=True)
axes[0,2].set_title('Density plots of Fea5-18-20')
axes[0,2].set_xlabel('Feature value')
axes[0,2].set_ylabel(None)
axes[0,2].set_xlim(-2,8)
axes[0,2].legend()


sns.kdeplot(A_20['Fea1'], ax=axes[1,0], label='Fea1', legend=True)
sns.kdeplot(A_20['Fea3'], ax=axes[1,0], label='Fea3', legend=True)
sns.kdeplot(A_20['Fea4'], ax=axes[1,0], label='Fea4', legend=True)
axes[1,0].set_title('Density plots of Fea1-3-4')
axes[1,0].set_xlabel('Feature value')
axes[1,0].set_ylabel('Density')
axes[1,0].set_xlim(-2,8)
axes[1,0].legend()


sns.kdeplot(A_20['Fea6'], ax=axes[1,1], label='Fea6', legend=True)
sns.kdeplot(A_20['Fea7'], ax=axes[1,1], label='Fea7', legend=True)
sns.kdeplot(A_20['Fea8'], ax=axes[1,1], label='Fea8', legend=True)
sns.kdeplot(A_20['Fea9'], ax=axes[1,1], label='Fea9', legend=True)
axes[1,1].set_title('Density plots of Fea6-9')
axes[1,1].set_xlabel('Feature value')
axes[1,1].set_ylabel(None)
axes[1,1].set_xlim(-2,8)
axes[1,1].legend()


sns.kdeplot(A_20['Fea10'], ax=axes[1,2], label='Fea10', legend=True)
sns.kdeplot(A_20['Fea12'], ax=axes[1,2], label='Fea12', legend=True)
sns.kdeplot(A_20['Fea15'], ax=axes[1,2], label='Fea15', legend=True)
sns.kdeplot(A_20['Fea16'], ax=axes[1,2], label='Fea16', legend=True)
axes[1,2].set_title('Density plots of Fea10-12-15-16')
axes[1,2].set_xlabel('Feature value')
axes[1,2].set_ylabel(None)
axes[1,2].set_xlim(-2,8)
axes[1,2].legend()

plt.tight_layout()
plt.savefig('Plots/A_Q1a.png')
plt.show()


# ------------------------------------------------------------------------------
# (b) Apply PCA to visualise the features in 2D
# ------------------------------------------------------------------------------    

# This time we use the full dataset, but only the features,
# so we drop the classification column 
# and the sample index column

A_fea = A.drop(['classification', 'Unnamed: 0'], axis=1)


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
plt.figure(figsize=(6, 5))

# Scatter plot
sns.scatterplot(x=A_pca_df['PC1'], y=A_pca_df['PC2'], color='blue', alpha=0.5)

# Density map overlay
sns.kdeplot(x=A_pca_df['PC1'], y=A_pca_df['PC2'], cmap="inferno", levels=5, thresh=0.05)

plt.grid()
plt.title('Scatter Plot with contours of A_pca')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('Plots/A_Q1b.png')
plt.show()



# ------------------------------------------------------------------------------
# (c) Default k-means clustering
# ------------------------------------------------------------------------------

# Partiton the data into 2 training sets of equal size

A_1, A_2 = train_test_split(A_fea, test_size=0.5, random_state=42)

# Apply k-means clustering on each of the training sets

# Train set 1
Model1 = KMeans().fit(A_1) # From https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
train1_cluster = Model1.labels_

# Train set 2
Model2 = KMeans().fit(A_2) 
train2_cluster = Model2.labels_


#Map the unused set to the clusters
clustering_1 = Model1.predict(A_2)
clustering_2 = Model2.predict(A_1)

# Get a contigency table of the clusters
contingency_table = pd.crosstab(clustering_1, clustering_2, rownames=['model 1'], colnames=['model 2'])
print('----------------------------------')
print("--------------Part c--------------")
print('----------------------------------')
print("Contingency table of the clusters:\n", contingency_table)

def cluster_plot(clustering1, clustering2, k, PATH):
    """
    Plots the data, color coded by cluster, for two different models.
    
    Parameters
    ----------
    clustering1 : array-like
        The cluster labels of the first model.
    clustering2 : array-like
        The cluster labels of the second model.
    k : int
        The number of clusters.
    """
    # Plot the scatter plot of the PCA visualised data
    # with density contours overlaid
    fig, axes = plt.subplots(1,2,figsize=(10, 5))

    # Scatter plot for the 1st model
    sns.scatterplot(x=A_pca_df['PC1'].loc[A_2.index.values], y=A_pca_df['PC2'].loc[A_2.index.values], ax = axes[0], hue=clustering1, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Scatter plot for the 2nd model
    sns.scatterplot(x=A_pca_df['PC1'].loc[A_1.index.values], y=A_pca_df['PC2'].loc[A_1.index.values], ax = axes[1], hue=clustering2, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel(None)
    plt.suptitle('Scatter Plot of the 2 clusterings, for k = '+str(k)+' clusters')
    plt.tight_layout()
    plt.savefig('Plots/'+PATH+'.png')
    plt.show()

cluster_plot(clustering_1, clustering_2, 8, 'A_Q1c_8clusters')



# ------------------------------------------------------------------------------
# (d) Silhouette score + clustering for optimal cluster number
# ------------------------------------------------------------------------------
print('----------------------------------')
print("--------------Part d--------------")
print('----------------------------------')



# Get the cluster sizes
cluster_sizes = []
for i in range(8):
    print('Cluster '+str(i+1)+' size, for model 1: '+str(np.count_nonzero(train1_cluster == i)))
    print('Cluster '+str(i+1)+' size, for model 2: '+str(np.count_nonzero(train2_cluster == i)))
    cluster_sizes.append(np.count_nonzero(train1_cluster == i))
    cluster_sizes.append(np.count_nonzero(train2_cluster == i))


# Assess the cluster stability



# Get the silhouette scores for different cluster numbers
silhouette_scores = []
for k in range(2, 8):
    Model = KMeans(n_clusters=k).fit(A_fea)
    cluster_labels = Model.labels_
    silhouette_scores.append(silhouette_score(A_fea, cluster_labels))

# Plot the silhouette scores
plt.figure(figsize=(6, 5))
plt.plot(range(2, 8), silhouette_scores, marker='o')
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('Silhouette score for different cluster numbers')
plt.tight_layout()
plt.savefig('Plots/A_Q1d_silhouette.png')
plt.show()

# Get the optimal cluster number
optimal_k = np.argmax(silhouette_scores) + 2 # to correct 0-based indexing

# Apply k-means clustering with the optimal cluster number
Model = KMeans(n_clusters=optimal_k).fit(A_fea)
cluster_labels = Model.labels_




# Plot the scatter plot of the PCA visualised data
# with cluster color coding

plt.figure(figsize=(6, 5))

# Scatter plot
sns.scatterplot(x=A_pca_df['PC1'], y=A_pca_df['PC2'], hue=cluster_labels, palette='plasma', alpha=0.5)

plt.grid()
plt.title('Scatter Plot of A_pca, with k = '+str(optimal_k)+' clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig('Plots/A_Q1d.png')
plt.show()




# ------------------------------------------------------------------------------
# (e) Identify the clusters within the PCA figure
# ------------------------------------------------------------------------------

