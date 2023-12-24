import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def A_Q1a(data):

    fig, axes = plt.subplots(3,2, figsize=(10, 15))


    sns.kdeplot(data['Fea11'], ax=axes[0,0], label='Fea11', legend=True)
    sns.kdeplot(data['Fea14'], ax=axes[0,0], label='Fea14', legend=True)
    sns.kdeplot(data['Fea19'], ax=axes[0,0], label='Fea19', legend=True)
    axes[0,0].set_title('Density plots of Fea11-14-19')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_xlabel('Feature value')
    axes[0,0].set_xlim(-2,8)
    axes[0,0].legend()

    sns.kdeplot(data['Fea13'], ax=axes[0,1], label='Fea13', legend=True)
    sns.kdeplot(data['Fea2'], ax=axes[0,1], label='Fea2', legend=True)
    sns.kdeplot(data['Fea17'], ax=axes[0,1], label='Fea17', legend=True)
    axes[0,1].set_title('Density plots of Fea2-13-17')
    axes[0,1].set_xlabel('Feature value')
    axes[0,1].set_ylabel(None)
    axes[0,1].set_xlim(-2,8)
    axes[0,1].legend()

    sns.kdeplot(data['Fea18'], ax=axes[1,0], label='Fea18', legend=True)
    sns.kdeplot(data['Fea20'], ax=axes[1,0], label='Fea20', legend=True)
    sns.kdeplot(data['Fea5'], ax=axes[1,0], label='Fea5', legend=True)
    axes[1,0].set_title('Density plots of Fea5-18-20')
    axes[1,0].set_xlabel('Feature value')
    axes[1,0].set_ylabel(None)
    axes[1,0].set_xlim(-2,8)
    axes[1,0].legend()


    sns.kdeplot(data['Fea1'], ax=axes[1,1], label='Fea1', legend=True)
    sns.kdeplot(data['Fea3'], ax=axes[1,1], label='Fea3', legend=True)
    sns.kdeplot(data['Fea4'], ax=axes[1,1], label='Fea4', legend=True)
    axes[1,1].set_title('Density plots of Fea1-3-4')
    axes[1,1].set_xlabel('Feature value')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_xlim(-2,8)
    axes[1,1].legend()


    sns.kdeplot(data['Fea6'], ax=axes[2,0], label='Fea6', legend=True)
    sns.kdeplot(data['Fea7'], ax=axes[2,0], label='Fea7', legend=True)
    sns.kdeplot(data['Fea8'], ax=axes[2,0], label='Fea8', legend=True)
    sns.kdeplot(data['Fea9'], ax=axes[2,0], label='Fea9', legend=True)
    axes[2,0].set_title('Density plots of Fea6-9')
    axes[2,0].set_xlabel('Feature value')
    axes[2,0].set_ylabel(None)
    axes[2,0].set_xlim(-2,8)
    axes[2,0].legend()


    sns.kdeplot(data['Fea10'], ax=axes[2,1], label='Fea10', legend=True)
    sns.kdeplot(data['Fea12'], ax=axes[2,1], label='Fea12', legend=True)
    sns.kdeplot(data['Fea15'], ax=axes[2,1], label='Fea15', legend=True)
    sns.kdeplot(data['Fea16'], ax=axes[2,1], label='Fea16', legend=True)
    axes[2,1].set_title('Density plots of Fea10-12-15-16')
    axes[2,1].set_xlabel('Feature value')
    axes[2,1].set_ylabel(None)
    axes[2,1].set_xlim(-2,8)
    axes[2,1].legend()

    plt.tight_layout()
    plt.savefig('Plots/A_Q1a.png')
    plt.close()


def A_Q1b(data):

    plt.figure(figsize=(6, 5))

    # Scatter plot of the PCA visualised data
    sns.scatterplot(x=data['PC1'], y=data['PC2'], color='blue', alpha=0.5)

    # Density map overlay
    sns.kdeplot(x=data['PC1'], y=data['PC2'], cmap="inferno", levels=5, thresh=0.05)

    plt.grid()
    plt.title('Scatter Plot with contours of A_pca')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig('Plots/A_Q1b.png')
    plt.close()


def cluster_plot(data, split1, split2, clustering1, clustering2, k, PATH):
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
    sns.scatterplot(x=data['PC1'].loc[split2.index.values], y=data['PC2'].loc[split2.index.values],
                    ax = axes[0], hue=clustering1, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Scatter plot for the 2nd model
    sns.scatterplot(x=data['PC1'].loc[split1.index.values], y=data['PC2'].loc[split1.index.values], 
                    ax = axes[1], hue=clustering2, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel(None)
    plt.suptitle('Scatter Plot of the 2 clusterings, for k = '+str(k)+' clusters')
    plt.tight_layout()
    plt.savefig('Plots/'+PATH+'.png')
    plt.close()


def A_Q1d_silhouette(silhouette_scores):
    

    plt.figure(figsize=(6, 5))

    # Plot the silhouette scores
    plt.plot(range(2, 8), silhouette_scores, marker='o')
    plt.grid()
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for different cluster numbers')
    plt.tight_layout()
    plt.savefig('Plots/A_Q1d_silhouette.png')
    plt.close()


def A_Q3c(imputed, original, num):
    
    fig, axes = plt.subplots(4,3, figsize=(10, 15))


    sns.kdeplot(original['Fea58'], ax=axes[0,0], label='Original', legend=True)
    sns.kdeplot(imputed['Fea58'], ax=axes[0,0], label='Imputed', legend=True)
    axes[0,0].set_title('Density plots of Fea58')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_xlabel('Feature 58')
    axes[0,0].set_xlim(-2,8)
    axes[0,0].legend()

    sns.kdeplot(original['Fea142'], ax=axes[0,1], label='Original', legend=True)
    sns.kdeplot(imputed['Fea142'], ax=axes[0,1], label='Imputed', legend=True)
    axes[0,1].set_title('Density plots of Fea58')
    axes[0,1].set_xlabel('Feature 142')
    axes[0,1].set_xlim(-2,8)
    axes[0,1].legend()

    sns.kdeplot(original['Fea150'], ax=axes[0,2], label='Original', legend=True)
    sns.kdeplot(imputed['Fea150'], ax=axes[0,2], label='Imputed', legend=True)
    axes[0,2].set_title('Density plots of Fea150')
    axes[0,2].set_xlabel('Feature 150')
    axes[0,2].set_xlim(-2,8)
    axes[0,2].legend()

    sns.kdeplot(original['Fea233'], ax=axes[1,0], label='Original', legend=True)
    sns.kdeplot(imputed['Fea233'], ax=axes[1,0], label='Imputed', legend=True)
    axes[1,0].set_title('Density plots of Fea233')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_xlabel('Feature 233')
    axes[1,0].set_xlim(-2,8)
    axes[1,0].legend()

    sns.kdeplot(original['Fea269'], ax=axes[1,1], label='Original', legend=True)
    sns.kdeplot(imputed['Fea269'], ax=axes[1,1], label='Imputed', legend=True)
    axes[1,1].set_title('Density plots of Fea269')
    axes[1,1].set_xlabel('Feature 269')
    axes[1,1].set_xlim(-2,8)
    axes[1,1].legend()

    sns.kdeplot(original['Fea299'], ax=axes[1,2], label='Original', legend=True)
    sns.kdeplot(imputed['Fea299'], ax=axes[1,2], label='Imputed', legend=True)
    axes[1,2].set_title('Density plots of Fea299')
    axes[1,2].set_xlabel('Feature 299')
    axes[1,2].set_xlim(-2,8)
    axes[1,2].legend()

    sns.kdeplot(original['Fea339'], ax=axes[2,0], label='Original', legend=True)
    sns.kdeplot(imputed['Fea339'], ax=axes[2,0], label='Imputed', legend=True)
    axes[2,0].set_title('Density plots of Fea339')
    axes[2,0].set_ylabel('Density')
    axes[2,0].set_xlabel('Feature 339')
    axes[2,0].set_xlim(-2,8)
    axes[2,0].legend()

    sns.kdeplot(original['Fea355'], ax=axes[2,1], label='Original', legend=True)
    sns.kdeplot(imputed['Fea355'], ax=axes[2,1], label='Imputed', legend=True)
    axes[2,1].set_title('Density plots of Fea355')
    axes[2,1].set_xlabel('Feature 355')
    axes[2,1].set_xlim(-2,8)
    axes[2,1].legend()

    sns.kdeplot(original['Fea458'], ax=axes[2,2], label='Original', legend=True)
    sns.kdeplot(imputed['Fea458'], ax=axes[2,2], label='Imputed', legend=True)
    axes[2,2].set_title('Density plots of Fea458')
    axes[2,2].set_xlabel('Feature 458')
    axes[2,2].set_xlim(-2,8)
    axes[2,2].legend()

    sns.kdeplot(original['Fea466'], ax=axes[3,0], label='Original', legend=True)
    sns.kdeplot(imputed['Fea466'], ax=axes[3,0], label='Imputed', legend=True)
    axes[3,0].set_title('Density plots of Fea466')
    axes[3,0].set_ylabel('Density')
    axes[3,0].set_xlabel('Feature 466')
    axes[3,0].set_xlim(-2,8)
    axes[3,0].legend()

    sns.kdeplot(original['Fea233'], ax=axes[3,1], label='Original', legend=True)
    sns.kdeplot(imputed['Fea233'], ax=axes[3,1], label='Imputed', legend=True)
    axes[3,1].set_title('Density plots of Fea233')
    axes[3,1].set_xlabel('Feature 233')
    axes[3,1].set_xlim(-2,8)
    axes[3,1].legend()

    plt.tight_layout()
    plt.savefig('Plots/A_Q3c_'+ str(num) +'.png')
    plt.close()

def B_Q4d(n_estimators, error_rate, num):
    # Plot the error rate against the number of trees
    plt.figure(figsize=(6, 5))
    plt.plot(n_estimators, error_rate)
    plt.xlabel('Number of trees')
    plt.ylabel('OOB error rate')
    plt.title('OOB error rate vs number of trees')
    plt.grid()
    plt.tight_layout()
    plt.savefig('Plots/B_Q4d_' + str(num) + '.png')
    plt.close()


def B_Q4e(feature_importances, num):
    plt.figure(figsize=(5, 12))
    feature_importances.nlargest(50).plot(kind='barh')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('Plots/B_Q4e_' + str(num) + '.png')


def B_Q5a_silhouette(silhouette_scores):
    plt.figure(figsize=(6, 5))
    plt.plot(range(2, 8), silhouette_scores, marker='o')
    plt.grid()
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for different cluster numbers')
    plt.tight_layout()
    plt.savefig('Plots/B_Q5a_silhouette.png')


def B_Q5c_1( data_pca, clustering1, clustering2, PATH):
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
    sns.scatterplot(x=np.array(data_pca['PC1']), 
                    y=np.array(data_pca['PC2']),
                    ax = axes[0], hue=clustering1, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Scatter plot for the 2nd model
    sns.scatterplot(x=np.array(data_pca['PC1']), 
                    y=np.array(data_pca['PC2']), 
                    ax = axes[1], hue=clustering2, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel(None)
    plt.suptitle('Scatter Plot of the 2 clustering techniques,\n K-Means on the left and GMM on the right')
    plt.tight_layout()
    plt.savefig('Plots/'+PATH+'.png')
    plt.close()

def B_Q5c_2(data, feature1, feature2, PATH):
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

    # Scatter plot for the 1st model with colour depending on the value of feature1
    sns.scatterplot(x=np.array(data['PC1']), 
                    y=np.array(data['PC2']),
                    ax = axes[0], hue=feature1, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # Scatter plot for the 2nd model
    sns.scatterplot(x=np.array(data['PC1']), 
                    y=np.array(data['PC2']), 
                    ax = axes[1], hue=feature2, palette='coolwarm')

    plt.xlabel('PC1')
    plt.ylabel(None)
    plt.suptitle('Scatter Plot of the data,\n according to most important feature (Feature)')
    plt.tight_layout()
    plt.savefig('Plots/'+PATH+'.png')
    plt.close()