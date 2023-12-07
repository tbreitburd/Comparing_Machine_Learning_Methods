import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from warnings import simplefilter


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