

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

# Get the data without the labels
Baseline_no_labels = Baseline.drop(columns=['type'])

# Scale the data, using a robust scaler
scaler = RobustScaler()
scaler.fit(Baseline_no_labels)
Baseline_no_labels = pd.DataFrame(scaler.transform(Baseline_no_labels), columns=Baseline_no_labels.columns, index=Baseline_no_labels.index)


# K-means

# Get the inertia for different values of k
inertia = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Baseline_no_labels)
    inertia.append(kmeans.inertia_)

# Plot the inertia
plt.figure(figsize=(6, 5))
plt.plot(range(1, 21), inertia, marker='o')
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Inertia for different cluster numbers')
plt.tight_layout()
plt.savefig('Plots/B_Q5a_inertia.png')
plt.show()
