import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Load in data set and assign column names
bh = pd.read_csv('boston_housing.txt', sep='\s+', names = 
                 ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                  'AGE', 'DIS', 'PTRATIO', 'LSTAT', 'MEDV'])

# Display first 5 lines of dataframe
print(bh.head())
rows, columns = bh.shape
# Display number of rows and columns
print(f'Number of rows: {rows}\nNumber of columns: {columns}')
# Print attribute types
print(f'\nAttribute types: \n{bh.dtypes}\n')
# Change attribute type of CHAS to category
bh = bh.astype({'CHAS' : 'category'}, copy=True)
# Print attribute types again to check modification
print(f'Altered attribute types: \n{bh.dtypes}')
# Print attributes with their associated number of NaN values to see which needed to be
# filled in
print(f'\nInformation on attributes and their NaN values: \n')
bh.info()
# Plot distribution of NOX attribute to decide which fill to use
bh['NOX'].hist(bins=30)
plt.xlabel('NOX Concentration')
plt.ylabel('Frequency')
plt.title('Distribution of NOX Values')
plt.show()

# Create copy of original dataframe
bh1 = bh.copy()
# Permanently fill null values in NOX attribute using column median
bh1['NOX'].fillna(bh1['NOX'].median(), inplace=True)
# Print attributes with associated number of null values to check
# previous fill worked correctly.
print(f'\nInformation on attributes and their null values after fill: \n')
bh1.info()

# Create categorical version of INDUS attribute and bin
bh1["INDUS_cat"] = pd.cut(bh1["INDUS"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])

# Use stratified split to create train and test sets
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.24, random_state=42)
for train_index, test_index in split.split(bh1, bh1["INDUS_cat"]):
     strat_train_set = bh1.loc[train_index]
     strat_test_set = bh1.loc[test_index]

# Plot histograms of test data and original data to ensure agreement between them
plt.figure(figsize=(10, 6))
sns.histplot(bh1['INDUS_cat'], color='blue', label='Original Data', kde=False, bins=10)
sns.histplot(strat_test_set['INDUS_cat'], color='red', label='Stratified Test Data', kde=False, bins=10)
plt.title('Histograms of INDUS_cat for Original and Stratified Test Data')
plt.xlabel('INDUS_cat')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Check test data and original data proportions agree
print('Comparison of test set and original set: \n')
print(f'Original Set: \n{bh1["INDUS_cat"].value_counts() / len(bh1)} \n')  
print(f'Test Set: \n{strat_test_set["INDUS_cat"].value_counts() / len(strat_test_set)} \n')

  

# Drop created INDUS cateogry column
for set_ in (strat_train_set, strat_test_set):
     set_.drop('INDUS_cat', axis=1, inplace=True)

# Print size of test and train set to ensure correct 24% ratio    
print(f'Test data shape: \n{strat_test_set.shape}')
print(f'Train data shape: \n{strat_train_set.shape}')
print(f'Shape of full data: \n{bh1.shape}\n')
print(f'Test to Original set ratio: {strat_test_set.shape[0]/bh1.shape[0]}\n')
print(f'Train to Original set ratio: {strat_train_set.shape[0]/bh1.shape[0]}\n')

# Create correlation coefficient matrix and sort in ascending correlation to RM attribute
corr_matrix = strat_train_set.corr()
print(f'Corr matrix: \n{corr_matrix["RM"].sort_values(ascending=True)}')

# PLot scatter matrix of 5 given attributes to assess correlation
attributes = ['MEDV', 'RM', 'LSTAT', 'INDUS', 'CRIM']
scatter_matrix(strat_train_set[attributes], figsize=(20, 12))
