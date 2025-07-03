#%% Question A.1 a) i)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Read in Titanic dataset and print info and head
tnctrain = pd.read_csv('titanic_strat_train.csv')
print(f'Head of tnctrain: \n{tnctrain.head()}')
print('\nBefore attribute type adjustment: \n')
tnctrain.info()

# Manually define correct attribute types
titanic_dtypes = {
    'pclass': 'category',
    'survived': 'category',
    'name': 'string',
    'sex': 'category',
    'age': 'float64',
    'sibsp': 'int64',
    'parch': 'int64',
    'ticket': 'string',
    'fare': 'float64',
    'cabin': 'string',
    'embarked': 'category',
    'boat': 'string',
    'body': 'float64',
    'home.dest': 'string'
}

# Redefine attributes with corrected attribute types
tnctrain1 = tnctrain.astype(titanic_dtypes)
print('\nAfter attribute type adjustment: \n')
tnctrain1.info()

#%% Question A.1 a) ii)

# Remove unsuitable columns
tnctrain1 = tnctrain1.drop(columns = ['cabin', 'boat', 'body', 'home.dest'])
print('After dropping unsuitable columns: \n')
tnctrain1.info()

# Plot distribution of age in dataset
tnctrain1['age'].hist(bins=100)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age Values')
plt.show()

# Plot distribution of fare in dataset
tnctrain1['fare'].hist(bins=100)
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Distribution of Fare Values')
plt.show()

# Fill NaN/missing values in age and fare with backward fill
tnctrain1['age'].bfill(inplace=True)
tnctrain1['fare'].bfill(inplace=True)
print('\nAfter fills: \n')
tnctrain1.info()
 
#%% Question A.1 a) iii)

# Split data into labels and predictors
x = tnctrain1.drop(columns=['survived', 'name', 'ticket'])
y = tnctrain1['survived']

# Identify categorical data and use OneHotEncoder to encode these attributes
# and remove initial columns
x_cat = ['pclass', 'sex', 'embarked']
ohe = ColumnTransformer([('cat', OneHotEncoder(), x_cat)], remainder='passthrough')
x_ohe_array = ohe.fit_transform(x)
ohe_column_names = ohe.named_transformers_['cat'].get_feature_names_out(x_cat)
all_columns = list(ohe_column_names) + list(x.drop(columns=x_cat).columns)
x_prep = pd.DataFrame(x_ohe_array, columns=all_columns, index=x.index)
print('Attrubutes of modified dataframe: \n')
x_prep.info()

#%% Question A.1 a) iv)

# Store copy of unaltered age and fare attributes to retain outliers
age_before = x_prep['age'].copy()
fare_before = x_prep['fare'].copy()

# Define IQR method to remove outliers
def IQR(data, attribute):
    q1 = data[attribute].quantile(0.25)
    q3 = data[attribute].quantile(0.75)
    iqr = q3 - q1
    bound1 = q1 - 1.5 * iqr
    bound2 = q3 + 1.5 * iqr
    outliers = (data[attribute] < bound1) | (data[attribute] > bound2)
    median = data[attribute].median()
    data.loc[outliers, attribute] = median

# Remove outliers in age and fare columns
IQR(x_prep, 'age')
IQR(x_prep, 'fare')

# Calculate how many outliers were altered in age and fare columns
num_age_replaced = (x_prep["age"] != age_before).sum()
num_fare_replaced = (x_prep["fare"] != fare_before).sum()

print(f'Number of values replaced in age attribute = {num_age_replaced}')
print(f'Number of values replaced in fare attribute = {num_fare_replaced}')

print(x_prep.info())

# Plot boxplots of age and fare before and after outlier alteration
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(y=age_before, ax=axes[0, 0]).set_title('Age - Before IQR')
sns.boxplot(y=x_prep['age'], ax=axes[0, 1]).set_title('Age - After IQR')
sns.boxplot(y=fare_before, ax=axes[1, 0]).set_title('Fare - Before IQR')
sns.boxplot(y=x_prep['fare'], ax=axes[1, 1]).set_title('Fare - After IQR')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%% Question A.1 b)

# Calculate survival rate of each gender and embarkation port and convert to 
# percentage
gender_survival = tnctrain.groupby('sex')['survived'].mean()*100
embarked_survival = tnctrain.groupby('embarked')['survived'].mean()*100  

# Plot survival rate of genders
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_survival.index, y=gender_survival.values)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate / %')
plt.show()

# Plot survival rate of embarkation ports
plt.figure(figsize=(8, 6))
sns.barplot(x=embarked_survival.index, y=embarked_survival.values)
plt.title('Survival Rate by Embarkation')
plt.xlabel('Embarkation Port')
plt.ylabel('Survival Rate / %')
plt.show()

# Print survival rates
print('Survival Rate in Percentage by Gender:')
print(gender_survival)
print('\nSurvival Rate in Percentage by Embarkation Port:')
print(embarked_survival)

#%% Question A.1 c) i)

# Define function to output prediction scores of a given model compared to 
# actual data
def measures(y_actual, y_pred):
    confuse = confusion_matrix(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred) 
    print(f'\nConfusion Matrix: \n{confuse}')
    print(f'\nAccuracy Score: \n{accuracy}')
    print(f'\nPrecision Score: \n{precision}')
    print(f'\nRecall Score: \n{recall}')
    print(f'\nF1 Score: \n{f1}')

#%% Question A.1 c) ii)
    
# Train DTC model and print scores
tree_clf = DecisionTreeClassifier()
tree_clf.fit(x_prep, y)
y_pred = tree_clf.predict(x_prep)
measures(y, y_pred)

#%% Question A.1 c) iii)

# Repeat training of DTC using 5-fold CV and print scores
tree_scores = cross_val_score(tree_clf, x_prep, y, scoring='accuracy', cv=5)

def display_scores(scores):
    print('\nScores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())

display_scores(tree_scores)
y_pred_cv = cross_val_predict(tree_clf, x_prep, y, cv=5)
measures(y, y_pred_cv)

#%% Question A.1 c) iv)

# Train DTC with depth of 10 and print scores
tree_clf_10 = DecisionTreeClassifier(max_depth=10)
tree_clf_10.fit(x_prep, y)
y_pred_10 = tree_clf_10.predict(x_prep)
measures(y, y_pred_10)

# Use gridsearch to optimise max_depth parameter
params = {'max_depth': np.arange(1, 41)}
optimise = GridSearchCV(tree_clf, params, cv=5, scoring='accuracy')
optimise.fit(x_prep, y)

# Print optimal max_depth and scores resulting from this optimal model
print('\nOptimal Depth:', optimise.best_params_)
print('Optimal Estimator:', optimise.best_estimator_)

optimal_model = optimise.best_estimator_
y_pred_optim = optimal_model.predict(x_prep)
measures(y, y_pred_optim)

#%% Question A.1 c) v)

# Train RFC model with optimal parameters and print resulting scores
forest_clf = RandomForestClassifier()
forest_optimise = GridSearchCV(forest_clf, params, cv=5, scoring='accuracy')
forest_optimise.fit(x_prep, y)

print('Optimal Depth:', forest_optimise.best_params_)
print('Optimal Estimator:', forest_optimise.best_estimator_)

optimal_forest_model = forest_optimise.best_estimator_
y_pred_optim_forest = optimal_forest_model.predict(x_prep)
measures(y, y_pred_optim_forest)

#%% Question A.1 d) i)

# Read in and prepare test set of Titanic dataset using same process used for 
# training data
tnctest = pd.read_csv('titanic_strat_test.csv')
tnctest1 = tnctest.astype(titanic_dtypes)
tnctest1 = tnctest1.drop(columns = ['cabin', 'boat', 'body', 'home.dest'])
tnctest1['age'].bfill(inplace=True)
tnctest1['fare'].bfill(inplace=True)
tnctest1['embarked'].fillna(tnctest1['embarked'].mode()[0], inplace=True)

x_test = tnctest1.drop(columns=['survived', 'name', 'ticket'])
y_test = tnctest1['survived']
x_cat_test = ['pclass', 'sex', 'embarked']
ohe_test = ColumnTransformer([('cat', OneHotEncoder(), x_cat_test)], remainder='passthrough')
x_ohe_array_test = ohe_test.fit_transform(x_test)
ohe_column_names_test = ohe_test.named_transformers_['cat'].get_feature_names_out(x_cat_test)
all_columns_test = list(ohe_column_names_test) + list(x.drop(columns=x_cat_test).columns)
x_prep_test = pd.DataFrame(x_ohe_array_test, columns=all_columns_test, index=x_test.index)
IQR(x_prep_test, 'age')
IQR(x_prep_test, 'fare')
print('Prepared Test Set: \n')
x_prep_test.info()

#%% Question A.1 d) ii)

# Compare predictions of DTC with no params, DTC with optimal params and RFC
# with optimal params to the test set and print resulting scores
y_pred_test = tree_clf.predict(x_prep_test)
y_pred_optim_test = optimal_model.predict(x_prep_test)
y_pred_optim_forest_test = optimal_forest_model.predict(x_prep_test)

print('\nDTC No Param (Test Set): \n')
measures(y_test, y_pred_test)

print('\nDTC Optimal Param (Test Set): \n')
measures(y_test, y_pred_optim_test)

print('\nRFC Optimal Param (Test Set): \n')
measures(y_test, y_pred_optim_forest_test)


