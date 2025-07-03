import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

# Import training and test data sets
bh_train = pd.read_csv('bh_train_acw1.csv')
bh_test = pd.read_csv('bh_test_acw1.csv')

# Set CHAS attibute as categorical in both data sets
bh_train = bh_train.astype({'CHAS': 'category'}, copy=True)
bh_test = bh_test.astype({'CHAS': 'category'}, copy=True)

# Split training set into features and labels space
X = bh_train.drop('MEDV', axis=1)
y = bh_train['MEDV'].copy()

# Split testing set into features and label space
X_test = bh_test.drop('MEDV', axis=1)
y_test = bh_test['MEDV'].copy()

# Split training set into numerical and categorical sets
X_num_attributes = list(X.drop('CHAS', axis=1))
X_cat_attributes = ['CHAS']

# Create pipeline to preprocess training set
full_pipeline = ColumnTransformer([
    ('num', StandardScaler(), X_num_attributes),     
    ('cat', OneHotEncoder(), X_cat_attributes)])

# Preprocess training set
X_prep = full_pipeline.fit_transform(X)

# Split test set into numerical and categorical sets
X_test_num_attributes = list(X_test.drop('CHAS', axis=1))
X_test_cat_attributes = ['CHAS']

# Create pipeline to preprocess test dset
full_pipeline = ColumnTransformer([
    ('num', StandardScaler(), X_test_num_attributes),     
    ('cat', OneHotEncoder(), X_test_cat_attributes)])

# Preprocess test set
X_test_prep = full_pipeline.fit_transform(X_test)

print('Train Set:\n')

# Train linear regression model on preprocessed training set
lin_reg = LinearRegression()
lin_reg.fit(X_prep, y)
# Use regression model to make predictions based on its training
lin_predictions = lin_reg.predict(X_prep)

# Calculate and print the RMSE of predictions
lin_rmse = np.sqrt(mean_squared_error(y, lin_predictions))
print(f'Linear Regression RMSE: {lin_rmse}')

# Train Random Forest Regressor model on preprocessed training set
forest_reg = RandomForestRegressor(max_depth=10, min_samples_split=4, min_samples_leaf=2, random_state=42)
forest_reg.fit(X_prep, y)
# Use forest model to make predictions based on its training
forest_predictions = forest_reg.predict(X_prep)
# Calculate and print the RMSE of predictions
forest_rmse = np.sqrt(mean_squared_error(y, forest_predictions))
print(f'Forest Regression RMSE: {forest_rmse}\n\n')

# Train linear regression model with cross validation and calculate cross validation scores
lin_scores = cross_val_score(lin_reg, X_prep, y, scoring = 'neg_mean_squared_error', cv = 3)
print(f'Raw Linear Regression CV Scores: {lin_scores}')
# Take square root of negative cross validation scores then find mean to enable comparison with previously
# calculated RMSE 
lin_rmse_scores = np.sqrt(-lin_scores)
print(f'Transformed Linear Regression RMSE Scores: {lin_rmse_scores}\nMean Linear Regression CV Score: {np.mean(lin_rmse_scores)}\n')

# Train random forest regression model with cross validation and calculate cross validation scores
forest_scores = cross_val_score(forest_reg, X_prep, y, scoring = 'neg_mean_squared_error', cv = 3)
print(f'Raw Random Forest CV Scores: {forest_scores}')
# Take square root of negative cross validation scores then find mean to enable comparison with previously
# calculated RMSE 
forest_rmse_scores = np.sqrt(-forest_scores)
print(f'Transformed Forest RMSE Scores: {forest_rmse_scores} \nMean Random Forest CV Score: {np.mean(forest_rmse_scores)}')

print('\n\nTest Set:\n')

# Repeat previous code under 'Train Set' with corresponding test set to assess validity of models
# on unseen data

lin_test_predictions = lin_reg.predict(X_test_prep)
lin_test_rmse = np.sqrt(mean_squared_error(y_test, lin_test_predictions))
print(f'Linear Regression RMSE: {lin_test_rmse}')

forest_test_predictions = forest_reg.predict(X_test_prep)
forest_test_rmse = np.sqrt(mean_squared_error(y_test, forest_test_predictions))
print(f'Forest Regression RMSE: {forest_test_rmse}\n\n')

lin_test_scores = cross_val_score(lin_reg, X_test_prep, y_test, scoring = 'neg_mean_squared_error', cv = 3)
print(f'Raw Linear Regression CV Scores: {lin_test_scores}')
lin_test_rmse_scores = np.sqrt(-lin_test_scores)
print(f'Transformed Linear Regression RMSE Scores: {lin_test_rmse_scores}\nMean Linear Regression CV Score: {np.mean(lin_test_rmse_scores)}\n')

forest_test_scores = cross_val_score(forest_reg, X_test_prep, y_test, scoring = 'neg_mean_squared_error', cv = 3)
print(f'Raw Random Forest CV Scores: {forest_test_scores}')
forest_test_rmse_scores = np.sqrt(-forest_test_scores)
print(f'Transformed Forest RMSE Scores: {forest_test_rmse_scores} \nMean Random Forest CV Score: {np.mean(forest_test_rmse_scores)}')



