#%% Question A.3 a)
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Load in MINST dataset and seperate data and targets and training and testing sets
minst = fetch_openml('mnist_784',version = 1)
X, y = minst["data"], minst["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Use PCA to reduce dimensionality of MINST set, both training and testing sets,
# using specified variance ratio
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

#%% Question A.3 b)

# Define measures function to display scores of a given model compared to actual data
def measures(y_actual, y_pred):
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred, average='weighted')
    recall = recall_score(y_actual, y_pred, average='weighted')
    print(f'\nAccuracy Score: \n{accuracy}')
    print(f'\nPrecision Score: \n{precision}')
    print(f'\nRecall Score: \n{recall}')

# Train RFC on full dataset
print('Random Forest Classifier')
print('Full Dataset:\n')
rfc_clf = RandomForestClassifier()
# Calculate training time of RFC model
t0 = time.time()
rfc_clf.fit(X_train, y_train)
t1 = time.time()
# Print training time and scores of RFC model on full data
print(f'Training took {t1 - t0:.2f} seconds')
y_pred = rfc_clf.predict(X_test)
measures(y_test, y_pred)

# Train RFC model on PCA reduced dataset and print time taken for training and
# resulting scores
print('\nReduced Dataset:\n')
rfc_clf_reduced = RandomForestClassifier()
t0_red = time.time()
rfc_clf_reduced.fit(X_train_reduced, y_train)
t1_red = time.time()
print(f'Training took {t1_red - t0_red:.2f} seconds')
y_pred_red = rfc_clf_reduced.predict(X_test_reduced)
measures(y_test, y_pred_red)

#%% Question A.3 c)

# Train LR model on full dataset and print time taken for training and
# resulting scores
lr = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')
t0_lr = time.time()
lr.fit(X_train, y_train)
t1_lr = time.time()
print('\nLogistic Regression Model')
print('\nFull Dataset:\n')
print(f'Training took {t1_lr - t0_lr:.2f} seconds')
y_pred_lr = lr.predict(X_test)
measures(y_test, y_pred_lr)

# Train LR model on PCA reduced dataset and print time taken for training and
# resulting scores
print('\nReduced Dataset:\n')
lr_reduced = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')
t0_lr_red = time.time()
lr_reduced.fit(X_train_reduced, y_train)
t1_lr_red = time.time()
print(f'Training took {t1_lr_red - t0_lr_red:.2f} seconds')
y_pred_lr_red = lr_reduced.predict(X_test_reduced)
measures(y_test, y_pred_lr_red)

#%% Question A.3 e)

# Print and plot confusion matrix of Logistic Regression model on PCA reduced 
# dataset
confuse_lr_red = confusion_matrix(y_test, y_pred_lr_red)
print(f'Confusion Matrix: \n{confuse_lr_red}')
plt.matshow(confuse_lr_red)

