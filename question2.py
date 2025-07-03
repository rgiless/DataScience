#%% Question A.2 a)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# Define TPR and FPR values
TPR = [0, 1/6, 1/3, 1/2, 1/2, 2/3, 5/6, 5/6, 1, 1, 1, 1, 1]
FPR = [0, 0, 0, 0, 1/6, 1/6, 1/6, 1/3, 1/3, 1/2, 2/3, 5/6, 1]

# Plot ROC curve from TPR and FPR values
plt.figure(figsize=(8, 6))
plt.plot(FPR, TPR, color='blue', linestyle='-', marker='o')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Question A.2 b)

# Load in predictions
predict = pd.read_csv('models_predictions.csv')

# Seperate columns of the loaded in predictions into the actual data and the 3
# models DTC, Optimal DTC and Optimal RFC
y_true = predict.iloc[:, 0]
dtc_noreg = predict.iloc[:, 1]
dtc_best = predict.iloc[:, 2]        
rfc_best = predict.iloc[:, 3]

# Define function to plot ROC curves of all 3 models and print their AOC scores 
# in the legend
def plot_roc_curve(model, label=None):
    fpr, tpr, thresholds = roc_curve(model, y_true)
    auc = roc_auc_score(model, y_true)
    plt.plot(fpr, tpr, linewidth=2, label=f'{label} - (AUC Score: {auc:.5f})')
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

# Use ROC function to plot ROC curves of the 3 models
plot_roc_curve(dtc_noreg, label='DTC No Reg')
plot_roc_curve(dtc_best, label='DTC Best')
plot_roc_curve(rfc_best, label='RFC Best')