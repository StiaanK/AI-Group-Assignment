

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the Red Wine Classification Dataset
#red_wine_data = pd.read_csv('C:\\Users\\User\\OneDrive\\Desktop\\Semester_2\\ITRI_626_AI\\Assignments\\UCI_Datasets\\Classification_Datasets\\wine+quality\\winequality-red.csv', delimiter=';')
red_wine_data = pd.read_csv('C:\\Users\\S_CSIS-PostGrad\\OneDrive\\Desktop\\Semester_2\\ITRI_626_AI\\Assignments\\UCI_Datasets\\Classification_Datasets\\wine+quality\\winequality-red.csv', delimiter=';')

# Split features and target variable
X_red_wine = red_wine_data.drop('quality', axis=1)
y_red_wine = red_wine_data['quality']

# Convert quality to binary class (good vs. not good)
y_red_wine_binary = (y_red_wine >= 6).astype(int)  # 6 and above are considered good quality

# Train/Test Split
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red_wine, y_red_wine_binary, test_size=0.2, random_state=42)

# Model Initialization
random_forest_clf_red = RandomForestClassifier(random_state=42)
gradient_boosting_clf_red = GradientBoostingClassifier(random_state=42)
svm_clf_red = SVC(probability=True, random_state=42)

# Model Training
random_forest_clf_red.fit(X_train_red, y_train_red)
gradient_boosting_clf_red.fit(X_train_red, y_train_red)
svm_clf_red.fit(X_train_red, y_train_red)

# Model Prediction
y_pred_rf_red = random_forest_clf_red.predict(X_test_red)
y_pred_gb_red = gradient_boosting_clf_red.predict(X_test_red)
y_pred_svm_red = svm_clf_red.predict(X_test_red)

# Model Evaluation
accuracy_rf_red = accuracy_score(y_test_red, y_pred_rf_red)
accuracy_gb_red = accuracy_score(y_test_red, y_pred_gb_red)
accuracy_svm_red = accuracy_score(y_test_red, y_pred_svm_red)

precision_rf_red = precision_score(y_test_red, y_pred_rf_red)
precision_gb_red = precision_score(y_test_red, y_pred_gb_red)
precision_svm_red = precision_score(y_test_red, y_pred_svm_red)

recall_rf_red = recall_score(y_test_red, y_pred_rf_red)
recall_gb_red = recall_score(y_test_red, y_pred_gb_red)
recall_svm_red = recall_score(y_test_red, y_pred_svm_red)

f1_rf_red = f1_score(y_test_red, y_pred_rf_red)
f1_gb_red = f1_score(y_test_red, y_pred_gb_red)
f1_svm_red = f1_score(y_test_red, y_pred_svm_red)

y_prob_rf_red = random_forest_clf_red.predict_proba(X_test_red)[:, 1]
fpr_rf_red, tpr_rf_red, _ = roc_curve(y_test_red, y_prob_rf_red)
roc_auc_rf_red = auc(fpr_rf_red, tpr_rf_red)

y_prob_gb_red = gradient_boosting_clf_red.predict_proba(X_test_red)[:, 1]
fpr_gb_red, tpr_gb_red, _ = roc_curve(y_test_red, y_prob_gb_red)
roc_auc_gb_red = auc(fpr_gb_red, tpr_gb_red)

y_prob_svm_red = svm_clf_red.predict_proba(X_test_red)[:, 1]
fpr_svm_red, tpr_svm_red, _ = roc_curve(y_test_red, y_prob_svm_red)
roc_auc_svm_red = auc(fpr_svm_red, tpr_svm_red)

# Print or store the results for analysis
print("Red Wine Classification Results:")
print(f"Random Forest Accuracy: {accuracy_rf_red:.4f}")
print(f"Random Forest Precision: {precision_rf_red:.4f}")
print(f"Random Forest Recall: {recall_rf_red:.4f}")
print(f"Random Forest F1-score: {f1_rf_red:.4f}")
print(f"Random Forest ROC AUC: {roc_auc_rf_red:.4f}")

print(f"\nGradient Boosting Accuracy: {accuracy_gb_red:.4f}")
print(f"Gradient Boosting Precision: {precision_gb_red:.4f}")
print(f"Gradient Boosting Recall: {recall_gb_red:.4f}")
print(f"Gradient Boosting F1-score: {f1_gb_red:.4f}")
print(f"Gradient Boosting ROC AUC: {roc_auc_gb_red:.4f}")

print(f"\nSVM Accuracy: {accuracy_svm_red:.4f}")
print(f"SVM Precision: {precision_svm_red:.4f}")
print(f"SVM Recall: {recall_svm_red:.4f}")
print(f"SVM F1-score: {f1_svm_red:.4f}")
print(f"SVM ROC AUC: {roc_auc_svm_red:.4f}")

# Plot ROC curves
plt.figure()
plt.plot(fpr_rf_red, tpr_rf_red, label='Random Forest (AUC = {:.4f})'.format(roc_auc_rf_red))
plt.plot(fpr_gb_red, tpr_gb_red, label='Gradient Boosting (AUC = {:.4f})'.format(roc_auc_gb_red))
plt.plot(fpr_svm_red, tpr_svm_red, label='SVM (AUC = {:.4f})'.format(roc_auc_svm_red))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Red Wine Classification')
plt.legend(loc='lower right')
plt.show()

