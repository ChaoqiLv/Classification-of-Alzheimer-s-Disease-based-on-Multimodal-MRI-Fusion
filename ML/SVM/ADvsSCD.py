# ============================ Import Required Libraries ============================ #
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report, matthews_corrcoef, balanced_accuracy_score)
from sklearn.ensemble import IsolationForest
import shap
from sklearn.utils import shuffle

# ============================ Custom class for output to txt file ============================ #
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def plot_and_save_confusion_matrix(cm, title, filename, normalize=False):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = '.2f'

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    classes = ['AD', 'SCD']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


sys.stdout = Logger("program_log.txt")

# ============================ Data Preparation ============================ #
df = pd.read_csv("ADvsSCD_Feature.csv")
df = shuffle(df, random_state=1)

X = df[['Component_1', 'Component_2', 'Component_3', 'Component_6', 'Component_9',
        'Component_13', 'Component_15', 'Component_16', 'Component_17', 'Component_18']]

y = df['Label'].map({'AD': 0, 'SCD': 1})
y_counts = y.value_counts()
mapped_counts = y_counts.rename(index={0: 'AD', 1: 'SCD'})
print(mapped_counts)

# ============================ Parameter Optimization Settings ============================ #
param_grid = [
    # Linear kernel specific parameters
    {
        'rfe__n_features_to_select': [10],   # Feature selection not actually used in the experiment
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough'],
        'svm__kernel': ['linear'],
        'svm__C': np.logspace(-1, 3, 20),
        'svm__class_weight': ['balanced', None]
    },
    # RBF kernel specific parameters
    {
        'rfe__n_features_to_select': [10],
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough'],
        'svm__kernel': ['rbf'],
        'svm__C': np.logspace(-1, 3, 20),
        'svm__gamma': np.logspace(-5, 1, 12),
        'svm__class_weight': ['balanced', None]
    },
    # Polynomial kernel specific parameters
    {
        'rfe__n_features_to_select': [10],
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough'],
        'svm__kernel': ['poly'],
        'svm__C': np.logspace(-1, 3, 20),
        'svm__degree': [2, 3, 4, 5],
        'svm__coef0': [0.0, 0.1, 0.5, 1.0],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'svm__class_weight': ['balanced', None]
    },
    # Sigmoid kernel specific parameters
    {
        'rfe__n_features_to_select': [10],
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough'],
        'svm__kernel': ['sigmoid'],
        'svm__C': np.logspace(-1, 3, 20),
        'svm__gamma': np.logspace(-5, 1, 12)
    }
]

random_states = 682

# ======================
# Initialize data storage
# ======================
train_metrics = {'accuracies': [], 'bacc': [], 'precisions': [], 'recalls': [], 'f1_scores': [], 'auc_scores': [],
                 'mcc': [], 'confusion_matrix': []}

test_metrics = {'accuracies': [], 'bacc': [], 'precisions': [], 'recalls': [], 'f1_scores': [], 'auc_scores': [],
                'mcc': [], 'confusion_matrix': []}

train_confusion_matrices = []
test_confusion_matrices = []
test_all_fpr = []
test_all_tpr = []
test_roc_aucs = []
train_all_fpr = []
train_all_tpr = []
train_roc_aucs = []
test_fold_reports = []
train_fold_reports = []
fold_feature_importances = []
fold_selected_features = []
fold_shap_values = []
all_feature_importance_dfs = []

# ============================ Stratified Cross-validation Process ============================ #
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=682)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n======= Fold {fold} =======")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Outlier removal
    random_seed = 682
    np.random.seed(random_seed)
    clf = IsolationForest(contamination=0.05, random_state=random_seed)
    outliers = clf.fit_predict(X_train)
    X_train = X_train[outliers == 1]
    y_train = y_train[outliers == 1]
    print(f"Training set size after outlier removal: {len(X_train)}")

    # Check label distribution
    y_train_counts = y_train.value_counts()
    mapped_train_counts = y_train_counts.rename(index={0: 'AD', 1: 'SCD'})
    print(mapped_train_counts)

    # Downsampling operation
    much_indices = y_train[y_train == 1].index
    few_indices = y_train[y_train == 0].index
    if len(much_indices) > len(few_indices):
        much_indices_downsampled = np.random.choice(much_indices, len(few_indices), replace=False)
        all_indices_downsampled = np.concatenate([much_indices_downsampled, few_indices])
        X_train = X_train.loc[all_indices_downsampled]
        y_train = y_train.loc[all_indices_downsampled]

    # Check label distribution
    y_train_counts = y_train.value_counts()
    mapped_train_counts = y_train_counts.rename(index={0: 'AD', 1: 'SCD'})
    print(f"Training set size after downsampling: ")
    print(mapped_train_counts)

    # ======= Output test set size =======
    y_test_counts = y_test.value_counts()
    mapped_test_counts = y_test_counts.rename(index={0: 'AD', 1: 'SCD'})
    print(f"Test set size: {len(X_test)}")
    print(mapped_test_counts)

    # Build pipeline
    pipeline = Pipeline([
        ('scaler', 'passthrough'),
        ('rfe', RFE(estimator=SVC(kernel='linear'), step=1)),
        ('svm', SVC(probability=True))
    ])

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        scoring='matthews_corrcoef',
        n_jobs=-1,
        random_state=random_seed
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    y_train_pred = best_model.predict(X_train)
    y_train_proba = best_model.predict_proba(X_train)[:, 1]

    # Training set metrics calculation
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_bacc = balanced_accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='macro')
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    # Add training set metrics calculation
    train_metrics['mcc'].append(train_mcc)
    train_metrics['accuracies'].append(train_accuracy)
    train_metrics['bacc'].append(train_bacc)
    train_metrics['precisions'].append(train_precision)
    train_metrics['recalls'].append(train_recall)
    train_metrics['f1_scores'].append(train_f1)
    train_metrics['auc_scores'].append(train_auc)

    # Test set metrics calculation
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_bacc = balanced_accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    # Add test set metrics calculation
    test_metrics['mcc'].append(test_mcc)
    test_metrics['bacc'].append(test_bacc)
    test_metrics['accuracies'].append(test_accuracy)
    test_metrics['precisions'].append(test_precision)
    test_metrics['recalls'].append(test_recall)
    test_metrics['f1_scores'].append(test_f1)
    test_metrics['auc_scores'].append(test_auc)

    # Add after confusion matrix calculation
    train_cm = confusion_matrix(y_train, y_train_pred)
    train_metrics['confusion_matrix'].append(train_cm)
    train_confusion_matrices.append(train_cm)

    test_cm = confusion_matrix(y_test, y_test_pred)
    test_metrics['confusion_matrix'].append(test_cm)
    test_confusion_matrices.append(test_cm)

    # ROC data collection
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_proba, pos_label=1)
    train_all_fpr.append(train_fpr)
    train_all_tpr.append(train_tpr)
    train_roc_aucs.append(train_auc)

    test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_proba, pos_label=1)
    test_all_fpr.append(test_fpr)
    test_all_tpr.append(test_tpr)
    test_roc_aucs.append(test_auc)

    # Generate classification report for each fold
    train_fold_report = classification_report(y_train, y_train_pred, target_names=['AD', 'SCD'])
    train_fold_reports.append(train_fold_report)

    test_fold_report = classification_report(y_test, y_test_pred, target_names=['AD', 'SCD'])
    test_fold_reports.append(test_fold_report)

    # Calculate SHAP values
    explainer = shap.Explainer(best_model.predict_proba, X_train)
    shap_values = explainer(X_test)
    fold_shap_values.append(shap_values)

    importance_values = np.abs(shap_values.values[:, :, 1]).mean(axis=0) if shap_values.shape[1] > 1 else np.abs(
        shap_values.values).mean(axis=0)

    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance_values.flatten()
    })

    # Sort feature importance
    sorted_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    sorted_feature_names = sorted_importance_df['Feature'].tolist()
    print("List of features sorted by importance:")
    print(sorted_feature_names)
    print("Feature importance ranking:")
    print(sorted_importance_df)

    all_feature_importance_dfs.append(sorted_importance_df)

    # ====================== Visualization: Feature Density Scatter Plot ======================
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values[:, :, 1],
        X_test,
        feature_names=X.columns,
        title=f"Fold {fold} SHAP Summary Plot",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"Fold_{fold}_SHAP_Summary_Plot.png")
    plt.pause(3)
    plt.close()

    # ====================== SHAP Bar Plot Visualization ======================
    top_ten_features = sorted_importance_df.head(10)
    top_ten_feature_names = top_ten_features['Feature'].tolist()
    top_ten_shap_values = shap_values[:, top_ten_feature_names]
    plt.figure(figsize=(10, 6))
    if shap_values.shape[1] > 1:
        shap.plots.bar(top_ten_shap_values[:, :, 1], show=False)
    else:
        shap.plots.bar(top_ten_shap_values, show=False)
    plt.title(f"Fold {fold} SHAP Bar Plot (Top 10 Features)")
    plt.tight_layout()
    plt.savefig(f"Fold_{fold}_SHAP_Bar_Plot_Top10.png")
    plt.pause(3)
    plt.close()

    # ====================== Print current fold results ======================
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Training set accuracy: {train_accuracy:.4f}")
    print(f"Training set balanced accuracy: {train_bacc:.4f}")
    print(f"Training set precision: {train_precision:.4f}")
    print(f"Training set recall: {train_recall:.4f}")
    print(f"Training set F1 score: {train_f1:.4f}")
    print(f"Training set AUC: {train_auc:.4f}")
    print(f"Training set MCC: {train_mcc:.4f}")
    print(f"Training set confusion matrix:\n{train_cm}")
    print(f"Training set classification report:\n{train_fold_report}")

    print(f"Test set accuracy: {test_accuracy:.4f}")
    print(f"Test set balanced accuracy: {test_bacc:.4f}")
    print(f"Test set precision: {test_precision:.4f}")
    print(f"Test set recall: {test_recall:.4f}")
    print(f"Test set F1 score: {test_f1:.4f}")
    print(f"Test set AUC: {test_auc:.4f}")
    print(f"Test set MCC: {test_mcc:.4f}")
    print(f"Test set confusion matrix:\n{test_cm}")
    print(f"Test set classification report:\n{test_fold_report}")

    # Calculate AUC values and store ROC curve metrics
    train_mean_fpr = np.linspace(0, 1, 100)
    train_interp_tprs = []

    for fpr, tpr in zip(train_all_fpr, train_all_tpr):
        train_interp_tpr = np.interp(train_mean_fpr, fpr, tpr)
        train_interp_tpr[0] = 0.0
        train_interp_tprs.append(train_interp_tpr)
    train_mean_tpr = np.mean(train_interp_tprs, axis=0)
    train_mean_tpr[-1] = 1.0
    train_mean_auc = np.mean(train_roc_aucs)
    test_mean_fpr = np.linspace(0, 1, 100)
    test_interp_tprs = []

    for fpr, tpr in zip(test_all_fpr, test_all_tpr):
        test_interp_tpr = np.interp(test_mean_fpr, fpr, tpr)
        test_interp_tpr[0] = 0.0
        test_interp_tprs.append(test_interp_tpr)
    test_mean_tpr = np.mean(test_interp_tprs, axis=0)
    test_mean_tpr[-1] = 1.0
    test_mean_auc = np.mean(test_roc_aucs)

# ====================== Print comprehensive metrics ======================
train_avg_auc = np.mean(train_metrics['auc_scores'])
train_avg_mcc = np.mean(train_metrics['mcc'])
train_avg_bacc = np.mean(train_metrics['bacc'])
print("\n===== Training Set Final Metrics =====")
print(f"Accuracy: {np.mean(train_metrics['accuracies']):.4f} (±{np.std(train_metrics['accuracies']):.4f})")
print(f"Balanced Accuracy: {train_avg_bacc:.4f}(±{np.std(train_metrics['bacc']):.4f})")
print(f"Precision: {np.mean(train_metrics['precisions']):.4f} (±{np.std(train_metrics['precisions']):.4f})")
print(f"Recall: {np.mean(train_metrics['recalls']):.4f} (±{np.std(train_metrics['recalls']):.4f})")
print(f"F1-score: {np.mean(train_metrics['f1_scores']):.4f} (±{np.std(train_metrics['f1_scores']):.4f})")
print(f"AUC: {train_avg_auc:.4f}(±{np.std(train_metrics['auc_scores']):.4f})")
print(f"MCC: {train_avg_mcc:.4f}(±{np.std(train_metrics['mcc']):.4f})")

test_avg_auc = np.mean(test_metrics['auc_scores'])
test_avg_mcc = np.mean(test_metrics['mcc'])
test_avg_bacc = np.mean(test_metrics['bacc'])
print("\n===== Test Set Final Metrics =====")
print(f"Accuracy: {np.mean(test_metrics['accuracies']):.4f} (±{np.std(test_metrics['accuracies']):.4f})")
print(f"Balanced Accuracy: {test_avg_bacc:.4f}(±{np.std(test_metrics['bacc']):.4f})")
print(f"Precision: {np.mean(test_metrics['precisions']):.4f} (±{np.std(test_metrics['precisions']):.4f})")
print(f"Recall: {np.mean(test_metrics['recalls']):.4f} (±{np.std(test_metrics['recalls']):.4f})")
print(f"F1-score: {np.mean(test_metrics['f1_scores']):.4f} (±{np.std(test_metrics['f1_scores']):.4f})")
print(f"AUC: {test_avg_auc:.4f}(±{np.std(test_metrics['auc_scores']):.4f})")
print(f"MCC: {test_avg_mcc:.4f}(±{np.std(test_metrics['mcc']):.4f})")

# Calculate average confusion matrix
train_mean_cm = np.mean(train_confusion_matrices, axis=0).astype(int)
plot_and_save_confusion_matrix(train_mean_cm,
                               'Average Train Confusion Matrix (Counts)',
                               'Train_Confusion_Matrix_Counts.png')
plot_and_save_confusion_matrix(train_mean_cm,
                               'Average Train Confusion Matrix (Normalized)',
                               'Train_Confusion_Matrix_Normalized.png',
                               normalize=True)

test_mean_cm = np.mean(test_confusion_matrices, axis=0).astype(int)
plot_and_save_confusion_matrix(test_mean_cm,
                               'Average Test Confusion Matrix (Counts)',
                               'Test_Confusion_Matrix_Counts.png')
plot_and_save_confusion_matrix(test_mean_cm,
                               'Average Test Confusion Matrix (Normalized)',
                               'Test_Confusion_Matrix_Normalized.png',
                               normalize=True)

# Save confusion matrix data to CSV
def save_confusion_matrix_data(cm, filename):
    df = pd.DataFrame(cm, index=['AD', 'SCD'], columns=['AD', 'SCD'])
    df.to_csv(filename)
save_confusion_matrix_data(train_mean_cm, 'Train_Confusion_Matrix_Data.csv')
save_confusion_matrix_data(test_mean_cm, 'Test_Confusion_Matrix_Data.csv')

# Plot ROC curves
plt.figure(figsize=(8, 6))
for fpr, tpr in zip(train_all_fpr, train_all_tpr):
    plt.plot(fpr, tpr, color='grey', lw=1, alpha=0.3, label='_nolegend_')
plt.plot(train_mean_fpr, train_mean_tpr, color='green', lw=2,
         label=f'Mean ROC (area = {train_mean_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('The ROC Curve for Train Set')
plt.legend(loc="lower right")
plt.show(block=False)
plt.pause(3)
plt.savefig('Train_ROC_Curve.png')
plt.close()

plt.figure(figsize=(8, 6))
for fpr, tpr in zip(test_all_fpr, test_all_tpr):
    plt.plot(fpr, tpr, color='grey', lw=1, alpha=0.3, label='_nolegend_')
plt.plot(test_mean_fpr, test_mean_tpr, color='darkorange', lw=2,
         label=f'Mean ROC (area = {test_mean_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('The ROC Curve for Test Set')
plt.legend(loc="lower right")
plt.show(block=False)
plt.pause(3)
plt.savefig('Test_ROC_Curve.png')
plt.close()

# Save ROC curve data
train_roc_data = []
for fpr, tpr, auc_score in zip(train_all_fpr, train_all_tpr, train_roc_aucs):
    fold_roc_data = pd.DataFrame({
        'FPR': fpr.round(4),
        'TPR': tpr.round(4),
        'Thresholds': np.linspace(0, 1, len(fpr)).round(4),
        'AUC': [round(auc_score, 4)] * len(fpr)
    })
    train_roc_data.append(fold_roc_data)
all_train_roc_data = pd.concat(train_roc_data, ignore_index=True)
all_train_roc_data.to_csv(
    "Train_ROC_Curve_Data.csv",
    index=False)
print("Training set ROC curve data saved as Train_ROC_Curve_Data.csv")
train_mean_roc_data = pd.DataFrame({
    'Mean_FPR': train_mean_fpr.round(4),
    'Mean_TPR': train_mean_tpr.round(4),
    'Mean_AUC': [round(train_mean_auc, 4)] * len(train_mean_fpr)
})
train_mean_roc_data.to_csv(
    "Train_Mean_ROC_Curve_Data.csv",
    index=False)
print("Training set average ROC curve data saved as Train_Mean_ROC_Curve_Data.csv")

test_roc_data = []
for fpr, tpr, auc_score in zip(test_all_fpr, test_all_tpr, test_roc_aucs):
    fold_roc_data = pd.DataFrame({
        'FPR': fpr.round(4),
        'TPR': tpr.round(4),
        'Thresholds': np.linspace(0, 1, len(fpr)).round(4),
        'AUC': [round(auc_score, 4)] * len(fpr)
    })
    test_roc_data.append(fold_roc_data)
all_test_roc_data = pd.concat(test_roc_data, ignore_index=True)
all_test_roc_data.to_csv(
    "Test_ROC_Curve_Data.csv", index=False)
print("Test set ROC curve data saved as Test_ROC_Curve_Data.csv")

# Save ROC curve data
test_mean_roc_data = pd.DataFrame({
    'Mean_FPR': test_mean_fpr.round(4),
    'Mean_TPR': test_mean_tpr.round(4),
    'Mean_AUC': [round(test_mean_auc, 4)] * len(test_mean_fpr)
})
test_mean_roc_data.to_csv(
    "Test_Mean_ROC_Curve_Data.csv",
    index=False)
print("Test set average ROC curve data saved as Test_Mean_ROC_Curve_Data.csv")

# Save feature importance results
combined_feature_importance_df = pd.concat(all_feature_importance_dfs,
                                           keys=range(1, len(all_feature_importance_dfs) + 1), names=['Fold'])
combined_feature_importance_df = combined_feature_importance_df.reset_index()

# Save to CSV file
combined_feature_importance_df.to_csv("All_Folds_Feature_Importance.csv", index=False)
print("All folds feature importance results saved as All_Folds_Feature_Importance.csv")

sys.stdout.log.close()
sys.stdout = sys.__stdout__
