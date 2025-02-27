# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from skopt import BayesSearchCV
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('input_dataset.csv')
X = data.drop(['class'], axis=1).values
y = data['class'].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifier models and hyperparameter spaces
classifier_models = {
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': (10, 200),
        'max_depth': (2, 30),
        'min_samples_split': (2, 30),
        'min_samples_leaf': (1, 20),
        'max_features': (0.1, 1.0),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }),
    "SVC": (SVC(), {
        'C': (0.001, 100.0),
        'gamma': ('scale', 'auto'),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }),
    "Logistic Regression": (LogisticRegression(), {
        'C': (0.001, 50.0),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {
        'n_neighbors': (2, 50),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan']
    }),
    "Gradient Boosting": (GradientBoostingClassifier(), {
        'n_estimators': (10, 200),
        'learning_rate': (1e-4, 1.0, 'log-uniform'),
        'max_depth': (1, 30),
        'min_samples_split': (2, 30),
        'max_features': (0.1, 1.0)
    })
}

# Train models using Bayesian Hyperparmeter optimization
results = {}
best_params = {}

for name, (model, search_space) in classifier_models.items():
    opt = BayesSearchCV(model, search_space, n_iter=50, cv=5, n_jobs=-1, scoring='accuracy')
    opt.fit(X_train, y_train)
    best_params[name] = opt.best_params_
    model = opt.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics for both training and test sets
    report_train = classification_report(y_train, y_train_pred, output_dict=True)
    report_test = classification_report(y_test, y_test_pred, output_dict=True)
    
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred)
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    
    auc_score_train = auc(fpr_train, tpr_train)
    auc_score_test = auc(fpr_test, tpr_test)

    # Confusion matrix for the test set
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    results[name] = {
        'report_train': report_train,
        'report_test': report_test,
        'precision_train': precision_train,
        'recall_train': recall_train,
        'fpr_train': fpr_train,
        'tpr_train': tpr_train,
        'auc_train': auc_score_train,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'fpr_test': fpr_test,
        'tpr_test': tpr_test,
        'auc_test': auc_score_test,
        'confusion_matrix': cm_test
    }

    print(f"{name} Classification Report (Test Set):")
    print(report_test)
    
    # Print best hyperparameters
    print(f"{name} Best Hyperparameters:")
    print(best_params[name])
    
    # Plot Precision-Recall and ROC curves for both train and test sets
    plt.figure(figsize=(12, 10))
    
    # Precision-Recall Curve
    plt.subplot(2, 2, 1)
    plt.plot(recall_train, precision_train, label=f'{name} Train PR Curve')
    plt.plot(recall_test, precision_test, label=f'{name} Test PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # ROC Curve
    plt.subplot(2, 2, 2)
    plt.plot(fpr_train, tpr_train, label=f'{name} Train ROC Curve (AUC = {auc_score_train:.2f})')
    plt.plot(fpr_test, tpr_test, label=f'{name} Test ROC Curve (AUC = {auc_score_test:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(2, 2, 3)
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test), rotation=45)
    plt.yticks(tick_marks, np.unique(y_test))
    
    # Adding text annotations
    thresh = cm_test.max() / 2
    for i in range(cm_test.shape[0]):
        for j in range(cm_test.shape[1]):
            plt.text(j, i, cm_test[i, j], horizontalalignment="center",
                     color="white" if cm_test[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.savefig(f'{name}_curves_cm.png')
    plt.show()
