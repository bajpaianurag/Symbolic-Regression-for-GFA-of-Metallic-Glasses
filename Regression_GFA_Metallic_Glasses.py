# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('input_dataset.csv')
X = data.drop(['Tg', 'Tx', 'Tl'], axis=1).values
y1 = data['Tg'].values
y2 = data['Tx'].values
y3 = data['Tl'].values

# Split dataset into train and test sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(X, y1, y2, y3, test_size=0.2, random_state=42)

regressor_models = {
    "RF": (RandomForestRegressor(), {
        'n_estimators': (10, 100),
        'max_depth': (2, 30),
        'min_samples_split': (2, 30),
        'min_samples_leaf': (1, 30),
        'max_features': (0.1, 1.0),
        'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
        'max_leaf_nodes': (10, 20)
    }),
    "Lasso": (Lasso(), {
        'alpha': (0.000001, 1.0),                     
        'selection': ['cyclic', 'random']
    }),
    "Ridge": (Ridge(), {
        'alpha': (0.000001, 10.0),                     
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }),
    "KNN": (KNeighborsRegressor(), {
        'n_neighbors': (2, 50),                      
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan']
    }),
    "GB": (GradientBoostingRegressor(), {
        'n_estimators': (10, 100),
        'learning_rate': (0.00001, 1.0),              
        'max_depth': (1, 30),                        
        'min_samples_split': (2, 30),                               
        'max_features': (0.1, 1.0),  
    })
}

## Modelling T_g, T_x or T_l

results = {}
results_pred_Tg = {}
best_models = {}

for name, (model, search_space) in regressor_models.items():
    # Use Bayesian optimization to find the best model parameters
    opt = BayesSearchCV(model, search_space, n_iter=100, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    opt.fit(X_train, y1_train)
    best_models[name] = opt.best_estimator_

    # Predictions
    y1_train_pred = opt.predict(X_train)
    y1_test_pred = opt.predict(X_test)
    Tg_pred = opt.predict(X_pred)

    # Metrics
    mse = mean_squared_error(y1_test, y1_test_pred)
    r2 = r2_score(y1_test, y1_test_pred)
    mae = mean_absolute_error(y1_test, y1_test_pred)
    
    results[name] = {
        'y1_train_pred': y1_train_pred,
        'y1_test_pred': y1_test_pred,
        'Tg_predictions': Tg_pred,
        'mse': mse,
        'r2': r2,
        'mae': mae
    }

    print(f"{name} MSE: {mse:.4f}")
    print(f"{name} R2 Score: {r2:.4f}")
    print(f"{name} MAE: {mae:.4f}")

    # Plot Parity Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y1_test.flatten(), y1_test_pred.flatten(), s=100, alpha=0.6)
    plt.plot([min(y1_test.flatten()), max(y1_test.flatten())], [min(y1_test.flatten()), max(y1_test.flatten())], 'r--', label='Ideal')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot for {name}')
    plt.legend()
    plt.savefig(f'{name}_parity_plot_Tg.png')
    plt.show()

    # Plot Residuals
    plt.figure(figsize=(12, 6))
    residuals = y1_test.flatten() - y1_test_pred.flatten()
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Residuals Plot for {name}')
    plt.savefig(f'{name}_residuals_plot_Tg.png')
    plt.show()
