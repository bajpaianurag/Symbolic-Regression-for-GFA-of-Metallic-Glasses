# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
import shap
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")

# Load dataset
data = pd.read_csv('Regression_Dataset.csv')
X = data.drop(['Tg', 'Tx', 'Tl'], axis=1).values
y1 = data['Tg'].values
y2 = data['Tx'].values
y3 = data['Tl'].values

# Split dataset into train and test sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(X, y1, y2, y3, test_size=0.2, random_state=42)

regressor_models = {
    "RF": (RandomForestRegressor(), {
        'n_estimators': (10, 200),
        'max_depth': (2, 30),
        'min_samples_split': (2, 30),
        'min_samples_leaf': (1, 30),
        'max_features': (0.1, 1.0),
        'bootstrap': [True, False],
        'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
        'min_impurity_decrease': (0.0, 0.1),  
        'min_weight_fraction_leaf': (0.0, 0.5),  
        'max_leaf_nodes': (10, 50),
        'ccp_alpha': (0.0, 0.2)
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
        'n_neighbors': (2, 100),                      
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan']
    }),
    "GB": (GradientBoostingRegressor(), {
        'n_estimators': (10, 200),
        'learning_rate': (0.00001, 1.0),              
        'max_depth': (1, 30),                        
        'min_samples_split': (2, 30),                               
        'max_features': (0.1, 1.0),  
    })
}


## Modelling T_g
results = {}
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
        'mse': mse,
        'r2': r2,
        'mae': mae
    }
    
    results_pred_Tg[name] = {
    'Tg_predictions': Tg_pred
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


## SHAP Analysis
sns.set(style="whitegrid")
shap_values_dict = {}
explainer_dict = {}

for name, model in best_models.items():
    # Use TreeExplainer or KernelExplainer depending on the model
    if hasattr(model, 'predict'):
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
        
        # Save explainer and SHAP values for later use
        explainer_dict[name] = explainer
        shap_values_dict[name] = shap_values
        
        # Enhanced SHAP Summary plot (feature importance)
        plt.figure(figsize=(14, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f'SHAP Summary Plot for {name}', fontsize=20, fontweight='bold')
        plt.xlabel("Mean |SHAP Value| (Feature Importance)", fontsize=16)
        plt.ylabel("Features", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'SHAP_summary_plot_{name}.jpg', format='jpg', dpi=600)
        plt.show()

        # Enhanced Individual feature impact plot
        plt.figure(figsize=(14, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f'SHAP Feature Impact Plot for {name}', fontsize=20, fontweight='bold')
        plt.xlabel("SHAP Value (Feature Impact)", fontsize=16)
        plt.ylabel("Features", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(f'SHAP_feature_impact_plot_{name}.jpg', format='jpg', dpi=600)
        plt.show()
        
    else:
        print(f"SHAP analysis is not applicable for {name} model")

## Modelling T_x
results = {}
best_models = {}

for name, (model, search_space) in regressor_models.items():
    model = BayesSearchCV(model, search_space, n_iter=100, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    model.fit(X_train, y2_train)
    y2_train_pred = model.predict(X_train)
    y2_test_pred = model.predict(X_test)
    Tx_pred = model.predict(X_pred)
    
    mse = mean_squared_error(y2_test, y2_test_pred)
    r2 = r2_score(y2_test, y2_test_pred)
    mae = mean_absolute_error(y2_test, y2_test_pred)
    
    results[name] = {'y2_train_pred': y2_train_pred, 'y2_test_pred': y2_test_pred, 'mse': mse, 'r2': r2, 'mae': mae}
    
    results_pred_Tx[name] = {
        'Tx_predictions': Tx_pred
    }
    
    best_models[name] = model

    print(f"{name} MSE: {mse}")
    print(f"{name} R2 Score: {r2}")
    print(f"{name} MAE: {mae}")

    # Plot Parity Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y2_test.flatten(), y2_test_pred.flatten(), s=100, alpha=0.6)
    plt.plot([min(y2_test.flatten()), max(y2_test.flatten())], [min(y2_test.flatten()), max(y2_test.flatten())], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot for {name}')
    plt.savefig(f'{name}_parity_plot_Tx.png')
    plt.show()

    # Plot Residuals
    plt.figure(figsize=(12, 6))
    residuals = y2_test.flatten() - y2_test_pred.flatten()
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Residuals Plot for {name}')
    plt.savefig(f'{name}_residuals_plot_Tx.png')
    plt.show()


## Modelling T_l
results = {}
best_models = {}

for name, (model, search_space) in regressor_models.items():
    model1 = BayesSearchCV(model, search_space, n_iter=100, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    model1.fit(X_train, y3_train)
    y3_train_pred = model1.predict(X_train)
    y3_test_pred = model1.predict(X_test)    
    Tl_pred = opt.predict(X_pred)
    
    mse = mean_squared_error(y3_test, y3_test_pred)
    r2 = r2_score(y3_test, y3_test_pred)
    mae = mean_absolute_error(y3_test, y3_test_pred)
    
    results[name] = {'y3_train_pred': y3_train_pred, 'y3_test_pred': y3_test_pred, 'mse': mse, 'r2': r2, 'mae': mae}
    
    results_pred_Tl[name] = {
        'Tl_predictions': Tl_pred
    }
    
    best_models[name] = model

    print(f"{name} MSE: {mse}")
    print(f"{name} R2 Score: {r2}")
    print(f"{name} MAE: {mae}")

    # Plot Parity Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y3_test.flatten(), y3_test_pred.flatten(), s=100, alpha=0.6)
    plt.plot([min(y3_test.flatten()), max(y3_test.flatten())], [min(y3_test.flatten()), max(y3_test.flatten())], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot for {name}')
    plt.savefig(f'{name}_parity_plot_Tl.png')
    plt.show()

    # Plot Residuals
    plt.figure(figsize=(12, 6))
    residuals = y3_test.flatten() - y3_test_pred.flatten()
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Residuals Plot for {name}')
    plt.savefig(f'{name}_residuals_plot_Tl.png')
    plt.show()
