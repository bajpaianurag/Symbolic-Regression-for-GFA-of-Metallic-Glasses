import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import cm
import csv
import os

# Load the dataset
df = pd.read_csv('regression_data_clusterwise.csv')

# Split the data into features and target (transformation temperatures)
X = df.drop(['Cluster_Label', 'Tg', 'Tx', 'Tl'], axis=1)  # Features (composition, atomic features, etc.)
y_Tg = df['Tg']  
y_Tx = df['Tx']  
y_Tl = df['Tl']  

# Cluster labels for splitting the data by clusters
clusters = df['Cluster_Label']

# Split data by clusters
clustered_data = {}
for cluster in np.unique(clusters):
    cluster_indices = clusters == cluster
    X_cluster = X[cluster_indices]
    y_cluster_Tg = y_Tg[cluster_indices]
    y_cluster_Tx = y_Tx[cluster_indices]
    y_cluster_Tl = y_Tl[cluster_indices]

    # Split the data once for each cluster and use the same split for all three temperatures
    X_train, X_test, y_train_Tg, y_test_Tg = train_test_split(X_cluster, y_cluster_Tg, test_size=0.2, random_state=42)
    _, _, y_train_Tx, y_test_Tx = train_test_split(X_cluster, y_cluster_Tx, test_size=0.2, random_state=42)
    _, _, y_train_Tl, y_test_Tl = train_test_split(X_cluster, y_cluster_Tl, test_size=0.2, random_state=42)

    clustered_data[cluster] = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_Tg': y_train_Tg,
        'y_test_Tg': y_test_Tg,
        'y_train_Tx': y_train_Tx,
        'y_test_Tx': y_test_Tx,
        'y_train_Tl': y_train_Tl,
        'y_test_Tl': y_test_Tl
    }

# Define Bayesian optimization function for Gradient Boosting Regressor
def gb_optimizer(X_train, y_train, init_points=25, n_iter=500):
    def gb_evaluate(n_estimators, learning_rate, max_depth, min_samples_split, max_features):
        model = GradientBoostingRegressor(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            max_features=max(0.1, min(1.0, max_features))
        )
        cv_score = cross_val_score(model, X_train, y_train, scoring='r2', cv=5).mean()
        return cv_score

    # Bounds for Bayesian optimization
    params = {
        'n_estimators': (10, 200),
        'learning_rate': (0.00001, 1.0),
        'max_depth': (1, 30),
        'min_samples_split': (2, 30),
        'max_features': (0.1, 1.0),
    }

    optimizer = BayesianOptimization(f=gb_evaluate, pbounds=params, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Get the best parameters
    best_params = optimizer.max['params']
    return GradientBoostingRegressor(
        n_estimators=int(best_params['n_estimators']),
        learning_rate=best_params['learning_rate'],
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        max_features=max(0.1, min(1.0, best_params['max_features']))
    )


# Fit the models for each cluster and temperature
optimized_models_Tg = {}
optimized_models_Tx = {}
optimized_models_Tl = {}

for cluster, data in clustered_data.items():
    X_train = data['X_train']
    y_train_Tg = data['y_train_Tg']
    y_train_Tx = data['y_train_Tx']
    y_train_Tl = data['y_train_Tl']
    
    # Optimize and fit for Tg
    model_Tg = gb_optimizer(X_train, y_train_Tg)
    model_Tg.fit(X_train, y_train_Tg)
    optimized_models_Tg[cluster] = model_Tg
    
    # Optimize and fit for Tx
    model_Tx = gb_optimizer(X_train, y_train_Tx)
    model_Tx.fit(X_train, y_train_Tx)
    optimized_models_Tx[cluster] = model_Tx
    
    # Optimize and fit for Tl
    model_Tl = gb_optimizer(X_train, y_train_Tl)
    model_Tl.fit(X_train, y_train_Tl)
    optimized_models_Tl[cluster] = model_Tl


# Parity and Residuals Plots
def plot_composite_parity(clustered_data, optimized_models_Tg, optimized_models_Tx, optimized_models_Tl):
    fig, axes = plt.subplots(len(clustered_data), 3, figsize=(18, len(clustered_data) * 5))
    
    # Iterate through clusters
    for i, cluster in enumerate(clustered_data.keys()):
        # Access train/test data for the current cluster
        data = clustered_data[cluster]
        X_train = data['X_train']
        X_test = data['X_test']
        
        # Tg temperature
        y_train_Tg = data['y_train_Tg']
        y_test_Tg = data['y_test_Tg']
        y_train_Tg_pred = optimized_models_Tg[cluster].predict(X_train)
        y_test_Tg_pred = optimized_models_Tg[cluster].predict(X_test)

        # Tx temperature
        y_train_Tx = data['y_train_Tx']
        y_test_Tx = data['y_test_Tx']
        y_train_Tx_pred = optimized_models_Tx[cluster].predict(X_train)
        y_test_Tx_pred = optimized_models_Tx[cluster].predict(X_test)

        # Tl temperature
        y_train_Tl = data['y_train_Tl']
        y_test_Tl = data['y_test_Tl']
        y_train_Tl_pred = optimized_models_Tl[cluster].predict(X_train)
        y_test_Tl_pred = optimized_models_Tl[cluster].predict(X_test)
        
        # Define subplots for each temperature
        temp_names = ['Tg', 'Tx', 'Tl']
        y_train_preds = [y_train_Tg_pred, y_train_Tx_pred, y_train_Tl_pred]
        y_test_preds = [y_test_Tg_pred, y_test_Tx_pred, y_test_Tl_pred]
        y_trains = [y_train_Tg, y_train_Tx, y_train_Tl]
        y_tests = [y_test_Tg, y_test_Tx, y_test_Tl]
        
        for j, temp_name in enumerate(temp_names):
            ax = axes[i, j]

            # Plot Parity
            sns.scatterplot(x=y_trains[j], y=y_train_preds[j], ax=ax, color='blue', label='Train', s=100, alpha=0.6, edgecolor='black')
            sns.scatterplot(x=y_tests[j], y=y_test_preds[j], ax=ax, color='orange', label='Test', s=100, alpha=0.6, edgecolor='black')
            ax.plot([min(y_tests[j]), max(y_tests[j])], [min(y_tests[j]), max(y_tests[j])], color='green', lw=2, label='Perfect Fit')
            ax.set_title(f'Parity Plot - {temp_name} - Cluster {cluster}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("composite_parity_plot.jpg", dpi=600)
    plt.show()

def plot_composite_residuals(clustered_data, optimized_models_Tg, optimized_models_Tx, optimized_models_Tl):
    fig, axes = plt.subplots(len(clustered_data), 3, figsize=(18, len(clustered_data) * 5))

    # Iterate through clusters
    for i, cluster in enumerate(clustered_data.keys()):
        # Access train/test data for the current cluster
        data = clustered_data[cluster]
        X_train = data['X_train']
        X_test = data['X_test']
        
        # Tg temperature
        y_train_Tg = data['y_train_Tg']
        y_test_Tg = data['y_test_Tg']
        y_train_Tg_pred = optimized_models_Tg[cluster].predict(X_train)
        y_test_Tg_pred = optimized_models_Tg[cluster].predict(X_test)

        # Tx temperature
        y_train_Tx = data['y_train_Tx']
        y_test_Tx = data['y_test_Tx']
        y_train_Tx_pred = optimized_models_Tx[cluster].predict(X_train)
        y_test_Tx_pred = optimized_models_Tx[cluster].predict(X_test)

        # Tl temperature
        y_train_Tl = data['y_train_Tl']
        y_test_Tl = data['y_test_Tl']
        y_train_Tl_pred = optimized_models_Tl[cluster].predict(X_train)
        y_test_Tl_pred = optimized_models_Tl[cluster].predict(X_test)
        
        # Define subplots for each temperature
        temp_names = ['Tg', 'Tx', 'Tl']
        y_train_preds = [y_train_Tg_pred, y_train_Tx_pred, y_train_Tl_pred]
        y_test_preds = [y_test_Tg_pred, y_test_Tx_pred, y_test_Tl_pred]
        y_trains = [y_train_Tg, y_train_Tx, y_train_Tl]
        y_tests = [y_test_Tg, y_test_Tx, y_test_Tl]
        
        for j, temp_name in enumerate(temp_names):
            ax = axes[i, j]

            # Plot Residuals
            sns.scatterplot(x=y_trains[j], y=y_trains[j] - y_train_preds[j], ax=ax, color='blue', label='Train', s=100, alpha=0.6, edgecolor='black')
            sns.scatterplot(x=y_tests[j], y=y_tests[j] - y_test_preds[j], ax=ax, color='orange', label='Test', s=100, alpha=0.6, edgecolor='black')
            ax.axhline(0, color='green', lw=2, label='Zero Error')
            ax.set_title(f'Residual Plot - {temp_name} - Cluster {cluster}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
            ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("composite_residual_plot.jpg", dpi=600)
    plt.show()


# Composite Parity Plot
plot_composite_parity(clustered_data, optimized_models_Tg, optimized_models_Tx, optimized_models_Tl)

# Composite Residual Plot
plot_composite_residuals(clustered_data, optimized_models_Tg, optimized_models_Tx, optimized_models_Tl)


# SHAP Analysis for each cluster and each temperature

# Function to apply the colormap and get the corresponding colors for each feature
def apply_colormap(shap_values, colormap):
    # Normalize SHAP values for the color mapping
    norm = plt.Normalize(vmin=np.min(shap_values), vmax=np.max(shap_values))
    return cm.get_cmap(colormap)(norm(shap_values))

# Summary plot function with SHAP value annotations and enhanced styling
def shap_summary_plot(ax, shap_values, X_train, feature_names, title, colormap):
    # Calculate the mean absolute SHAP values
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    colors = apply_colormap(mean_shap_values, colormap)
    ax.barh(np.arange(len(mean_shap_values)), mean_shap_values, color=colors)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=8, color='purple')
    ax.set_xlabel('Mean Absolute SHAP Value', fontsize=10, fontweight='bold')
    ax.set_ylabel('Features', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8, labelcolor='black')
    ax.tick_params(axis='y', labelsize=8, labelcolor='black')
    ax.set_yticks(np.arange(len(mean_shap_values)))
    ax.set_yticklabels(feature_names, fontsize=8, color='black')
    for i, v in enumerate(mean_shap_values):
        ax.text(v + 0.02, i, f'+{v:.2f}', color='black', va='center', fontsize=10, fontweight='bold')
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():  # Loop through each spine (border)
        spine.set_edgecolor('black')  # Set border color to black
        spine.set_linewidth(2)        # Set the thickness of the border

# Dictionary to store SHAP values for each cluster and each temperature
shap_values_Tg = {}
shap_values_Tx = {}
shap_values_Tl = {}

# Loop through each cluster to compute SHAP values for Tg, Tx, and Tl models
for cluster, data in clustered_data.items():
    X_train = data['X_train']
    
    # SHAP for Tg
    explainer_Tg = shap.Explainer(optimized_models_Tg[cluster], X_train)
    shap_values_Tg[cluster] = explainer_Tg(X_train)
    
    # SHAP for Tx
    explainer_Tx = shap.Explainer(optimized_models_Tx[cluster], X_train)
    shap_values_Tx[cluster] = explainer_Tx(X_train)
    
    # SHAP for Tl
    explainer_Tl = shap.Explainer(optimized_models_Tl[cluster], X_train)
    shap_values_Tl[cluster] = explainer_Tl(X_train)

# Define number of clusters and temperatures
num_clusters = len(clustered_data.keys())
temperatures = ['Tg', 'Tx', 'Tl']

# Create a figure with subplots (3 rows for temperatures, columns for clusters)
fig, axes = plt.subplots(3, num_clusters, figsize=(2.5 * num_clusters, 9), sharey=True)

# Enhanced Summary Plots for SHAP values (Tg, Tx, Tl) with SHAP value annotations in a grid
for col, cluster in enumerate(clustered_data.keys()):
    # SHAP summary plot for Tg
    shap_summary_plot(
        axes[0, col],  # First row for Tg
        shap_values_Tg[cluster], 
        clustered_data[cluster]['X_train'], 
        X.columns, 
        title=f'SHAP Summary Plot for Tg - Cluster {cluster}', 
        colormap="summer"  # Change color map for Tg plots
    )

    # SHAP summary plot for Tx
    shap_summary_plot(
        axes[1, col],  # Second row for Tx
        shap_values_Tx[cluster], 
        clustered_data[cluster]['X_train'], 
        X.columns, 
        title=f'SHAP Summary Plot for Tx - Cluster {cluster}', 
        colormap="autumn"  # Change color map for Tx plots
    )

    # SHAP summary plot for Tl
    shap_summary_plot(
        axes[2, col],  # Third row for Tl
        shap_values_Tl[cluster], 
        clustered_data[cluster]['X_train'], 
        X.columns, 
        title=f'SHAP Summary Plot for Tl - Cluster {cluster}', 
        colormap="winter"  # Change color map for Tl plots
    )

plt.tight_layout()
plt.savefig("shap_summary_grid.jpg", dpi=600)
plt.show()


# SHAP dependence plots
def shap_dependence_plot_with_custom_font(shap_values, X_train, target_values, feature_name, target_name, cluster, cmap='seismic', font_size=14, font_weight='bold'):
    plt.figure(figsize=(4, 6))
    
    # SHAP dependence plot
    shap.dependence_plot(
        feature_name, 
        shap_values, 
        X_train, 
        interaction_index=None,
        show=False
    )
    
    # Use the target value as the color map
    scatter = plt.scatter(
        X_train[feature_name], 
        shap_values[:, X_train.columns.get_loc(feature_name)], 
        c=target_values, cmap=cmap, s=150, alpha=0.8, edgecolor='black'
    )
    
    plt.title(f'Dependence of {feature_name} on {target_name} for Cluster {cluster}', fontsize=12, fontweight=font_weight)
    plt.xlabel(f'{feature_name} Values', fontsize=18, fontweight=font_weight)
    plt.ylabel(f'SHAP Value for {feature_name}', fontsize=18, fontweight=font_weight)
    cbar = plt.colorbar(scatter, label=f'{target_name} Values', pad=0.02)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(f'{target_name} Values', fontsize=18, fontweight=font_weight)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.01, alpha=0.01)
    plt.tight_layout()
    plt.savefig(f"shap_dependence_{feature_name}_{target_name}_cluster_{cluster}.jpg", dpi=600)
    plt.show()

# Loop over each cluster to generate SHAP dependence plots for Tg, Tx, and Tl with customized color map and font size
for cluster, data in clustered_data.items():
    X_train = data['X_train']
    
    # SHAP for Tg
    for feature in X.columns:
        shap_dependence_plot_with_custom_font(
            shap_values_Tg[cluster].values, 
            X_train, 
            data['y_train_Tg'],  # Use Tg as the color map
            feature, 
            'Tg', 
            cluster,
            cmap='seismic',  # You can customize the color map here
            font_size=16,    # Customize font size
            font_weight='bold'  # Customize font weight
        )
    
    # SHAP for Tx
    for feature in X.columns:
        shap_dependence_plot_with_custom_font(
            shap_values_Tx[cluster].values, 
            X_train, 
            data['y_train_Tx'],  # Use Tx as the color map
            feature, 
            'Tx', 
            cluster,
            cmap='PiYG',  # Another color map
            font_size=16,    # Customize font size
            font_weight='bold'  # Customize font weight
        )
    
    # SHAP for Tl
    for feature in X.columns:
        shap_dependence_plot_with_custom_font(
            shap_values_Tl[cluster].values, 
            X_train, 
            data['y_train_Tl'],  # Use Tl as the color map
            feature, 
            'Tl', 
            cluster,
            cmap='PuOr',  # Another color map
            font_size=16,    # Customize font size
            font_weight='bold'  # Customize font weight
        )