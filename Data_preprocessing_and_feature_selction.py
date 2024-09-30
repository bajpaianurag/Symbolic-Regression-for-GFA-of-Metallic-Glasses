#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


# In[2]:


# Load the dataset
data = pd.read_csv("data_for_processing.csv")


# In[3]:


# Check for missing values
print("Missing values:\n", data.isnull().sum())


# In[4]:


# Descriptive statistics
print("Descriptive statistics:\n", data.describe())


# In[5]:


sns.set_style('white')
plt.figure(figsize=(12, 12))

# Set font size and line width parameters
sns.set(font_scale=1.2, rc={'lines.linewidth': 1.5})

# Create the pairplot with custom styling
plot = sns.pairplot(data, diag_kind='hist', diag_kws={'color': 'maroon'}, plot_kws={'color': 'maroon'})

# Set black borders for the plot
plot = plot.map_upper(plt.scatter, edgecolor="k", s=10)
plot = plot.map_lower(sns.kdeplot, colors="k", linewidths=0.5)
plot = plot.map_diag(sns.histplot, color="maroon", edgecolor="k")

# Increase font size for all axes and tick labels
for ax in plot.axes.flatten():
    ax.tick_params(labelsize=20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(24)

# Remove spines
sns.despine()

# Save the plot
plt.savefig('pairplot_transformed_data.png', dpi=600)

# Show the plot
plt.show()


# # Feature Selection

# ### Spearman Rank Correlation

# In[6]:


from scipy.stats import spearmanr

# Calculate Spearman rank correlation matrix
corr_matrix_spearman, _ = spearmanr(data)

# Create a heatmap
sns.heatmap(corr_matrix_spearman, annot=False, cmap='rainbow', vmin=-1, vmax=1)
plt.title("Spearman Rank Correlation Matrix Heatmap")
plt.show()


# In[7]:


# Convert the NumPy array to a pandas DataFrame
corr_matrix_spearman_df = pd.DataFrame(corr_matrix_spearman, columns=data.columns, index=data.columns)
output_filename = 'Spearman_correlation_matrix.xlsx'

corr_matrix_spearman_df.to_excel(output_filename, index=False)
print(f"Spearman correlation matrix saved to {output_filename}")


# ### RF based Feature Selection

# In[9]:


X = data.drop('Tg', axis=1).values
y = data['Tg'].values

# Train a Random Forest regressor
model = RandomForestRegressor(random_state=0)
model.fit(X, y)

# Compute feature importances
importances = gbr.feature_importances_


# In[10]:


importances


# ### Permutation Importance

# In[17]:


# Ensure that you provide 11 feature names
feature_names = ['H_Mix','delta_H','S_mix','delta','delta_w','CN','eba','elec','EA','Tm','eta']

# Convert X to a DataFrame
X_df = pd.DataFrame(X, columns=feature_names)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Calculate baseline Mean Squared Error (MSE) on the test set
y_pred_baseline = model.predict(X_test)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)

# Calculate permutation feature importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

# Calculate normalized importance scores
normalized_importance = perm_importance.importances_mean / mse_baseline

# Create a DataFrame to display the results with feature names
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': normalized_importance})

# Sort the DataFrame by importance score
df_importance = df_importance.sort_values(by='Importance', ascending=False)

# Print the DataFrame to display feature importance
print(df_importance)


# ## Exhaustive Feature Selection

# In[21]:


# Assuming X is a NumPy array
num_features = X.shape[1]
all_feature_indices = list(range(num_features))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a list to store the results
results_list = []

# Iterate over all possible feature combinations
for subset_size in range(1, num_features + 1):
    # Generate all possible combinations of feature indices
    feature_combinations = list(itertools.combinations(all_feature_indices, subset_size))
    
    for feature_indices in feature_combinations:
        # Select the subset of features using NumPy slicing
        X_train_subset = X_train[:, feature_indices]
        X_test_subset = X_test[:, feature_indices]
        
        # Train the model on the subset of features
        model.fit(X_train_subset, y_train)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test_subset)
        
        # Evaluate the model's performance using Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        
        # Store the results in the list
        feature_subset = ",".join([str(i) for i in feature_indices])
        results_list.append({"Feature Subset": feature_subset, "Mean Squared Error": mse})

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the results to an Excel file
results_df.to_excel("Best_subset_based_feature_selection_results.xlsx", index=False)

# Optionally display the DataFrame
print(results_df)


# In[22]:


# Extract the number of features in each subset
num_features_in_subset = [len(subset.split(",")) for subset in results_df["Feature Subset"]]

# Extract the MSE values
mse_values = results_df["Mean Squared Error"]

# Create a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(num_features_in_subset, mse_values, marker='o', s=30, alpha=0.5)
plt.title("MSE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()


# In[ ]:




