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
from scipy.stats import spearmanr

# Load the dataset
data = pd.read_csv("data_for_preprocessing.csv")

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Descriptive statistics
print("Descriptive statistics:\n", data.describe())

sns.set_style('white')
plt.figure(figsize=(12, 12))

# Set font size and line width parameters
sns.set(font_scale=1.2, rc={'lines.linewidth': 1.5})

# Create the pairplot
plot = sns.pairplot(data, diag_kind='hist', diag_kws={'color': 'maroon'}, plot_kws={'color': 'maroon'})
plot = plot.map_upper(plt.scatter, edgecolor="k", s=10)
plot = plot.map_lower(sns.kdeplot, colors="k", linewidths=0.5)
plot = plot.map_diag(sns.histplot, color="maroon", edgecolor="k")

for ax in plot.axes.flatten():
    ax.tick_params(labelsize=20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.title.set_size(24)
sns.despine()
plt.savefig('pairplot_transformed_data.png', dpi=600)
plt.show()


## Feature Selection

# Spearman Rank Correlation
corr_matrix_spearman, _ = spearmanr(data)
sns.heatmap(corr_matrix_spearman, annot=False, cmap='rainbow', vmin=-1, vmax=1)
plt.title("Spearman Rank Correlation Matrix Heatmap")
plt.show()

corr_matrix_spearman_df = pd.DataFrame(corr_matrix_spearman, columns=data.columns, index=data.columns)
output_filename = 'Spearman_correlation_matrix.xlsx'
corr_matrix_spearman_df.to_excel(output_filename, index=False)
print(f"Spearman correlation matrix saved to {output_filename}")

# RF based Feature Selection
X = data.drop(['Tg','Tx','Tl'], axis=1).values
y1 = data['Tg'].values
y2 = data['Tx'].values
y3 = data['Tl'].values

model = RandomForestRegressor(random_state=0)
model_Tg=model.fit(X, y1)
model_Tx=model.fit(X, y2)
model_Tl=model.fit(X, y3)
importances_Tg = model_Tg.feature_importances_
importances_Tx = model_Tx.feature_importances_
importances_Tl = model_Tl.feature_importances_

# Permutation Importance
feature_names = ['Hmix','Smix','delta_r','delta_elec','CN_avg','ebya_avg','EA_avg','Tm_avg','eta','BP']

X_df = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X_df, y1, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred_baseline = model.predict(X_test)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
normalized_importance = perm_importance.importances_mean / mse_baseline
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': normalized_importance})
df_importance = df_importance.sort_values(by='Importance', ascending=False)
print(df_importance)

# Exhaustive Feature Selection
num_features = X.shape[1]
all_feature_indices = list(range(num_features))
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)
results_list = []

# Iterate over all possible feature combinations
for subset_size in range(1, num_features + 1):
    feature_combinations = list(itertools.combinations(all_feature_indices, subset_size))
  
    for feature_indices in feature_combinations:
        X_train_subset = X_train[:, feature_indices]
        X_test_subset = X_test[:, feature_indices]

        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_test_subset)
        mse = mean_squared_error(y_test, y_pred)
        feature_subset = ",".join([str(i) for i in feature_indices])
        results_list.append({"Feature Subset": feature_subset, "Mean Squared Error": mse})

results_df = pd.DataFrame(results_list)
results_df.to_excel("Best_subset_based_feature_selection_results.xlsx", index=False)

num_features_in_subset = [len(subset.split(",")) for subset in results_df["Feature Subset"]]
mse_values = results_df["Mean Squared Error"]

# Create a scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(num_features_in_subset, mse_values, marker='o', s=30, alpha=0.5)
plt.title("MSE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.show()
