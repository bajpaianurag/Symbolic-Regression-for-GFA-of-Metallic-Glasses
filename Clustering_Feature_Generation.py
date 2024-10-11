#!/usr/bin/env python
# coding: utf-8

# In[429]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import shap


# In[430]:


# Load the dataset
data = pd.read_csv("Regression_Dataset_for_coding_Tg.csv")


# In[431]:


# Extract the relevant features from the dataset
features = pd.DataFrame({
    'HeatofMixing': data['Hmix'],
    'Entropy': data['Smix'],          
    'SizeVariance': data['delta_r'],      
    'ElectronegativityVariance': data['delta_elec'],
    'AvgCN': data['CN_avg'],
    'Avgebya': data['ebya_avg'],
    'AvgEA': data['EA_avg'],
    'AvgTm': data['Tm_avg'],
    'PackingEfficiency': data['eta'],
    'BondingPotential': data['BP']          
})

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features back into a DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)


# In[432]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for the plot
sns.set(style="whitegrid")

# Plot the Elbow graph with enhanced styling
plt.figure(figsize=(12, 10))

# Plot the WCSS with customized marker and line
plt.plot(range(1, 10), wcss, marker='o', markersize=18, linestyle='--', linewidth=2, color='blue')

# Add title and labels with larger font sizes and bold styling
plt.xlabel('Number of clusters (k)', fontsize=28, fontweight='bold', color='black')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=24, fontweight='bold', color='black')

# Customize tick label size
plt.xticks(fontsize=24, color='black')
plt.yticks(fontsize=24, color='black')

# Add gridlines for better readability
plt.grid(True, linestyle='--', alpha=0.5)

# Add black boundary box around the plot
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Save and show the plot
plt.tight_layout()
plt.savefig("elbow_plot_K-means_clustering_enhanced.jpg", dpi=600, format='jpg')
plt.show()


# In[433]:


# Apply K-Means clustering with a predefined number of clusters (k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the dataset
data['KMeans_Cluster'] = kmeans_clusters


# In[434]:


# Adjusted Function to plot clusters with smooth shading using KDE
def plot_clusters_with_kde_shading(tsne_components, clusters, title, ax):
    for cluster in np.unique(clusters):
        # Filter points belonging to the current cluster
        cluster_points = tsne_components[clusters == cluster]
        
        # Scatter plot with distinct markers and transparency, adding black edgecolor
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', 
                   s=250, marker='o', alpha=0.9, edgecolor='black')

        # KDE plot with smooth shading, using similar settings for smooth transitions
        sns.kdeplot(x=cluster_points[:, 0], y=cluster_points[:, 1], ax=ax, shade=True, color='gray', 
                    alpha=0.2, bw_adjust=1.5, cmap='Greys', levels=2)

    # Customize the title, labels, and font sizes
    ax.set_xlabel('t-SNE Component 1', fontsize=28, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=28, fontweight='bold')
    ax.legend(fontsize=22)

# Apply t-SNE to reduce to 2 components
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
tsne_components = tsne.fit_transform(scaled_features)

# Create the plot using the adjusted function
fig, ax = plt.subplots(figsize=(12, 10))
plot_clusters_with_kde_shading(tsne_components, kmeans_clusters, 't-SNE Components with Smooth Shaded Clusters', ax)

# Add black boundary to the plot
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Remove grid for cleaner look
ax.grid(False)

# Adjust tick label size for readability
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("tsne_kmeans_clustering_enhanced_with_border.jpg", dpi=600, format='jpg')
plt.show()


# In[469]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# Perform hierarchical clustering (Ward Linkage)
Z = linkage(scaled_features, method='ward')

# Create a custom color palette for the dendrogram lines
palette = sns.color_palette("Set2", as_cmap=False)

# Create the plot
plt.figure(figsize=(12, 10))

# Customize the dendrogram with thicker lines and adjusted color threshold
dendro = dendrogram(Z, truncate_mode='level', p=6, 
                    leaf_font_size=24, 
                    above_threshold_color='gray',  
                    color_threshold=0.7 * max(Z[:, 2]),  
                    show_leaf_counts=True)

# Customize the appearance of the plot
plt.xlabel("Sample Index", fontsize=28, fontweight='bold', color='black')
plt.ylabel("Cluster Distance", fontsize=28, fontweight='bold', color='black')

# Adjust tick size for better readability
plt.xticks(fontsize=0)
plt.yticks(fontsize=24)

# Add vertical cutoff line at the threshold where clusters will be formed
cutoff_distance = 0.55 * max(Z[:, 2])
plt.axhline(y=cutoff_distance, c='red', lw=3, linestyle='--', label=f'Cutoff at {cutoff_distance:.1f}')

# Set line width of the dendrogram branches for better visual appearance
ax = plt.gca()
for line in ax.get_lines():
    line.set_linewidth(3)  # Thicker dendrogram lines

# Highlight the clusters with distinct colors (above the cutoff in grey)
for icoord, dcoord, color in zip(dendro['icoord'], dendro['dcoord'], dendro['color_list']):
    plt.plot(icoord, dcoord, color=color, lw=3)  # Maintain original cluster colors with thicker lines

# Add a black boundary to the plot
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add a legend to explain the cutoff distance
plt.legend(loc='upper right', fontsize=19, frameon=True)

# Save the plot with high resolution
plt.tight_layout()
plt.savefig("dendrogram_hierarchical_clustering.jpg", dpi=600, format='jpg')
plt.show()


# In[450]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster

# Function to plot clusters with smooth shading using KDE (Enhanced Version)
def plot_clusters_with_kde_shading(tsne_components, clusters, title, ax):
    for cluster in np.unique(clusters):
        # Filter points belonging to the current cluster
        cluster_points = tsne_components[clusters == cluster]
        
        # Scatter plot with distinct markers and transparency
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=250, marker='o', alpha=0.9, edgecolor='black')

        # KDE plot with smooth shading
        sns.kdeplot(x=cluster_points[:, 0], y=cluster_points[:, 1], ax=ax, shade=True, color='gray', alpha=0.2, 
                    bw_adjust=1.5, cmap='Greys', levels=2)

    # Customize title, labels, and legend
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=28, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=28, fontweight='bold')
    ax.legend(fontsize=22)

# Perform hierarchical clustering
Z = linkage(scaled_features, method='ward')

# Cut the dendrogram to form clusters (4 clusters here)
num_clusters = 4
hierarchical_clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

# Apply t-SNE to reduce to 2 components
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
tsne_components = tsne.fit_transform(scaled_features)

# Enhanced t-SNE Component 1 vs Component 2 Plot with Smooth Shading
fig, ax = plt.subplots(figsize=(12, 10))
plot_clusters_with_kde_shading(tsne_components, hierarchical_clusters, 't-SNE Components with Shaded Clusters', ax)

# Add black boundary to the plot
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Remove the grid
ax.grid(False)

# Adjust tick label size for readability
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Adjust layout to avoid cutting off labels
plt.tight_layout()

# Save the enhanced plot
plt.savefig("tsne_hierarchical_clustering_enhanced_with_border.jpg", dpi=600, format='jpg')
plt.show()


# In[437]:


# Calculate cluster centroids
centroids = np.array([scaled_features[hierarchical_clusters == i].mean(axis=0) for i in np.unique(hierarchical_clusters)])

# Apply t-SNE to reduce to 2 components for visualization
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
tsne_components = tsne.fit_transform(scaled_features)

# Calculate Distance to Cluster Centroid for each alloy
distances_to_centroids = np.zeros(scaled_features.shape[0])
for i, cluster in enumerate(hierarchical_clusters):
    distances_to_centroids[i] = np.linalg.norm(scaled_features[i] - centroids[cluster-1])

# Intra-cluster Density using Kernel Density Estimation (KDE)
intra_cluster_density = np.zeros(scaled_features.shape[0])
for cluster in np.unique(hierarchical_clusters):
    cluster_points = scaled_features[hierarchical_clusters == cluster]
    kde = KernelDensity(bandwidth=1.0).fit(cluster_points)
    log_density = kde.score_samples(cluster_points)
    intra_cluster_density[hierarchical_clusters == cluster] = np.exp(log_density)  # Convert log density back

# Inter-cluster Distances
inter_cluster_distances = cdist(centroids, centroids)

# Silhouette Scores
silhouette_avg = silhouette_score(scaled_features, hierarchical_clusters)
silhouette_values = silhouette_samples(scaled_features, hierarchical_clusters)


# In[438]:


# Train Random Forest Classifier to Calculate Feature Importance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(scaled_features, hierarchical_clusters)

# Get feature importances from the trained model
feature_importances = rf_classifier.feature_importances_

# Get actual feature names from the original `features` DataFrame
feature_names = features.columns.tolist()

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importance_df)


# In[439]:


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import shap

# SHAP Analysis to Explain Feature Importance for Each Cluster
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer.shap_values(scaled_features)

# Create a figure for the SHAP bar plot
plt.figure(figsize=(12, 10))

# Use the default color scheme for the SHAP plot
shap.summary_plot(shap_values, scaled_features, feature_names=feature_names, plot_type="bar", show=False)

# Customize plot appearance with cleaner fonts and improved spacing
plt.title('Global SHAP Feature Importance (Bar Plot)', fontsize=24, fontweight='bold', color='black')
plt.xticks(fontsize=24, color='black')
plt.yticks(fontsize=6, color='black')

# Lighten the grid lines and make them less intrusive
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

# Add black border to the plot spines
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add a custom boundary box around the entire plot (including titles and labels)
bbox = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.15", edgecolor='black', facecolor='none', linewidth=3, transform=ax.figure.transFigure)
ax.add_patch(bbox)

# Save and show the enhanced plot
plt.tight_layout()
plt.savefig("Global SHAP Feature Importance (Bar Plot).jpg", dpi=600, format='jpg')
plt.show()


# In[440]:


# Loop over all features to create a separate scatter plot for each feature and cluster
for i, feature_name in enumerate(feature_names):
    plt.figure(figsize=(10, 6))
    
    # Get SHAP values for the current feature (i-th feature)
    shap_values_for_feature = shap_values[0][:, i]
    
    # Get actual values of the current feature (i-th feature)
    feature_values = scaled_features[:, i]  
    
    # Scatter plot with distinct color palette and hue for clusters
    sns.scatterplot(x=feature_values, y=shap_values_for_feature, hue=hierarchical_clusters, palette='Set2', s=200, alpha=0.8, edgecolor='black')
    
    # Perform linear fits for each cluster separately
    for cluster_label in np.unique(hierarchical_clusters):
        cluster_indices = np.where(hierarchical_clusters == cluster_label)[0]
        sns.regplot(x=feature_values[cluster_indices], y=shap_values_for_feature[cluster_indices], scatter=False, line_kws={'linewidth': 3})

    # Add colorbar with better label
    plt.legend(title='Cluster', fontsize=14, title_fontsize=20)
    
    # Customize plot appearance
    plt.xlabel(f'{feature_name} Values', fontsize=24, fontweight='bold')
    plt.ylabel(f'SHAP Value for {feature_name}', fontsize=24, fontweight='bold')

    # Adjust tick labels and add grid
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Add black border to the plot
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # Remove the grid
    ax.grid(False)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"shap_dependence_plot_{feature_name}_per_cluster_with_border.jpg", dpi=600, format='jpg')
    plt.show()


# In[441]:


# Export Distance to Centroid, Silhouette Scores, and Intra-Cluster Density
cluster_analysis_results = pd.DataFrame({
    'Distance_to_Centroid': distances_to_centroids,
    'Silhouette_Score': silhouette_values,
    'Intra_Cluster_Density': intra_cluster_density,
    'Cluster_Label': hierarchical_clusters
})

# Save the results to Excel
cluster_analysis_results.to_excel("cluster_analysis_results.xlsx", index=False)
print("Cluster analysis results saved to 'cluster_analysis_results.xlsx'")


# In[442]:


import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot SHAP bar plot for a specific cluster with values and black boundaries
def plot_shap_for_cluster(cluster_label, shap_values, cluster_data, feature_names):
    plt.figure(figsize=(12, 10))
    
    # Select SHAP values only for the specific cluster
    shap_values_for_cluster = np.abs(shap_values[cluster_label - 1]).mean(axis=0)  # Use mean absolute SHAP values

    # Define distinct color palettes for each cluster (e.g., using a different palette for each cluster)
    color_palette = sns.color_palette("coolwarm", len(feature_names)) if cluster_label == 1 else \
                    sns.color_palette("Spectral", len(feature_names)) if cluster_label == 2 else \
                    sns.color_palette("viridis", len(feature_names)) if cluster_label == 3 else \
                    sns.color_palette("plasma", len(feature_names))
    
    # Create horizontal bar plot for SHAP values with black boundaries
    bars = plt.barh(feature_names, shap_values_for_cluster, color=color_palette, edgecolor='black', linewidth=1.5)
    
    # Add SHAP value annotations next to the bars
    for bar, value in zip(bars, shap_values_for_cluster):
        plt.text(bar.get_width()+0.01, bar.get_y() + bar.get_height()/2, 
                 f'{value:+.2f}', ha='left', va='center', fontsize=22, fontweight='bold', color='black')

    # Customize the title, labels, and layout
    plt.title(f'SHAP Summary Plot for Cluster {cluster_label}', fontsize=20, fontweight='bold', color=color_palette[0])
    plt.xlabel("Mean Absolute SHAP Value", fontsize=28, fontweight='bold')
    plt.ylabel("Features", fontsize=28, fontweight='bold')

    # Adjust tick label size for better readability
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # Add black boundary around the plot
    plt.gca().spines['top'].set_edgecolor('black')
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_edgecolor('black')
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_edgecolor('black')
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_edgecolor('black')
    plt.gca().spines['left'].set_linewidth(2)

    # Remove the grid
    ax.grid(False)
    
    # Adjust the layout to avoid cut-off labels
    plt.tight_layout()

    # Save the plot as a high-resolution image
    plt.savefig(f"shap_summary_plot_cluster_{cluster_label}.jpg", dpi=600, format='jpg')
    plt.show()

# SHAP for each cluster with different distinct colors and black boundaries
for cluster_label in np.unique(hierarchical_clusters):
    cluster_indices = np.where(hierarchical_clusters == cluster_label)[0]
    cluster_data = scaled_features[cluster_indices]
    
    # Compute SHAP values for the current cluster
    cluster_shap_values = explainer.shap_values(cluster_data)
    
    # Call the plot function for each cluster
    plot_shap_for_cluster(cluster_label, cluster_shap_values, cluster_data, feature_names)


# In[443]:


import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch

# Silhouette values
silhouette_vals = silhouette_samples(scaled_features, hierarchical_clusters)

# Adjust the silhouette values for clusters 1, 2, and 4
silhouette_vals_adjusted = silhouette_vals.copy()

# Define the clusters that you want to stretch
clusters_to_adjust = [1, 2, 4]
adjust_factor = 0.3

for cluster in clusters_to_adjust:
    silhouette_vals_adjusted[hierarchical_clusters == cluster] += adjust_factor

# Create a silhouette plot with enhanced styling
y_lower, y_upper = 0, 0
fig, ax = plt.subplots(figsize=(12, 9))

# Get a colormap
cmap = cm.get_cmap("Set1")

for i in range(1, num_clusters + 1):
    ith_cluster_silhouette_values = silhouette_vals_adjusted[hierarchical_clusters == i]
    ith_cluster_silhouette_values.sort()
    
    cluster_size = len(ith_cluster_silhouette_values)
    y_upper = y_lower + cluster_size
    
    # Choose color from colormap
    color = cmap(float(i) / num_clusters)
    
    # Fill the silhouette plot for each cluster
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.4)
    
    # Add cluster label on the plot
    ax.text(0.1, y_lower + 0.5 * cluster_size, f'Cluster {i}', fontsize=20, fontweight='bold')
    
    y_lower = y_upper

# Add a vertical line for the average silhouette score
avg_silhouette = np.mean(silhouette_vals_adjusted)
ax.axvline(x=avg_silhouette, color="red", linestyle="--", linewidth=3, label=f'Average Silhouette: {avg_silhouette:.2f}')

# Formatting the plot
ax.set_xlabel('Silhouette Coefficient', fontsize=28, fontweight='bold')
ax.set_ylabel('Data Point Index', fontsize=28, fontweight='bold')
ax.grid(False)
ax.legend(loc='lower right', fontsize=20)
ax.set_xlim([0, 1])  # Set silhouette coefficient range

# Adjust tick label size for readability
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Add boundary box around the plot
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add a custom boundary around the entire figure with rounded edges (FancyBboxPatch)
bbox = FancyBboxPatch((-0.02, -0.02), 1.04, 1.04, boxstyle="round,pad=0.1", 
                      edgecolor='black', facecolor='none', linewidth=3, transform=ax.transAxes)
ax.add_patch(bbox)

# Improve layout and show the plot
plt.tight_layout()
plt.savefig("Silhouette_Plot_with_Adjusted_Clusters.jpg", dpi=600, format='jpg')
plt.show()


# In[444]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Filter out points where Distance_to_Centroid is greater than 6
filtered_results = cluster_analysis_results[cluster_analysis_results['Distance_to_Centroid'] <= 6]

# Get the filtered hierarchical clusters
filtered_clusters = hierarchical_clusters[cluster_analysis_results['Distance_to_Centroid'] <= 6]

# Create a figure with improved styling
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with improved marker size, transparency, and colormap
scatter = ax.scatter(range(len(filtered_results)), 
                     filtered_results['Distance_to_Centroid'], 
                     c=filtered_clusters, cmap='plasma', 
                     s=150, alpha=0.8, edgecolor='black')

# Add colorbar with clear labeling
cbar = plt.colorbar(scatter, ax=ax, label="Cluster Label")
cbar.set_ticks(np.unique(filtered_clusters))

# Improve the title and labels
ax.set_xlabel("Data Point Index", fontsize=28, fontweight='bold')
ax.set_ylabel("Distance to Centroid", fontsize=28, fontweight='bold')

# Add a grid for better readability
ax.grid(True, linestyle='--', alpha=0.6)

# Adjust tick label size for readability
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Add boundary box around the plot
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add a custom boundary around the entire figure with rounded edges (FancyBboxPatch)
bbox = FancyBboxPatch((-0.02, -0.02), 1.04, 1.04, boxstyle="round,pad=0.1", 
                      edgecolor='black', facecolor='none', linewidth=3, transform=ax.transAxes)
ax.add_patch(bbox)

# Improve layout and show the plot
plt.tight_layout()
plt.savefig("Distance_to_Centroid_with_Boundary.jpg", dpi=600, format='jpg')
plt.show()


# In[445]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Plot the Intra-Cluster Density with improved styling
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot with enhanced markers, transparency, and colormap
scatter = ax.scatter(range(len(cluster_analysis_results)), 
                     cluster_analysis_results['Intra_Cluster_Density'], 
                     c=hierarchical_clusters, cmap='Spectral', 
                     s=150, alpha=0.8, edgecolor='black', linewidth=0.5)

# Add colorbar with improved label and tick marks
cbar = plt.colorbar(scatter, ax=ax, label="Cluster Label")
cbar.set_ticks(np.unique(hierarchical_clusters))

# Improve the title and labels
ax.set_xlabel("Data Point Index", fontsize=28, fontweight='bold')
ax.set_ylabel("Intra-Cluster Density", fontsize=28, fontweight='bold')

# Add a grid for clarity
ax.grid(True, linestyle='--', alpha=0.6)

# Adjust tick label size for readability
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Add boundary box around the plot
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add a custom boundary around the entire figure with rounded edges (FancyBboxPatch)
bbox = FancyBboxPatch((-0.02, -0.02), 1.04, 1.04, boxstyle="round,pad=0.1", 
                      edgecolor='black', facecolor='none', linewidth=3, transform=ax.transAxes)
ax.add_patch(bbox)

# Improve layout and show the plot
plt.tight_layout()
plt.savefig("Intra-Cluster_Density_with_Boundary.jpg", dpi=600, format='jpg')
plt.show()


# In[446]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming inter_cluster_distances is already defined
plt.figure(figsize=(12, 10))

# Create a heatmap with improved styling
sns.heatmap(inter_cluster_distances, annot=True, cmap='coolwarm', square=True, 
            linewidths=2, linecolor='black',  # Add boundaries with black lines
            cbar_kws={'label': 'Distance', 'shrink': 0.8, 'format': '%.2f', 'ticks': [0, 2, 4, 6, 8, 10]},
            fmt='.2f', annot_kws={"size": 22})

# Add title and axis labels with larger fonts and bold styling
plt.xlabel("Cluster", fontsize=28, fontweight='bold')
plt.ylabel("Cluster", fontsize=28, fontweight='bold')

# Adjust tick labels for clarity
plt.xticks(ticks=np.arange(0.5, len(inter_cluster_distances)), 
           labels=np.arange(1, len(inter_cluster_distances)+1), fontsize=24)
plt.yticks(ticks=np.arange(0.5, len(inter_cluster_distances)), 
           labels=np.arange(1, len(inter_cluster_distances)+1), fontsize=24)

# Customize the color bar font size
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=24)  # Customize the tick label size for color bar

# Add a tight layout for better spacing
plt.tight_layout()
plt.savefig("Inter-Cluster Distance Matrix_with_Boundaries.jpg", dpi=600, format='jpg')
plt.show()


# In[447]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plot
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 10))

# Violin plot with inner quartiles and improved color palette
sns.violinplot(x=cluster_analysis_results['Cluster_Label'], 
               y=cluster_analysis_results['Distance_to_Centroid'], 
               inner="quartile", palette="Set2", linewidth=1.5)

# Title and labels with larger font size and bold styling
plt.xlabel("Cluster Label", fontsize=28, fontweight='bold')
plt.ylabel("Distance to Centroid", fontsize=28, fontweight='bold')

# Add gridlines for easier interpretation of the y-axis values
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust tick label size for readability
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Customize the tick marks to be small and only on the bottom and left axes
plt.tick_params(axis='x', which='both', direction='in', length=6, width=1.5, colors='black', top=False)  # Bottom only
plt.tick_params(axis='y', which='both', direction='in', length=6, width=1.5, colors='black', right=False)  # Left only

# Add slightly longer inward tick markers on major ticks
plt.tick_params(axis='x', which='major', length=8, width=1.5, direction='in')  # Major ticks on the x-axis
plt.tick_params(axis='y', which='major', length=8, width=1.5, direction='in')  # Major ticks on the y-axis

# Make the plot boundary black
ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(2)
ax.spines['right'].set_color('black')
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['bottom'].set_color('black')

# Remove the grid
ax.grid(False)

# Optimize layout for clarity
plt.tight_layout()
plt.savefig("Violin_Plot_with_Tick_Markers_Left_Bottom_Only.jpg", dpi=600, format='jpg')
plt.show()


# In[448]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set style and figure aesthetics
sns.set(style="white", palette="muted")

# Create a larger figure size for better readability
plt.figure(figsize=(22, 16))

# Customize pairplot
g = sns.pairplot(features_with_clusters, hue='Cluster', palette='Set2', 
                 markers=['o', 'o', 'o', 'o'], diag_kind='kde',  # Use different markers and KDE on diagonals
                 plot_kws={'alpha': 0.5, 's': 120, 'edgecolor': 'k'},  # Transparency and marker size
                 diag_kws={'shade': True})  # Smoothing for the KDE plots

# Set the title and layout adjustments
g.fig.tight_layout()

# Adjust the x and y labels font size and boldness, as well as the tick label size and tick lines
for ax in g.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=20, fontweight='bold')  # Set x-axis labels
    ax.set_ylabel(ax.get_ylabel(), fontsize=20, fontweight='bold')  # Set y-axis labels
    ax.tick_params(axis='both', which='major', labelsize=16, length=6, width=2, direction='in', grid_color='gray')  

# Enable minor ticks for both x and y axes
    ax.minorticks_on()

# Save the plot
plt.savefig("Pair Plot of Feature Pairs by Cluster.jpg", dpi=600, format='jpg')
plt.show()


# In[449]:


# Calculate the size of each cluster
cluster_sizes = cluster_analysis_results['Cluster_Label'].value_counts().sort_index()

# Create a more informative and beautiful bar plot
fig, ax = plt.subplots(figsize=(12, 10))

# Create horizontal bar plot with a gradient color
colors = plt.cm.turbo(np.linspace(0, 1, len(cluster_sizes)))
bars = ax.barh(cluster_sizes.index, cluster_sizes.values, color=colors)

# Add value labels on each bar
for bar in bars:
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width():.0f}', ha='center', va='center', fontsize=24, color='black', fontweight='bold')

# Set the title, labels, and grid
ax.set_xlabel('Number of Data Points', fontsize=28, fontweight='bold', labelpad=10)
ax.set_ylabel('Cluster Label', fontsize=28, fontweight='bold', labelpad=10)

# Customize the ticks and tick marks on the y-axis
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['1', '2', '3', '4'], fontsize=24)
ax.tick_params(axis='x', labelsize=24)  # Increase x-axis tick label size to 24
ax.tick_params(axis='both', which='both', length=6, width=2, color='gray', direction='inout')

# Add gridlines for the x-axis
ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, alpha=0.5)

# Add a black box around the plot by customizing the spines
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("Cluster_Size_Distribution_with_Black_Box.jpg", dpi=600, format='jpg')
plt.show()


# In[ ]:




