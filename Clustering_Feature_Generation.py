#!/usr/bin/env python
# coding: utf-8

# In[174]:


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


# In[175]:


# Load the dataset
data = pd.read_csv("Regression_Dataset_for_coding_Tg.csv")


# In[176]:


# Extract the relevant features from the dataset
features = pd.DataFrame({
    'HeatofMixing': data['Hmix'],
    'Entropy': data['Smix'],          
    'SizeVariance': data['delta_r'],      
    'ElectronegativityVariance': data['delta_elec'],
    'AvgCN': data['CN_avg'],
    'Avge/a': data['ebya_avg'],
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

# Display the scaled features
print(scaled_features_df.head())


# In[177]:


wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 10), wcss, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.savefig("elbow_plot_K-means_clustering.jpg", dpi=600, format='jpg')
plt.show()


# In[178]:


# Apply K-Means clustering with a predefined number of clusters (k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the dataset
data['KMeans_Cluster'] = kmeans_clusters


# In[179]:


# Function to plot clusters with smooth shading using KDE
def plot_clusters_with_kde_shading(tsne_components, clusters, title, ax):
    for cluster in np.unique(clusters):
        # Filter points belonging to the current cluster
        cluster_points = tsne_components[clusters == cluster]
        
        # Plot the points for this cluster with a vibrant colormap and transparency
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=200, alpha=0.8, cmap='plasma')

        # Use seaborn's kdeplot to generate smooth cluster areas (without boundaries)
        sns.kdeplot(x=cluster_points[:, 0], y=cluster_points[:, 1], ax=ax, shade=True, color='gray', alpha=0.25, 
                    bw_adjust=1.2, cmap='Greys', levels=2)  # Adjusted for smoother transitions

    # Customize the title, labels, and add gridlines for clarity
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=16)
    ax.set_ylabel('t-SNE Component 2', fontsize=16)
    ax.legend(fontsize=12)

# Apply t-SNE to reduce to 2 components
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
tsne_components = tsne.fit_transform(scaled_features)

# Plot t-SNE Component 1 vs Component 2 with smooth shading using KDE
fig, ax = plt.subplots(figsize=(10, 8)) 
plot_clusters_with_kde_shading(tsne_components, kmeans_clusters, 't-SNE Components with Smooth Shaded Clusters', ax)

# Adjust layout to avoid cutting off labels
plt.tight_layout()
plt.savefig("t-SNE_K-means_clustering.jpg", dpi=600, format='jpg')
plt.show()


# In[183]:


# Step 2: Perform hierarchical clustering (Agglomerative)
# Use 'ward', 'complete', or 'average' linkage
Z = linkage(scaled_features, method='ward')  # You can also try 'complete' or 'average'

# Step 3: Plot the dendrogram to visualize the hierarchy
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Hierarchical Clustering (Ward Linkage)")
dendrogram(Z, truncate_mode='level', p=5)
clusters = fcluster(Z, t=4, criterion='maxclust')
plt.xlabel('Sample Index')
plt.ylabel('Distance (Ward)')
plt.tight_layout()
plt.savefig("dendogram_hierarchical_clustering.jpg", dpi=600, format='jpg')
plt.show()


# In[181]:


# Function to plot clusters with smooth shading using KDE
def plot_clusters_with_kde_shading(tsne_components, clusters, title, ax):
    for cluster in np.unique(clusters):
        # Filter points belonging to the current cluster
        cluster_points = tsne_components[clusters == cluster]
        
        # Plot the points for this cluster with a vibrant colormap and transparency
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=200, alpha=0.8, cmap='plasma')

        # Use seaborn's kdeplot to generate smooth cluster areas (without boundaries)
        sns.kdeplot(x=cluster_points[:, 0], y=cluster_points[:, 1], ax=ax, shade=True, color='gray', alpha=0.25, 
                    bw_adjust=1.2, cmap='Greys', levels=2)  # Adjusted for smoother transitions

    # Customize the title, labels, and add gridlines for clarity
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=16)
    ax.set_ylabel('t-SNE Component 2', fontsize=16)
    ax.legend(fontsize=12)

# Step 1: Perform hierarchical clustering
Z = linkage(scaled_features, method='ward')

# Step 2: Cut the dendrogram to form clusters (4 clusters here)
num_clusters = 4
hierarchical_clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

# Step 3: Apply t-SNE to reduce to 2 components
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
tsne_components = tsne.fit_transform(scaled_features)

# Step 4: Plot t-SNE Component 1 vs Component 2 with smooth shading using KDE
fig, ax = plt.subplots(figsize=(10, 8))
plot_clusters_with_kde_shading(tsne_components, hierarchical_clusters, 't-SNE Components with Smooth Shaded Clusters (Hierarchical Clustering)', ax)

# Adjust layout to avoid cutting off labels
plt.tight_layout()
plt.savefig("tsne_hierarchical_clustering.jpg", dpi=600, format='jpg')
plt.show()


# In[169]:


# Create a DataFrame to store the alloy system identifiers and the cluster labels
alloy_systems_clusters = pd.DataFrame({
    'Alloy_System': data.index,
    'Cluster': hierarchical_clusters
})

# Export this DataFrame to an Excel file
output_path = "alloy_systems_clusters.xlsx"
alloy_systems_clusters.to_excel(output_path, index=False)

print(f"Cluster assignments have been exported to: {output_path}")


# In[ ]:




