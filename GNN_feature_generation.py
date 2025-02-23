import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import optuna
import optuna.visualization as vis
import plotly.express as px


composition_df = pd.read_csv("input_regression_SR_compositions.csv", index_col=0)
atomic_properties_df = pd.read_csv("elemental_properties.csv", index_col=0)
binary_enthalpy_df = pd.read_csv("enthalpy_of_mixing.csv", index_col=0)
tg_df = pd.read_csv("crystallization_temperature.csv", index_col=0)

tg_mean = tg_df.values.mean()
tg_std = tg_df.values.std()
print(f"Target Tg Normalization: mean={tg_mean:.4f}, std={tg_std:.4f}")
tg_df_normalized = (tg_df - tg_mean) / tg_std

alloy_properties_df = pd.read_csv("input_regression_SR_alloy_properties.csv", index_col=0)

if "Alloy_ID" in alloy_properties_df.columns:  
    alloy_properties_df.set_index("Alloy_ID", inplace=True)

alloy_properties_df.index = alloy_properties_df.index.astype(str)
alloy_properties_df["Tm_avg"] = 1 / alloy_properties_df["Tm_avg"]
print("\n Alloy Properties:")
print(alloy_properties_df.head())

# Conatruct Initial Graph Representation
def construct_alloy_graphs(composition_df, atomic_properties_df, binary_enthalpy_df):
    """
    Constructs a graph for each alloy.
    Nodes: Each element with attributes (e.g., "fraction", "Rm", "XP").
    Edges: Between every pair of elements. Each edge stores:
       - "hmix", "delta_r", "delta_elec" (computed as before)
       - "weight": originally computed as (hmix+delta_r+delta_elec)*(c1*c2)
       - "comp": the product of the composition fractions (c1 * c2)
    """
    graphs = {}
    for alloy_id, row in composition_df.iterrows():
        G = nx.Graph()
        for element, fraction in row.items():
            if fraction > 0 and element in atomic_properties_df.index:
                node_features = atomic_properties_df.loc[element].to_dict()
                node_features["fraction"] = fraction
                G.add_node(element, **node_features)

        elements = list(G.nodes())
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                elem_A, elem_B = elements[i], elements[j]
                radius_A = atomic_properties_df.loc[elem_A, "Rm"]
                radius_B = atomic_properties_df.loc[elem_B, "Rm"]
                elec_A = atomic_properties_df.loc[elem_A, "XP"]
                elec_B = atomic_properties_df.loc[elem_B, "XP"]
                if elem_A in binary_enthalpy_df.index and elem_B in binary_enthalpy_df.columns:
                    hmix = binary_enthalpy_df.loc[elem_A, elem_B]
                else:
                    hmix = 0
                delta_r = abs(radius_A - radius_B) / max(radius_A, radius_B)
                delta_elec = abs(elec_A - elec_B)
                weight_orig = (hmix + delta_r + delta_elec) * (row[elem_A] * row[elem_B])
                comp = row[elem_A] * row[elem_B]
                G.add_edge(elem_A, elem_B, weight=weight_orig,
                           hmix=hmix, delta_r=delta_r, delta_elec=delta_elec,
                           comp=comp)
        graphs[alloy_id] = G
    return graphs

graphs = construct_alloy_graphs(composition_df, atomic_properties_df, binary_enthalpy_df)
print(f"Constructed {len(graphs)} Alloy Graphs!")

# Plot Initial Graph Representation
def plot_graph_with_weights(G, title="Alloy Graph"):
    fig, ax = plt.subplots(figsize=(7, 6)) 
    pos = nx.spring_layout(G, seed=42, k=0.8)
    node_colors = np.array([G.nodes[node]["Rm"] for node in G.nodes()])
    edge_weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    node_sizes = 2000 * (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min()) + 1000
    edge_weights_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, cmap=plt.cm.Set2, 
        node_color=node_colors, alpha=1
    )
    
    edges = nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=edge_weights_norm, edge_cmap=plt.cm.coolwarm, 
        width=2.5, alpha=0.85
    )
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, font_color="black", font_weight="bold")
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=14, font_color="black")
    plt.title(title, fontsize=10, fontweight="bold")
    plt.show()

first_alloy_id = list(graphs.keys())[0]
plot_graph_with_weights(graphs[first_alloy_id], title=f"Graph for Alloy {first_alloy_id}")

# Convert Grpahs to PyG format
def convert_graphs_to_pyg(graphs, tg_df):
    pyg_graphs = []
    for alloy_id, G in graphs.items():
        if alloy_id not in tg_df.index:
            print(f"Missing Tg value for {alloy_id}. Skipping.")
            continue
        pyg_data = from_networkx(G, group_node_attrs=["fraction", "Rm", "XP"],
                                  group_edge_attrs=["weight", "hmix", "delta_r", "delta_elec", "comp"])

        target_tg = torch.tensor([tg_df_normalized.loc[alloy_id].values], dtype=torch.float)
        pyg_data.y = target_tg
        pyg_graphs.append(pyg_data)
    return pyg_graphs

pyg_graphs = convert_graphs_to_pyg(graphs, tg_df_normalized)
print(f"Converted {len(pyg_graphs)} Graphs to PyTorch Geometric Format!")

# Train-Test Split Prior to Training
train_val_graphs, test_graphs = train_test_split(pyg_graphs, test_size=0.15, random_state=42)
train_graphs, val_graphs = train_test_split(train_val_graphs, test_size=0.1765, random_state=42)
print("Dataset Split Summary:")
print(f"Total graphs: {len(pyg_graphs)}")
print(f"Training graphs: {len(train_graphs)}")
print(f"Validation graphs: {len(val_graphs)}")
print(f"Test graphs: {len(test_graphs)}")

batch_size = 32
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_dim = train_graphs[0].x.shape[1]
edge_dim = train_graphs[0].edge_attr.shape[1]

# Initialize the GNN framework
class TgPredictorGNN_v2(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super(TgPredictorGNN_v2, self).__init__()
        self.w = nn.Parameter(torch.ones(3, 1))
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for layer in [self.conv1, self.conv2]:
            if hasattr(layer, 'lin'):
                nn.init.kaiming_uniform_(layer.lin.weight, nonlinearity='relu')
                if layer.lin.bias is not None:
                    layer.lin.bias.data.fill_(0)
        for fc in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(fc.weight, nonlinearity='relu')
            if fc.bias is not None:
                fc.bias.data.fill_(0)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if torch.isnan(x).any() or torch.isnan(edge_attr).any():
            print("NaN detected in input features! Skipping batch.")
            return torch.tensor(float('nan')).to(x.device)
        
        edge_attr = edge_attr.float()
        
        hmix = edge_attr[:, 1].unsqueeze(1).float()      
        delta_r = edge_attr[:, 2].unsqueeze(1).float()     
        delta_elec = edge_attr[:, 3].unsqueeze(1).float()  
        comp = edge_attr[:, 4].float()                     
        
        features = torch.cat([hmix, delta_r, delta_elec], dim=1)  
        s = torch.matmul(features, self.w).squeeze()
        s = F.softplus(s)
        new_edge_weight = s * comp
        
        x = F.relu(self.conv1(x, edge_index, edge_weight=new_edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=new_edge_weight))
        
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1)

# Hyperparameter Optimizaiton
def objective(trial):
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128, 256])
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    
    model = TgPredictorGNN_v2(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    num_epochs = 1000  
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            target = batch.y.view(-1).float()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                target = batch.y.view(-1).float()
                loss = criterion(out, target)
                total_val_loss += loss.item()
                val_batches += 1
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
        
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return avg_val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
print("Best hyperparameters:")
print(study.best_params)

best_params = study.best_params
final_hidden_channels = best_params['hidden_channels']
final_lr = best_params['lr']
final_weight_decay = best_params['weight_decay']

final_model = TgPredictorGNN_v2(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=final_hidden_channels).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)
final_criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.8, patience=200, verbose=True)

# Training with the best hyperparameters
def train_final_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=2000, patience=200):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    lrs = [] 
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            target = batch.y.view(-1).float()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_train_loss = total_loss / batch_count if batch_count > 0 else None
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                target = batch.y.view(-1).float()
                loss = criterion(out, target)
                total_val_loss += loss.item()
                val_batches += 1
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else None
        val_losses.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, lrs

final_model, train_losses, val_losses, lrs = train_final_model(final_model, train_loader, val_loader,
                                                                final_optimizer, final_criterion, scheduler,
                                                                num_epochs=2000, patience=200)


def evaluate_and_export(model, loader, filename="test_predictions.csv"):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    r2 = r2_score(trues, preds)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    print(f"Test R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

# Plot Loss Curves
sns.set_theme(style="white", palette="pastel")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', linestyle='-', linewidth=2, color='red')
plt.plot(val_losses, label='Validation Loss', linestyle='-', linewidth=2, color='blue')
plt.xlabel("Epoch", fontsize=14, fontweight='bold')
plt.ylabel("Loss", fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(False)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig("loss_curves_Tg.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()

# Parity Plots
def predict_and_plot(model, loader, filename="pred_vs_true_Tg.jpg"):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.view(-1).cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    sns.set_theme(style="white", context="talk", font_scale=1.2)
    plt.figure(figsize=(8, 8))

    plt.scatter(trues, preds,
                alpha=1.0, s=180,
                facecolors='slategrey', edgecolors='black', linewidth=1.2)

    plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()],
             'r--', linewidth=2, label='Ideal Prediction')

    plt.xlabel("True Normalized Tg", fontsize=14, fontweight='bold')
    plt.ylabel("Predicted Normalized Tg", fontsize=14, fontweight='bold')
    plt.title("Predicted vs. True Tg", fontsize=18, fontweight='bold', pad=20)

    plt.grid(False)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(filename, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()

predict_and_plot(final_model, test_loader)

# Plot Optimized Graph Representation
def plot_graph_with_new_edge_weights(G, model, title="Alloy Graph with New Edge Weights"):
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42, k=0.8)
    
    node_colors = np.array([G.nodes[node]["Rm"] for node in G.nodes()])
    node_range = node_colors.max() - node_colors.min()
    if node_range == 0:
        node_sizes = np.full_like(node_colors, 2000)
    else:
        node_sizes = 2000 * (node_colors - node_colors.min())/(node_range + 1e-6) + 1000

    new_edge_weights = []
    for u, v in G.edges():
        hmix = G[u][v]["hmix"]
        delta_r = G[u][v]["delta_r"]
        delta_elec = G[u][v]["delta_elec"]
        comp = G[u][v]["comp"]
        features = torch.tensor([hmix, delta_r, delta_elec], dtype=torch.float32)
        s = torch.matmul(features, final_model.w).item()
        new_weight = s * comp
        new_edge_weights.append(new_weight)
    
    new_edge_weights = np.array(new_edge_weights)
    if new_edge_weights.size == 0:
        edge_weights_norm = np.array([])
    else:
        edge_range = new_edge_weights.max() - new_edge_weights.min()
        if edge_range == 0:
            edge_weights_norm = np.ones_like(new_edge_weights)
        else:
            edge_weights_norm = (new_edge_weights - new_edge_weights.min())/(edge_range + 1e-6)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, cmap=plt.cm.Set2,
                           node_color=node_colors, alpha=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_weights_norm, edge_cmap=plt.cm.coolwarm,
                           width=2.5, alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=16, font_color="black", font_weight="bold")
    edge_labels = {(u,v): f"{w:.2f}" for (u,v), w in zip(G.edges(), new_edge_weights)}
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=14, font_color="black")
    
    plt.title(title, fontsize=10, fontweight="bold")
    plt.show()

first_alloy_id = list(graphs.keys())[0]
plot_graph_with_new_edge_weights(graphs[first_alloy_id], final_model,
                                 title=f"Graph for Alloy {first_alloy_id} with Learned New Edge Weights")

# Graph Embeddings Extraction and t-SNE Visualization
def extract_graph_embeddings(model, loader):
    model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x = model.conv1(batch.x, batch.edge_index, edge_weight=None)
            x = F.relu(x)
            x = model.conv2(x, batch.edge_index, edge_weight=None)
            emb = global_mean_pool(x, batch.batch)
            embeddings.append(emb.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    targets = np.concatenate(targets, axis=0).flatten()
    return embeddings, targets

embeddings, targets = extract_graph_embeddings(final_model, test_loader)

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

sns.set_theme(style="white", context="talk", font_scale=1.2)
plt.figure(figsize=(10, 8))

scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=targets.flatten(), cmap='coolwarm', 
                      s=250, edgecolors='black', linewidth=1.5, alpha=0.7)

cbar = plt.colorbar(scatter, label="Glass Transition Temperature", pad=0.01)
cbar.ax.tick_params(labelsize=19)

plt.xlabel("Dimension 1", fontsize=22, fontweight='bold', labelpad=10)
plt.ylabel("Dimension 2", fontsize=22, fontweight='bold', labelpad=10)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("tsne_projection_Tg.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()

# Embeddings Representation using PCA
sns.set_theme(style="whitegrid", context="talk", font_scale=1.3)

def extract_layer_embeddings(model, data):
    x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
    emb1 = F.relu(model.conv1(x, edge_index, edge_weight=None))
    emb2 = F.relu(model.conv2(emb1, edge_index, edge_weight=None))
    pool1 = global_mean_pool(emb1, batch)
    pool2 = global_mean_pool(emb2, batch)
    return pool1.cpu().detach().numpy(), pool2.cpu().detach().numpy()

sample_data = next(iter(test_loader))
emb1, emb2 = extract_layer_embeddings(final_model, sample_data)

pca = PCA(n_components=2)
pca_emb1 = pca.fit_transform(emb1)
pca_emb2 = pca.fit_transform(emb2)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

marker_style = dict(edgecolors='black', linewidths=2, s=400, alpha=0.85)

# Plot PCA of embeddings after Conv1
axes[0].scatter(pca_emb1[:, 0], pca_emb1[:, 1], color='blue', **marker_style)
axes[0].set_title("PCA of Embeddings after Conv1", fontsize=18, fontweight='bold', pad=15)
axes[0].set_xlabel("Principal Component 1", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Principal Component 2", fontsize=16, fontweight='bold')
axes[0].tick_params(axis='both', labelsize=14)
axes[0].grid(False)

# Plot PCA of embeddings after Conv2
axes[1].scatter(pca_emb2[:, 0], pca_emb2[:, 1], color='red', **marker_style)
axes[1].set_title("PCA of Embeddings after Conv2", fontsize=18, fontweight='bold', pad=15)
axes[1].set_xlabel("Principal Component 1", fontsize=16, fontweight='bold')
axes[1].tick_params(axis='both', labelsize=14)
axes[1].grid(False)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

plt.tight_layout(pad=2)
plt.savefig("pca_embeddings_Tg.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()

# Export GNN features
def extract_alloy_features(G, pyg_data, model, alloy_id):
    features_dict = {}
    features_dict["alloy_id"] = alloy_id

    # -------- Graph-Level Embedding Features --------
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = pyg_data.x.to(device)
        edge_index = pyg_data.edge_index.to(device)
        if not hasattr(pyg_data, 'batch') or pyg_data.batch is None:
            pyg_data.batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        batch = pyg_data.batch.to(device)
        
        emb_conv1 = F.relu(model.conv1(x, edge_index, edge_weight=None))
        emb_conv2 = F.relu(model.conv2(emb_conv1, edge_index, edge_weight=None))
        emb_final = global_mean_pool(emb_conv2, batch)
    emb_final_np = emb_final.cpu().numpy().flatten()
    features_dict["emb_final_mean"] = np.mean(emb_final_np)
    features_dict["emb_final_std"] = np.std(emb_final_np)
    features_dict["emb_final_min"] = np.min(emb_final_np)
    features_dict["emb_final_max"] = np.max(emb_final_np)

    # -------- Intermediate Node Features --------
    emb_conv1_np = emb_conv1.cpu().numpy()
    emb_conv2_np = emb_conv2.cpu().numpy()
    features_dict["conv1_node_mean"] = np.mean(emb_conv1_np)
    features_dict["conv1_node_std"] = np.std(emb_conv1_np)
    features_dict["conv2_node_mean"] = np.mean(emb_conv2_np)
    features_dict["conv2_node_std"] = np.std(emb_conv2_np)
    
    # -------- Layer-wise Embedding Difference --------
    pool_conv1 = global_mean_pool(emb_conv1, batch)
    pool_conv2 = global_mean_pool(emb_conv2, batch)
    diff = pool_conv2 - pool_conv1
    features_dict["pool_diff_norm"] = np.linalg.norm(diff.cpu().numpy())
    
    # -------- Edge Weight Statistics --------
    new_edge_weights = []
    for u, v in G.edges():
        hmix = G[u][v]["hmix"]
        delta_r = G[u][v]["delta_r"]
        delta_elec = G[u][v]["delta_elec"]
        comp = G[u][v]["comp"]
        feat = torch.tensor([hmix, delta_r, delta_elec], dtype=torch.float32)
        s = torch.matmul(feat, model.w).item()
        s = F.softplus(torch.tensor(s)).item()
        new_weight = s * comp
        new_edge_weights.append(new_weight)
    new_edge_weights = np.array(new_edge_weights)
    if new_edge_weights.size > 0:
        features_dict["edge_new_mean"] = np.mean(new_edge_weights)
        features_dict["edge_new_std"] = np.std(new_edge_weights)
        features_dict["edge_new_min"] = np.min(new_edge_weights)
        features_dict["edge_new_max"] = np.max(new_edge_weights)
    else:
        features_dict["edge_new_mean"] = np.nan
        features_dict["edge_new_std"] = np.nan
        features_dict["edge_new_min"] = np.nan
        features_dict["edge_new_max"] = np.nan

    # -------- Topological Graph Features --------
    degrees = [deg for node, deg in G.degree()]
    features_dict["avg_degree"] = np.mean(degrees) if degrees else np.nan
    features_dict["clustering_coef"] = nx.average_clustering(G) if len(G) > 0 else np.nan
    betweenness = list(nx.betweenness_centrality(G).values())
    features_dict["avg_betweenness"] = np.mean(betweenness) if betweenness else np.nan
    
    return features_dict

alloy_ids = sorted(list(graphs.keys()))
all_features = []

for alloy_id in alloy_ids:
    idx = alloy_ids.index(alloy_id)
    pyg_data = pyg_graphs[idx]
    G = graphs[alloy_id]
    feats = extract_alloy_features(G, pyg_data, final_model, alloy_id)
    all_features.append(feats)

features_df = pd.DataFrame(all_features)
features_df.to_csv("alloy_features_Tg.csv", index=False)
print("Extracted features for all alloys and saved to 'alloy_features_Tg.csv'.")

# Optimization Plots
optuna_history_fig = vis.plot_optimization_history(study)

def style_common_layout(fig, title):
    fig.update_layout(
        template='plotly_white',
        title_text=title,
        title_x=0.5,
        title_font=dict(size=10, color='black'),
        font=dict(size=16, color='black'),
        margin=dict(l=60, r=60, b=60, t=80),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=900,
        height=500,
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            showgrid=False
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            showgrid=False
        )
    )

style_common_layout(optuna_history_fig, "Hyperparameter Optimization History")
optuna_history_fig.update_xaxes(title_text='Trial')
optuna_history_fig.update_yaxes(title_text='Objective Value')

history_colors = ['#e63946', '#457b9d', '#f4a261', '#2a9d8f', '#d1495b', '#118ab2', '#ef476f']

for i, trace in enumerate(optuna_history_fig.data):
    if hasattr(trace, 'line'):
        trace.line.update(width=3, color=history_colors[i % len(history_colors)])
    if hasattr(trace, 'marker'):
        trace.marker.update(
            size=12,
            line=dict(color='black', width=1.5),
            color=history_colors[i % len(history_colors)]
        )

optuna_history_fig.show()
