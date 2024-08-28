#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


# Load data
data = pd.read_csv('MG_data_for_VAE.csv')
compositions = data.iloc[:, :48].values  
features = data.iloc[:, 48:].values


# In[8]:


# Normalize data
scaler_compositions = MinMaxScaler()
compositions_normalized = scaler_compositions.fit_transform(compositions)
scaler_features = MinMaxScaler()
features_normalized = scaler_features.fit_transform(features)
input_data = np.hstack([compositions_normalized, features_normalized])


# In[9]:


# Define the generator, discriminator, and encoder models
def build_generator(latent_dim, output_dim, layers, units):
    model = Sequential()
    for _ in range(layers):
        model.add(Dense(units, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        latent_dim = units
    model.add(Dense(output_dim, activation='tanh'))
    return model

def build_discriminator(input_dim, layers, units):
    model = Sequential()
    for _ in range(layers):
        model.add(Dense(units, input_dim=input_dim))
        model.add(LeakyReLU(alpha=0.2))
        input_dim = units
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_encoder(input_dim, latent_dim, layers, units):
    model = Sequential()
    for _ in range(layers):
        model.add(Dense(units, input_dim=input_dim))
        model.add(LeakyReLU(alpha=0.2))
        input_dim = units
    model.add(Dense(latent_dim))
    return model

def build_gan(generator, discriminator, lr):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])
    z = Input(shape=(generator.input_shape[1],))
    alloy = generator(z)
    discriminator.trainable = False
    validity = discriminator(alloy)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(lr))
    return combined


# In[10]:


# Define the hyperparameter grid (latent_dim fixed at 2)
param_grid1 = {
    'lr': [0.0001, 0.0002, 0.0005],
    'batch_size': [32, 64],
    'generator_layers': [2, 3],
    'generator_units': [128, 256],
    'discriminator_layers': [2, 3],
    'discriminator_units': [128, 256],
    'epochs': [2000] 
}


# In[11]:


# Initialize grid search
best_model = None
best_params = None
best_loss = float('inf')

# Fixed latent dimension
latent_dim = 2

# Iterate over each combination of hyperparameters
for params in ParameterGrid(param_grid):
    print(f"Testing parameters: {params}")
    
    lr = params['lr']
    batch_size = params['batch_size']
    generator_layers = params['generator_layers']
    generator_units = params['generator_units']
    discriminator_layers = params['discriminator_layers']
    discriminator_units = params['discriminator_units']
    epochs = params['epochs']
    
    generator = build_generator(latent_dim, input_data.shape[1], generator_layers, generator_units)
    discriminator = build_discriminator(input_data.shape[1], discriminator_layers, discriminator_units)
    gan = build_gan(generator, discriminator, lr)
    encoder = build_encoder(input_data.shape[1], latent_dim, generator_layers, generator_units)
    
    # Train the GAN
    half_batch = int(batch_size / 2)
    d_losses = []
    g_losses = []
    
    for epoch in range(epochs):
        idx = np.random.randint(0, input_data.shape[0], half_batch)
        real_compositions = input_data[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_compositions = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_compositions, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_compositions, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
    
    # Evaluate the final loss (or other metrics)
    final_loss = np.mean(g_losses[-100:])
    
    print(f"Final loss: {final_loss}")
    
    # Track the best model
    if final_loss < best_loss:
        best_loss = final_loss
        best_model = (generator, discriminator, gan, encoder)
        best_params = params

# Output the best parameters
print(f"Best parameters found: {best_params}")


# In[23]:


# Assuming best_model has been identified as the best combination of (generator, discriminator, gan, encoder)
_, _, _, best_encoder = best_model

# Encode the input data to get the latent dimensions
latent_dims = best_encoder.predict(input_data)

# Create a DataFrame with the latent dimensions
latent_df = pd.DataFrame(latent_dims, columns=[f"Latent_Dim_{i+1}" for i in range(latent_dims.shape[1])])

# Export the DataFrame to a CSV file
latent_df.to_csv('latent_dimensions.csv', index=False)

print("Latent dimensions exported successfully.")


# In[24]:


# Visualizing training losses for the best model
plt.figure(figsize=(10, 10))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[25]:


# Project known compositions into the latent space
real_compositions_input = Input(shape=(input_data.shape[1],))
encoded_compositions = best_model[3](real_compositions_input)
encoder_model = Model(real_compositions_input, encoded_compositions)
latent_vectors = encoder_model.predict(input_data)


# In[26]:


# Example major element threshold (adjust based on your data)
major_element_threshold = 0.3333

# Classify alloys based on which elements exceed the threshold
def classify_alloys(compositions, threshold=major_element_threshold):
    classifications = []
    major_elements_list = []
    for comp in compositions:
        major_elements = tuple(np.where(comp > threshold)[0])  # Tuple of indices of major elements
        classifications.append(major_elements)
        major_elements_list.append({f"Element {i+1}": comp[i] for i in major_elements})
    return classifications, major_elements_list

# Get the classifications and major elements
alloy_classes, major_elements_list = classify_alloys(compositions)

# Assign a unique color or marker to each unique class
unique_classes = list(set(alloy_classes))
color_map = plt.get_cmap('tab10')  # Use a colormap with enough distinct colors
colors = {cls: color_map(i % 10) for i, cls in enumerate(unique_classes)}  # Cycle colors if more than 10 classes

# List of all possible markers in matplotlib
markers = ['o', '^', 'v', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '+', 'x', '1', '2', '3', '4', '8']

markers_map = {cls: markers[i % len(markers)] for i, cls in enumerate(unique_classes)}

# Generate a summary of the symbols and related major elements
summary = pd.DataFrame({
    'Major Elements': [", ".join([f"Element {i+1}" for i in cls]) for cls in unique_classes],
    'Color': [colors[cls] for cls in unique_classes],
    'Marker': [markers_map[cls] for cls in unique_classes]
})

print("Summary of symbols and related major elements:")
print(summary)

# Create a DataFrame to hold the latent vectors, associated class, color, marker, and major elements
alloy_symbols = pd.DataFrame({
    'Latent Dim 1': [vec[0] for vec in latent_vectors],
    'Latent Dim 2': [vec[1] for vec in latent_vectors],
    'Class': alloy_classes,
    'Color': [colors[cls] for cls in alloy_classes],
    'Marker': [markers_map[cls] for cls in alloy_classes],
    'Major Elements': ["; ".join([f"{element}: {concentration:.2f}" for element, concentration in major_elements.items()])
                       for major_elements in major_elements_list]
})

# Export the alloy symbols DataFrame to a CSV file
alloy_symbols.to_csv('alloy_symbols.csv', index=False)

# Plot the latent space with colors and markers
plt.figure(figsize=(10, 7))
for vec, alloy_class in zip(latent_vectors, alloy_classes):
    plt.scatter(vec[0], vec[1], color=colors[alloy_class], marker=markers_map[alloy_class], s=100)

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.show()

# Export the summary of the symbols and related major elements to an Excel file
alloy_symbols.to_excel('alloy_classification_summary_with_elements.xlsx', index=False)

print("Alloy symbols, classifications, and major elements exported successfully.")


# In[36]:


# Specific element indices
specific_element_indices = [0, 11, 28, 40, 47]

# Adjust the generator model to only output 5 elements with a range from 0 to 40
def build_generator(latent_dim, output_dim, layers, units):
    model = Sequential()
    for _ in range(layers):
        model.add(Dense(units, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        latent_dim = units  # update input dimension for next layer
    model.add(Dense(output_dim))  # Linear activation to allow a wider range
    return model

# Define latent dimension, number of layers, and units for the generator
latent_dim = 2  
generator_layers = 2  
generator_units = 256  

# Build the generator with the specified configuration
generator = build_generator(latent_dim, len(specific_element_indices), generator_layers, generator_units)

# Generate new compositions
num_samples = 5000
noise = np.random.normal(0, 1, (num_samples, latent_dim))
new_compositions = generator.predict(noise)

# Scale the outputs to the desired range (0 to 40)
new_compositions = np.clip(new_compositions, 0, 1)  # First, clip values to ensure they're within a reasonable range
new_compositions = new_compositions * 40  # Scale to 0-40 range

# Normalize each composition so that the sum equals 100
new_compositions = 100 * (new_compositions / np.sum(new_compositions, axis=1, keepdims=True))

# Optionally, expand the compositions to a full array of 48 elements
full_compositions = np.zeros((num_samples, 48))  # Assuming the full array has 48 elements
full_compositions[:, specific_element_indices] = new_compositions

# Export the new compositions for just the five elements
new_compositions_df = pd.DataFrame(new_compositions, columns=['Cu', 'Zr', 'Ag', 'Ti', 'Ni'])
new_compositions_df.to_excel('generated_compositions_CuZrAgTiNi.xlsx', index=False)

print("Generated compositions exported to 'generated_compositions_CuZrAgTiNi.xlsx'")

# Assuming you have a trained encoder model; you can build or load it here
# Replace this with loading your actual encoder model
def build_encoder(input_dim, latent_dim, layers, units):
    model = Sequential()
    for _ in range(layers):
        model.add(Dense(units, input_dim=input_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        input_dim = units  # update input dimension for next layer
    model.add(Dense(latent_dim, activation='linear'))  # Linear activation to encode into latent space
    return model

# Rebuild or load the pre-trained encoder model
encoder_layers = 2  # Number of layers in the encoder
encoder_units = 256  # Number of units per layer
encoder = build_encoder(len(specific_element_indices), latent_dim, encoder_layers, encoder_units)

# Encode the generated compositions into the latent space
latent_space_values = encoder.predict(new_compositions)

# Plot the latent space
plt.figure(figsize=(10, 7))
plt.scatter(latent_space_values[:, 0], latent_space_values[:, 1], alpha=0.6)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space of Generated Compositions')
plt.show()

# Optionally, export the latent space values for further analysis
latent_space_df = pd.DataFrame(latent_space_values, columns=['Latent Dim 1', 'Latent Dim 2'])
latent_space_df.to_csv('generated_compositions_latent_space.csv', index=False)

print("Latent space values exported to 'generated_compositions_latent_space.csv'")


# In[ ]:




