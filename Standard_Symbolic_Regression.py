#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from pysr import PySRRegressor
from sympy import preview, symbols, pretty


# In[16]:


# Load your dataset (Replace 'your_dataset.csv' with actual dataset path)
df = pd.read_csv('symbolic_dataset.csv')

# Prepare features and target (Assuming Tg, Tx, and Tl are target variables)
X = df.drop(['Tg', 'Tx', 'Tl'], axis=1)
y_Tg = df['Tg']
y_Tx = df['Tx']
y_Tl = df['Tl']

# Split data into training and test sets
X_train, X_test, y_train_Tg, y_test_Tg = train_test_split(X, y_Tg, test_size=0.2, random_state=42)
_, _, y_train_Tx, y_test_Tx = train_test_split(X, y_Tx, test_size=0.2, random_state=42)
_, _, y_train_Tl, y_test_Tl = train_test_split(X, y_Tl, test_size=0.2, random_state=42)


# In[17]:


# Define universal gas constant (R)
R = 8.314  # Universal gas constant in J/(mol·K)


# In[18]:


# Create a symbolic regressor that includes all possible operators
sym_regressor_Tl = PySRRegressor(
    maxsize=20,
    niterations=500,
    binary_operators=["+", "-", "*", "/", "pow"],
    unary_operators=["exp", "log", "sqrt", "abs"],
    extra_sympy_mappings={"R": R},  # Including universal gas constant
    multithreading=True,
    populations=20,
    should_optimize_constants=True,  
    should_simplify=True, 
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    equation_file="equations_Tl.csv", 
    progress=True
)


# In[19]:


# Define the function to optimize symbolic regression hyperparameters
def symbolic_loss(population_size, parsimony, maxsize, niterations, ncycles_per_iteration, weight_mutate_constant, weight_mutate_operator):
    sym_regressor_Tl.set_params(
        population_size=int(population_size),
        parsimony=parsimony,
        maxsize=int(maxsize),
        niterations=int(niterations),
        ncycles_per_iteration=int(ncycles_per_iteration),
        weight_mutate_constant=weight_mutate_constant,
        weight_mutate_operator=weight_mutate_operator
    )
    
    # Fit the symbolic regressor model
    sym_regressor_Tl.fit(X_train.values, y_train_Tl.values)
    
    # Predict on the test data
    pred = sym_regressor_Tl.predict(X_test.values)
    
    # Return the negative MSE (because Bayesian Optimization tries to maximize)
    return -mean_squared_error(y_test_Tl.values, pred)

# Define wide bounds for the Bayesian Optimization hyperparameter search space
param_bounds = {
    'population_size': (50, 500),  # Population size from small to large
    'parsimony': (1e-3, 1e-1),      # Parsimony (model simplicity penalty)
    'maxsize': (10, 30),             # Maximum number of operations in an equation
    'niterations': (50, 200),     # Number of iterations for the genetic algorithm
    'ncycles_per_iteration': (30, 100),  # Genetic cycles per iteration
    'weight_mutate_constant': (0.01, 1.0),  # Mutation probability for constants
    'weight_mutate_operator': (0.01, 1.0)   # Mutation probability for operators
}

# Run Bayesian optimization
optimizer = BayesianOptimization(
    f=symbolic_loss, 
    pbounds=param_bounds, 
    random_state=42
)

# Maximize the objective function (in this case, minimizing MSE)
optimizer.maximize(init_points=4, n_iter=5)

# Extract the best parameters from the optimizer
best_params = optimizer.max['params']
print(f"Best hyperparameters: {best_params}")


# In[20]:


# Set the symbolic regressor with the best parameters
sym_regressor_Tl.set_params(
    population_size=int(best_params['population_size']), 
    parsimony=best_params['parsimony'], 
    ncycles_per_iteration=int(best_params['ncycles_per_iteration']),
    weight_mutate_constant=best_params['weight_mutate_constant'],
    weight_mutate_operator=best_params['weight_mutate_operator']
)

# Train the symbolic regressor for Tl
sym_regressor_Tl.fit(X_train.values, y_train_Tl.values)

# Predict on the test set
y_pred_Tl = sym_regressor_Tl.predict(X_test.values)

# Calculate Mean Squared Error (MSE)
mse_Tl = mean_squared_error(y_test_Tl.values, y_pred_Tl)
print(f'MSE for Tl: {mse_Tl}')

# Create DataFrames for the training and test sets
train_data = pd.DataFrame(X_train, columns=['Feature_'+str(i) for i in range(1, X_train.shape[1]+1)])
train_data['Tl'] = y_train_Tl.values

test_data = pd.DataFrame(X_test, columns=['Feature_'+str(i) for i in range(1, X_test.shape[1]+1)])
test_data['True_Tl'] = y_test_Tl.values
test_data['Predicted_Tl'] = y_pred_Tl

# Export training and test data to Excel
output_path = 'standard_symbolic_regression_Tl_results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    train_data.to_excel(writer, sheet_name='Train_Data', index=False)
    test_data.to_excel(writer, sheet_name='Test_Data', index=False)

print(f'Training and test data saved to {output_path}')


# In[21]:


# Get the equation with the lowest loss from the equations_ DataFrame
best_equation_row = sym_regressor_Tl.equations_.iloc[sym_regressor_Tl.equations_['loss'].idxmin()]

# Extract the SymPy equation
best_equation = best_equation_row['sympy_format']

# Save the best equation to a text file
equation_file = 'standard_symbolic_regression_best_equation_Tl.txt'
with open(equation_file, 'w') as f:
    f.write(str(best_equation))

print(f'Best equation saved to {equation_file}')


# In[22]:


# Plot 1: Parity Plot for Train and Test Sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_train_Tl, y=sym_regressor_Tl.predict(X_train.values), color='blue', label='Train Data')
plt.plot([min(y_train_Tl), max(y_train_Tl)], [min(y_train_Tl), max(y_train_Tl)], color='red', linestyle='--')
plt.xlabel('True Tl')
plt.ylabel('Predicted Tl')
plt.title('Train Set Parity Plot')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_Tl, y=y_pred_Tl, color='green', label='Test Data')
plt.plot([min(y_test_Tl), max(y_test_Tl)], [min(y_test_Tl), max(y_test_Tl)], color='red', linestyle='--')
plt.xlabel('True Tl')
plt.ylabel('Predicted Tl')
plt.title('Test Set Parity Plot')
plt.legend()
plt.tight_layout()
plt.savefig(f"parity_plot_standard_symbolic_regressor_train_test_Tl.jpg", dpi=600, format='jpg')
plt.show()


# In[23]:


# Plot 2: Residual Plot for Test Set
residuals = y_test_Tl.values - y_pred_Tl
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_Tl, y=residuals, lowess=True, color="purple", scatter_kws={'alpha': 0.5})
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Tl')
plt.ylabel('Residuals')
plt.title('Residuals Plot for Test Set')
plt.savefig(f"residuals_plot_standard_symbolic_regressor_test_Tl.jpg", dpi=600, format='jpg')
plt.show()


# In[24]:


# Ensure Seaborn aesthetics
sns.set(style="whitegrid", context="talk")

# Extract loss/fitness values over generations from the symbolic regressor's equations log
equations_log = sym_regressor_Tl.equations_  

# Extract loss values
loss_reduction = equations_log['loss'].values

# Extract generation numbers
generations = np.arange(len(loss_reduction))

# Create a beautiful plot for Loss Reduction Over Generations
plt.figure(figsize=(12, 8))
plt.plot(generations, loss_reduction, color='red', marker='o', markersize=15, linestyle='-', linewidth=2, alpha=0.8, label='Loss Reduction')

# Add grid and beautify axes
plt.grid(False)
plt.minorticks_on()

# Add labels and title
plt.xlabel('Generations', fontsize=16, fontweight='bold', color='#2c3e50')
plt.ylabel('Loss Reduction (Fitness)', fontsize=16, fontweight='bold', color='#2c3e50')
plt.title('Loss Reduction Over Generations', fontsize=18, fontweight='bold', color='#34495e')

# Customize ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Customize the color of the spine for better aesthetics
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['left'].set_color('#2c3e50')
plt.gca().spines['bottom'].set_color('#2c3e50')

# Add a legend
plt.legend(fontsize=14, loc='best')

# Annotate key points (optional)
for i, txt in enumerate(loss_reduction):
    if i % 10 == 0:  # Annotate every 10th point for clarity
        plt.annotate(f'{txt:.4f}', (generations[i], loss_reduction[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='#1f77b4')

# Show plot
plt.tight_layout()
plt.savefig(f"Loss_Reduction_Over_Generation_standard_symbolic_regressor_Tl.jpg", dpi=600, format='jpg')
plt.show()


# In[25]:


# Extract the complexity and score (loss/fitness) from the equations log
equations_log = sym_regressor_Tl.equations_

# Assuming 'complexity' and 'loss' or 'fitness' are columns in the equations_ DataFrame
complexity = equations_log['complexity'].values
score = equations_log['loss'].values  # or replace 'loss' with 'fitness' if that's the term used

# Create a beautiful plot for Score vs Complexity
plt.figure(figsize=(12, 8))
plt.plot(complexity, score, color='green', marker='o', linestyle='-', linewidth=2, markersize=15, alpha=0.8)

# Add labels and title
plt.xlabel('Complexity', fontsize=16, fontweight='bold', color='#2c3e50')
plt.ylabel('Score (Loss)', fontsize=16, fontweight='bold', color='#2c3e50')
plt.title('Score vs Complexity Plot', fontsize=18, fontweight='bold', color='#34495e')

# Customize the ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add grid for clarity
plt.grid(False)

# Customize the color of the spine for aesthetics
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['left'].set_color('#2c3e50')
plt.gca().spines['bottom'].set_color('#2c3e50')

# Show the plot
plt.tight_layout()
plt.savefig(f"Score_vs_Complexity_Plot_standard_symbolic_regressor_Tl.jpg", dpi=600, format='jpg')
plt.show()


# In[27]:


# Extract the number of terms (complexity) and mean squared error (accuracy)
def extract_complexity_and_mse(model):
    complexity = len(model.equations_)  # Number of terms in the final symbolic equation
    mse = mean_squared_error(y_test_Tl.values, model.predict(X_test.values))  # Mean squared error
    return complexity, mse

# Track the Pareto front of complexity and accuracy over iterations
complexities, mses = zip(*[extract_complexity_and_mse(sym_regressor_Tl)])

# Plot the Pareto front: Model Complexity vs Accuracy
plt.scatter(complexities, mses, color='blue')
plt.xlabel('Model Complexity (Number of Terms)')
plt.ylabel('Mean Squared Error')
plt.title('Pareto Front: Trade-off between Complexity and Accuracy')
plt.grid(False)
plt.savefig(f"Pareto_Plot_standard_symbolic_regressor_Tl.jpg", dpi=600, format='jpg')
plt.show()


# In[28]:


import matplotlib.pyplot as plt
import sympy as sp

# Get the final equation from the symbolic regression model
final_eq = sym_regressor_Tl.sympy()

# Extract terms from the final equation
terms = list(final_eq.free_symbols)

# Convert the coefficients to float values, using evalf() and checking if a coefficient exists
importance = []
for term in terms:
    coeff = final_eq.coeff(term).evalf()
    # Check if the coefficient can be converted to float; if not, assign it a value of 0
    try:
        importance.append(float(coeff))
    except TypeError:
        importance.append(0.0)

# Convert the terms to strings for better labeling
terms_str = [str(term) for term in terms]

# Plot term importance
plt.figure(figsize=(12, 6))
plt.bar(terms_str, importance, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Terms in Final Equation', fontsize=14, fontweight='bold')
plt.ylabel('Coefficient Magnitude', fontsize=14, fontweight='bold')
plt.title('Importance of Terms in Final Symbolic Equation', fontsize=16, fontweight='bold')

# Customize grid and spines
plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')

# Save the plot to a high-resolution image
plt.tight_layout()
plt.savefig("Importance_of_Terms_standard_symbolic_regressor_Tl.jpg", dpi=600, format='jpg')
plt.show()


# In[29]:


import matplotlib.pyplot as plt
import sympy as sp

# Get the final equation from the symbolic regression model
final_eq = sym_regressor_Tl.sympy()

# Convert the equation to a LaTeX string
equation_string = sp.latex(final_eq)

# Create a figure for displaying the equation
plt.figure(figsize=(10, 2))
plt.text(0.5, 0.5, f"${equation_string}$", horizontalalignment='center', verticalalignment='center', fontsize=20)
plt.axis('off')  # Hide axes

# Save the equation as an image
plt.tight_layout()
plt.savefig("final_equation_standard_symbolic_regressor_Tl.png", dpi=600, format='png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




