# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from pysr import PySRRegressor
import sympy as sp
from sympy import preview, symbols, pretty

# Load your dataset (Replace 'your_dataset.csv' with actual dataset path)
df = pd.read_csv('symbolic_dataset.csv')

# Prepare features and target (Assuming Tg, Tx, and Tl are target variables)
X = df.drop(['Tg', 'Tx', 'Tl'], axis=1)
y_Tg = df['Tg']
y_Tx = df['Tx']
y_Tl = df['Tl']
X_train, X_test, y_train_Tg, y_test_Tg = train_test_split(X, y_Tg, test_size=0.2, random_state=42)
_, _, y_train_Tx, y_test_Tx = train_test_split(X, y_Tx, test_size=0.2, random_state=42)
_, _, y_train_Tl, y_test_Tl = train_test_split(X, y_Tl, test_size=0.2, random_state=42)

# Define universal gas constant (R)
R = 8.314  # Universal gas constant in J/(mol·K)

# Create a symbolic regressor that includes all possible operators
sym_regressor_Tl = PySRRegressor(
    maxsize=20,
    niterations=500,
    binary_operators=["+", "-", "*", "/", "pow"],
    unary_operators=["exp", "log", "sqrt", "abs"],
    extra_sympy_mappings={"R": R},
    multithreading=True,
    populations=20,
    should_optimize_constants=True,  
    should_simplify=True, 
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    equation_file="equations_Tl.csv", 
    progress=True
)

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
    
    sym_regressor_Tl.fit(X_train.values, y_train_Tl.values)
    pred = sym_regressor_Tl.predict(X_test.values)
    return -mean_squared_error(y_test_Tl.values, pred)

param_bounds = {
    'population_size': (50, 500), 
    'parsimony': (1e-3, 1e-1),
    'maxsize': (10, 30), 
    'niterations': (50, 200),    
    'ncycles_per_iteration': (30, 100), 
    'weight_mutate_constant': (0.01, 1.0),  
    'weight_mutate_operator': (0.01, 1.0)   
}

# Run Bayesian optimization
optimizer = BayesianOptimization(
    f=symbolic_loss, 
    pbounds=param_bounds, 
    random_state=42
)

optimizer.maximize(init_points=4, n_iter=5)
best_params = optimizer.max['params']
print(f"Best hyperparameters: {best_params}")

# Set the symbolic regressor with the best parameters
sym_regressor_Tl.set_params(
    population_size=int(best_params['population_size']), 
    parsimony=best_params['parsimony'], 
    ncycles_per_iteration=int(best_params['ncycles_per_iteration']),
    weight_mutate_constant=best_params['weight_mutate_constant'],
    weight_mutate_operator=best_params['weight_mutate_operator']
)

sym_regressor_Tl.fit(X_train.values, y_train_Tl.values)
y_pred_Tl = sym_regressor_Tl.predict(X_test.values)
mse_Tl = mean_squared_error(y_test_Tl.values, y_pred_Tl)
print(f'MSE for Tl: {mse_Tl}')

# Get the equation with the lowest loss from the equations_ DataFrame
best_equation_row = sym_regressor_Tl.equations_.iloc[sym_regressor_Tl.equations_['loss'].idxmin()]
best_equation = best_equation_row['sympy_format']
equation_file = 'standard_symbolic_regression_best_equation_Tl.txt'
with open(equation_file, 'w') as f:
    f.write(str(best_equation))
print(f'Best equation saved to {equation_file}')

# Parity Plot for Train and Test Sets
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

# Residual Plot for Test Set
residuals = y_test_Tl.values - y_pred_Tl
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_Tl, y=residuals, lowess=True, color="purple", scatter_kws={'alpha': 0.5})
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Tl')
plt.ylabel('Residuals')
plt.title('Residuals Plot for Test Set')
plt.savefig(f"residuals_plot_standard_symbolic_regressor_test_Tl.jpg", dpi=600, format='jpg')
plt.show()

# Loss vs. Generations plot
sns.set(style="whitegrid", context="talk")
equations_log = sym_regressor_Tl.equations_  
loss_reduction = equations_log['loss'].values
generations = np.arange(len(loss_reduction))

plt.figure(figsize=(12, 8))
plt.plot(generations, loss_reduction, color='red', marker='o', markersize=15, linestyle='-', linewidth=2, alpha=0.8, label='Loss Reduction')
plt.grid(False)
plt.minorticks_on()
plt.xlabel('Generations', fontsize=16, fontweight='bold', color='#2c3e50')
plt.ylabel('Loss Reduction (Fitness)', fontsize=16, fontweight='bold', color='#2c3e50')
plt.title('Loss Reduction Over Generations', fontsize=18, fontweight='bold', color='#34495e')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['left'].set_color('#2c3e50')
plt.gca().spines['bottom'].set_color('#2c3e50')
plt.legend(fontsize=14, loc='best')

for i, txt in enumerate(loss_reduction):
    if i % 10 == 0:
        plt.annotate(f'{txt:.4f}', (generations[i], loss_reduction[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='#1f77b4')

plt.tight_layout()
plt.savefig(f"Loss_Reduction_Over_Generation_standard_symbolic_regressor_Tl.jpg", dpi=600, format='jpg')
plt.show()


# Score vs Complexity Plot
equations_log = sym_regressor_Tl.equations_
complexity = equations_log['complexity'].values
score = equations_log['loss'].values 

plt.figure(figsize=(12, 8))
plt.plot(complexity, score, color='green', marker='o', linestyle='-', linewidth=2, markersize=15, alpha=0.8)
plt.xlabel('Complexity', fontsize=16, fontweight='bold', color='#2c3e50')
plt.ylabel('Score (Loss)', fontsize=16, fontweight='bold', color='#2c3e50')
plt.title('Score vs Complexity Plot', fontsize=18, fontweight='bold', color='#34495e')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['left'].set_color('#2c3e50')
plt.gca().spines['bottom'].set_color('#2c3e50')
plt.tight_layout()
plt.savefig(f"Score_vs_Complexity_Plot_standard_symbolic_regressor_Tl.jpg", dpi=600, format='jpg')
plt.show()


# Get the final equation from the symbolic regression model
final_eq = sym_regressor_Tl.sympy()
equation_string = sp.latex(final_eq)

plt.figure(figsize=(10, 2))
plt.text(0.5, 0.5, f"${equation_string}$", horizontalalignment='center', verticalalignment='center', fontsize=20)
plt.axis('off')  # Hide axes
plt.tight_layout()
plt.savefig("final_equation_standard_symbolic_regressor_Tl.png", dpi=600, format='png')
plt.show()
