import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
import matplotlib.pyplot as plt
import seaborn as sns

# Universal constants
R = 8.31

# Load data from CSV
file_path = 'symbolic_Tx.csv'
data = pd.read_csv(file_path)

# Unit dictionary for dimensional tracking (SI Units)
units_dict = {
    'heating_rate': np.array([0, 0, -1, 0, 1]),  # K/s 
    'formation_enthalpy': np.array([2, 1, -2, -1, 0]),  # kJ/mol -> [m^2 * kg / s^2 / mol]
    'diff_formation_enthalpy': np.array([4, 2, -4, -2, 0]),  # (kJ/mol)^2 -> [m^4 * kg^2 / s^4 / mol^2]
    'mixing_entropy': np.array([0, 0, -2, 0, 1]),  # kJ/mol-K -> [kg * m^2 / s^2 / mol / K]
    'atomic_size_mismatch': np.zeros(7),  # Unitless
    'atomic_weight_diff': np.array([0, 2, 0, 0, 0]),  # (g/mol)^2 -> [kg^2 / mol^2]
    'coordination_number': np.zeros(7),  # Unitless
    'itinerant_electrons': np.zeros(7),  # Unitless
    'electronegativity': np.zeros(7),  # Unitless
    'electron_affinity': np.array([2, 1, -2, 0, 0]),  # kJ/mol -> [m^2 * kg / s^2 / mol]
    'melting_temperature': np.array([0, 0, 0, 0, 1]),  # K
    'packing_efficiency': np.zeros(7),  # Unitless 
    'crystallization_onset_temp': np.array([0, 0, 0, 0, 1])  # K -> target variable
}

# Check dimensional consistency of a formula
def is_dimensionally_consistent(formula_units, target_units):
    return np.array_equal(formula_units, target_units)

# Example usage of this function during the validation step
def dimensional_consistency_validation(features, operations, units_dict):
    result_units = np.zeros(7)  # Initialize the result unit vector
    
    # Apply operations on the features according to their units
    for i, feature in enumerate(features):
        operation = operations[i]
        if operation == 'add' or operation == 'sub':
            # Addition/Subtraction requires identical units
            result_units += units_dict[feature]
        elif operation == 'mul':
            # Multiplication adds units
            result_units += units_dict[feature]
        elif operation == 'div':
            # Division subtracts units
            result_units -= units_dict[feature]
        # You can extend this logic for more operations (like power, sqrt, etc.)
    
    # Check if the result is dimensionally consistent with the target
    return is_dimensionally_consistent(result_units, units_dict['Tl'])

# Define custom function to include R explicitly
def include_R_in_expressions(x):
    return R * x

# Add the function to the symbolic regression function set
include_R = make_function(function=include_R_in_expressions, name="include_R", arity=1)

# Define custom division function with protection against division by zero
def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.0)

protected_div = make_function(function=protected_div, name="protected_div", arity=2)

# Define custom exponential function
def exp_func(x):
    return np.exp(np.clip(x, -100, 100))  # Clipping to avoid overflow

exp_function = make_function(function=exp_func, name="exp", arity=1)

# Split the data into training and testing sets
X = data.drop(columns=['crystallization_onset_temp']) 
y = data['crystallization_onset_temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Typical Symbolic Regressor (without dimensional consistency)
sr = SymbolicRegressor(
    population_size=1000,
    generations=200,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.1,
    random_state=42,
    function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', 'sin', 'cos', 'tan', exp_function, include_R),
    const_range=(-5.0, 5.0),
    n_jobs=-1
)

# GridSearchCV for hyperparameter optimization for typical SR
param_grid = {
    'population_size': [500, 1000],
    'generations': [100, 200, 250],
    'p_crossover': [0.6, 0.7],
    'p_subtree_mutation': [0.1, 0.15],
    'p_hoist_mutation': [0.05],
    'p_point_mutation': [0.1]
}

grid_search_sr = GridSearchCV(estimator=sr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_sr.fit(X_train, y_train)

# Best estimator after GridSearchCV for typical SR
best_sr = grid_search_sr.best_estimator_

# Predict and calculate R2 score for typical SR model
y_pred_sr = best_sr.predict(X_test)
print(f'R2 Score (Typical SR): {r2_score(y_test, y_pred_sr)}')


# Define a safe exponential function that handles large values
def protected_exp(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.0)

# Create the custom function
exp_function = make_function(function=protected_exp, name='exp', arity=1)

# Custom Symbolic Regressor with Dimensional Check
class SymbolicRegressorWithDimensionalCheck(SymbolicRegressor):
    def _validate(self, programs, X, y, sample_weight, random_state):
        valid_programs = []
        for program in programs:
            # Extract the features and operations used in the program
            features = [str(symbol) for symbol in program.program if str(symbol) in units_dict]
            operations = [str(symbol) for symbol in program.program if str(symbol) in ['add', 'sub', 'mul', 'div']]
            
            # Perform the dimensional consistency check
            if dimensional_consistency_validation(features, operations, units_dict):
                valid_programs.append(program)
        
        # Proceed with the validation using only dimensionally consistent programs
        return super()._validate(valid_programs, X, y, sample_weight, random_state)

# Train the Symbolic Regressor with Dimensional Consistency (SR-DC)
sr_dc = SymbolicRegressorWithDimensionalCheck(
    population_size=1000,
    generations=200,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.1,
    random_state=42,
    function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', 'sin', 'cos', exp_function, include_R),
    const_range=(-5.0, 5.0),
    n_jobs=-1
)

grid_search_dc = GridSearchCV(estimator=sr_dc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search_dc.fit(X_train, y_train)

# Best estimator after GridSearchCV with dimensional consistency (SR-DC)
best_sr_dc = grid_search_dc.best_estimator_

# Predict and calculate R2 score for SR-DC model
y_pred_dc = best_sr_dc.predict(X_test)
print(f'R2 Score (SR-DC): {r2_score(y_test, y_pred_dc)}')

# Get the best expression for Tx from both models
best_formula_sr = best_sr._program  
best_formula_dc = best_sr_dc._program 
print(f'Best Formula for crystallization_onset_temp (Typical SR): {best_formula_sr}')
print(f'Best Formula for crystallization_onset_temp (SR-DC): {best_formula_dc}')

# Visualization 1: Predicted vs Actual Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_sr, color='blue', label='Typical SR')
plt.scatter(y_test, y_pred_dc, color='red', label='SR-DC')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Actual crystallization_onset_temp')
plt.ylabel('Predicted crystallization_onset_temp')
plt.title('Predicted vs Actual crystallization_onset_temp')
plt.legend()
plt.show()

# Visualization 2: Residuals Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_sr, y_test - y_pred_sr, color='blue', label='Typical SR')
plt.scatter(y_pred_dc, y_test - y_pred_dc, color='red', label='SR-DC')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Predicted crystallization_onset_temp')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.legend()
plt.show()

# Visualization 3: Distribution of Errors
plt.figure(figsize=(10, 5))
plt.hist(y_test - y_pred_sr, bins=20, color='blue', alpha=0.5, label='Typical SR')
plt.hist(y_test - y_pred_dc, bins=20, color='red', alpha=0.5, label='SR-DC')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.legend()
plt.show()

# Generations and R2 score tracking
generations = [10, 20, 30, 50, 70, 100, 120, 150, 180, 200]
r2_scores_sr = []
r2_scores_dc = []
complexity_sr = []
complexity_dc = []

# Function to count the number of terms (operators and operands) in the formula
def count_terms(program):
    # Convert the program to string form
    formula_str = str(program)
    # Split by space to count individual elements (operators and operands)
    terms = formula_str.split()
    # Return the number of terms
    return len(terms)

# Loop over different generations and retrain the models
for gen in generations:
    # Train Typical SR model
    sr_temp = SymbolicRegressor(
        population_size=1000,
        generations=gen,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=0,
        parsimony_coefficient=0.1,
        random_state=42,
        function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', 'sin', 'cos', exp_function, include_R),
        const_range=(-5.0, 5.0),
        n_jobs=-1
    )
    sr_temp.fit(X_train, y_train)
    y_pred_sr_temp = sr_temp.predict(X_test)
    r2_scores_sr.append(r2_score(y_test, y_pred_sr_temp))
    
    # Calculate and append the number of terms (complexity) of the formula
    complexity_sr.append(count_terms(sr_temp._program))
    
    # Train SR-DC model
    sr_dc_temp = SymbolicRegressorWithDimensionalCheck(
        population_size=1000,
        generations=gen,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=0,
        parsimony_coefficient=0.1,
        random_state=42,
        function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', 'sin', 'cos', exp_function, include_R),
        const_range=(-5.0, 5.0),
        n_jobs=-1
    )
    sr_dc_temp.fit(X_train, y_train)
    y_pred_dc_temp = sr_dc_temp.predict(X_test)
    r2_scores_dc.append(r2_score(y_test, y_pred_dc_temp))
    
    # Calculate and append the number of terms (complexity) of the formula
    complexity_dc.append(count_terms(sr_dc_temp._program))

# Ensure lengths are consistent
min_len = min(len(generations), len(r2_scores_sr), len(r2_scores_dc))

# Slice all lists to the minimum length
generations = generations[:min_len]
r2_scores_sr = r2_scores_sr[:min_len]
r2_scores_dc = r2_scores_dc[:min_len]

# Plot R2 score vs. generations for both models
plt.figure(figsize=(10, 5))
plt.plot(generations, r2_scores_sr, label='Typical SR', marker='o', color='blue')
plt.plot(generations, r2_scores_dc, label='SR-DC', marker='o', color='red')
plt.xlabel('Number of Generations')
plt.ylabel('R2 Score')
plt.title('R2 Score vs. Number of Generations')
plt.legend()
plt.show()

# Ensure lengths are consistent
min_len_sr = min(len(complexity_sr), len(r2_scores_sr))
min_len_dc = min(len(complexity_dc), len(r2_scores_dc))

# Slice the lists to the minimum length for both models
complexity_sr = complexity_sr[:min_len_sr]
r2_scores_sr = r2_scores_sr[:min_len_sr]
complexity_dc = complexity_dc[:min_len_dc]
r2_scores_dc = r2_scores_dc[:min_len_dc]

# Plot R2 score vs. model complexity for both models
plt.figure(figsize=(10, 5))
plt.plot(complexity_sr, r2_scores_sr, label='Typical SR', marker='o', color='blue')
plt.plot(complexity_dc, r2_scores_dc, label='SR-DC', marker='o', color='red')
plt.xlabel('Model Complexity (Number of Terms)')
plt.ylabel('R2 Score')
plt.title('R2 Score vs. Model Complexity')
plt.legend()
plt.show()
