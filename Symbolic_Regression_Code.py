import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Universal constants
R = 8.31

# Load data from CSV
file_path = 'symbolic_Tx.csv'
data = pd.read_csv(file_path)


# Unit dictionary for dimensional tracking (SI Units)
units_dict = {
    'heating_rate': np.array([0, 0, -1, 0, 0, 0, 1]),  # K/s -> [length, mass, time, ...]
    'formation_enthalpy': np.array([2, 1, -2, 0, -1, 0, 0]),  # kJ/mol -> [m^2 * kg / s^2 / mol]
    'diff_formation_enthalpy': np.array([4, 2, -4, 0, -2, 0, 0]),  # (kJ/mol)^2 -> [m^4 * kg^2 / s^4 / mol^2]
    'mixing_entropy': np.array([0, 0, -2, 0, 0, 0, 1]),  # kJ/mol-K -> [kg * m^2 / s^2 / mol / K]
    'atomic_size_mismatch': np.zeros(7),  # Unitless
    'atomic_weight_diff': np.array([0, 2, 0, 0, 0, 0, 0]),  # (g/mol)^2 -> [kg^2 / mol^2]
    'coordination_number': np.zeros(7),  # Unitless
    'itinerant_electrons': np.zeros(7),  # Unitless
    'electronegativity': np.zeros(7),  # Unitless
    'electron_affinity': np.array([2, 1, -2, 0, 0, 0, 0]),  # kJ/mol -> [m^2 * kg / s^2 / mol]
    'melting_temperature': np.array([0, 0, 0, 0, 0, 0, 1]),  # K
    'crystallization_onset_temp': np.array([0, 0, 0, 0, 0, 0, 1])  # K -> target variable
}

# Check dimensional consistency of a formula
def is_dimensionally_consistent(formula_units, target_units):
    return np.array_equal(formula_units, target_units)

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
include_R = make_function(function=include_R_in_expressions, name="include_R", arity=1)

# Define custom division function with protection against division by zero
def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.0)
protected_div = make_function(function=protected_div, name="protected_div", arity=2)

# Define an exponential function that handles large values
def protected_exp(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.0)
exp_function = make_function(function=protected_exp, name='exp', arity=1)


# Custom function to create Arrhenius-type expressions
def arrhenius_form(x1, x2):
    # The form is exp(-x1 / (R * x2)) where x1 is akin to activation energy and x2 is temperature
    return np.exp(-x1 / (R * np.clip(x2, 1e-10, None)))  # Clip to avoid division by zero
arrhenius_function = make_function(function=arrhenius_form, name='arrhenius', arity=2)


# Split the data into training and testing sets
X = data.drop(columns=['crystallization_onset_temp'])  # 'Tl' should be the name of your target variable column
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
    function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', exp_function, include_R, 'arrhenius'),
    const_range=(-5.0, 5.0),
    n_jobs=-1
)

param_space = {
    'population_size': (500, 2000),
    'generations': (100, 500),
    'p_crossover': (0.5, 0.9, 'uniform'),
    'p_subtree_mutation': (0.05, 0.2, 'uniform'),
    'p_hoist_mutation': (0.01, 0.1, 'uniform'),
    'p_point_mutation': (0.05, 0.2, 'uniform')
}

# Perform Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=sr_dc,
    search_spaces=param_space,
    n_iter=1000,  
    cv=5,  
    n_jobs=-1,
    verbose=2
)

# Fit the Bayesian optimization on training data
bayes_search.fit(X_train, y_train)

# Best estimator and prediction
best_sr = bayes_search.best_estimator_
y_pred_sr = best_sr_dc.predict(X_test)

# Calculate the R2 score
r2_sr = r2_score(y_test, y_pred_sr)
print(f"Best R2 Score (Typical SR after Bayesian Optimization): {r2_sr}")

# Custom Symbolic Regressor with Dimensional Check and Memory for Best Accuracy
class SymbolicRegressorWithDimensionalCheck(SymbolicRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_r2_score = -np.inf  # Initial best score set to negative infinity
        self.stop_storing = False     # Flag to stop storing inferior models
    
    def _validate(self, programs, X, y, sample_weight, random_state):
        valid_programs = []
        for program in programs:
            features = [str(symbol) for symbol in program.program if str(symbol) in units_dict]
            operations = [str(symbol) for symbol in program.program if str(symbol) in ['add', 'sub', 'mul', 'div']]
            if dimensional_consistency_validation(features, operations, units_dict):
                valid_programs.append(program)
        return super()._validate(valid_programs, X, y, sample_weight, random_state)

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self._remember_best_accuracy(X, y)

    def _remember_best_accuracy(self, X, y):
        if self.stop_storing:
            return  # Stop further calculations if storing has stopped

        y_pred = self.predict(X)
        current_r2_score = r2_score(y, y_pred)
        
        # Check if the new R2 score is an improvement
        if current_r2_score > self.best_r2_score:
            self.best_r2_score = current_r2_score  # Update the best score
            print(f"Improved R2 Score: {self.best_r2_score}")
        else:
            print(f"Current R2 Score ({current_r2_score}) is not better than Best R2 Score ({self.best_r2_score})")
            self.stop_storing = True  # Stop further storage if no improvement

# Train the Symbolic Regressor with Dimensions Embedding
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
    function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', exp_function, include_R, 'arrhenius'),
    const_range=(-5.0, 5.0),
    n_jobs=-1
)

# Perform Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=sr_dc,
    search_spaces=param_space,
    n_iter=32,  # Number of iterations
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,
    verbose=2
)

# Fit the Bayesian optimization on training data
bayes_search.fit(X_train, y_train)

# Best estimator and prediction
best_sr_dc = bayes_search.best_estimator_
y_pred_dc = best_sr_dc.predict(X_test)

# Calculate the R2 score
r2_dc = r2_score(y_test, y_pred_dc)
print(f"Best R2 Score (SR-DC after Bayesian Optimization): {r2_dc}")

# Get the best expression for Tx from both models
best_formula_sr = best_sr._program  # Typical SR model
best_formula_dc = best_sr_dc._program  # SR-DC model
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

# Creating a DataFrame to store the data
df_export = pd.DataFrame({
    'Actual_crystallization_onset_temp': y_test,
    'Predicted_crystallization_onset_temp_SR': y_pred_sr,
    'Predicted_crystallization_onset_temp_DC': y_pred_dc
})

# Exporting the DataFrame to an Excel file
file_path = 'predicted_vs_actual_crystallization_onset_temp.xlsx'
df_export.to_excel(file_path, index=False)

print(f'Data has been exported to {file_path}')

# Ensure only improved models are stored and plotted
generations = [10, 20, 30, 50, 70, 100, 120, 150, 180, 200]
r2_scores_sr = []
r2_scores_dc = []
complexity_sr = []
complexity_dc = []
best_r2_dc = -np.inf 

# Function to count the number of terms (operators and operands) in the formula
def count_terms(program):
    # Convert the program to string form
    formula_str = str(program)
    # Split by space to count individual elements (operators and operands)
    terms = formula_str.split()
    # Return the number of terms
    return len(terms)

# Loop over different generations and retrain the models
# Loop over different generations and retrain the models
for gen in generations:
    if sr_dc.stop_storing:  # Stop further training if storing has stopped
        break
        
    # Train Typical SR model with best hyperparmaters
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
    
    # Train SR-DC model with best hyperparmaters
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
    
  # Check if the R2 score improves
    current_r2_dc = r2_score(y_test, y_pred_dc_temp)
    if current_r2_dc > best_r2_dc:
        best_r2_dc = current_r2_dc  # Update best score
        
        # Store improved R2 score and complexity
        r2_scores_dc.append(current_r2_dc)
        complexity_dc.append(count_terms(sr_dc_temp._program))
    else:
        print(f"No improvement at generation {gen}. Stopping further complexity increase.")
        break  # Stop the loop if no improvement

# Ensure lengths are consistent
min_len = min(len(generations), len(r2_scores_sr), len(r2_scores_dc))

# Slice all lists to the minimum length
generations = generations[:min_len]
r2_scores_sr = r2_scores_sr[:min_len]
r2_scores_dc = r2_scores_dc[:min_len]

# Visualization 4: Plot R2 score vs. generations for both models
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

# Visualization 5: Plot R2 score vs. model complexity for both models
plt.figure(figsize=(10, 5))
plt.plot(complexity_sr, r2_scores_sr, label='Typical SR', marker='o', color='blue')
plt.plot(complexity_dc, r2_scores_dc, label='SR-DC', marker='o', color='red')
plt.xlabel('Model Complexity (Number of Terms)')
plt.ylabel('R2 Score')
plt.title('R2 Score vs. Model Complexity')
plt.legend()
plt.show()
