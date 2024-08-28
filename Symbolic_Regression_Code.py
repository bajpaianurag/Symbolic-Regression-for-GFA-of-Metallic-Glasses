#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gplearn')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Universal constants
R = 8.314  # Universal gas constant in J/(mol K)


# In[4]:


# Load data from CSV
file_path = 'symbolic_Tg.csv'
data = pd.read_csv(file_path)


# In[5]:


# Units are expressed in the form of [length, mass, time, current, amount of substance, luminous intensity, temperature]
units_dict = {
    'Hmix': np.array([2, 1, -2, 0, -1, 0, 0]), 
    'delta_H': np.array([4, 2, -4, 0, -2, 0, 0]), 
    'Smix': np.array([2, 1, -2, 0, -1, 0, 0]),  
    'delta': np.array([0, 0, 0, 0, 0, 0, 0]), 
    'delta_weight': np.array([0, 2, 0, 0, 0, 0, 0]),  
    'CN_avg': np.array([0, 0, 0, 0, 0, 0, 0]), 
    'ebya_avg': np.array([0, 0, 0, 0, 0, 0, 0]),  
    'Elec': np.array([0, 0, 0, 0, 0, 0, 0]), 
    'EA_avg': np.array([2, 1, -2, 0, -1, 0, 0]), 
    'Tg': np.array([0, 0, 0, 0, 0, 0, 1])       
}


# In[6]:


# Check dimensional consistency of a formula
def is_dimensionally_consistent(formula_units, target_units):
    return np.array_equal(formula_units, target_units)


# In[7]:


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
    return is_dimensionally_consistent(result_units, units_dict['Tg'])


# In[8]:


# Define custom function to include R explicitly
def include_R_in_expressions(x):
    return R * x


# In[9]:


# Add the function to the symbolic regression function set
include_R = make_function(function=include_R_in_expressions, name="include_R", arity=1)


# In[10]:


# Define custom division function with protection against division by zero
def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.0)

protected_div = make_function(function=protected_div, name="protected_div", arity=2)


# In[11]:


# Define custom exponential function
def exp_func(x):
    return np.exp(np.clip(x, -100, 100))  # Clipping to avoid overflow

exp_function = make_function(function=exp_func, name="exp", arity=1)


# In[12]:


# Split the data into training and testing sets
X = data.drop(columns=['Tg'])  # 'Tg' should be the name of your target variable column
y = data['Tg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


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
    parsimony_coefficient=0.01,
    random_state=42,
    function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', 'sin', 'cos', exp_function, include_R),
    const_range=(-5.0, 5.0),
    n_jobs=-1
)


# In[14]:


# GridSearchCV for hyperparameter optimization for typical SR
param_grid = {
    'population_size': [500, 1000, 2000],
    'generations': [100, 200, 300],
    'p_crossover': [0.6, 0.7, 0.8],
    'p_subtree_mutation': [0.05, 0.1, 0.2],
    'p_hoist_mutation': [0.05, 0.1, 0.2],
    'p_point_mutation': [0.05, 0.1, 0.2]
}


# In[ ]:


grid_search_sr = GridSearchCV(estimator=sr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_sr.fit(X_train, y_train)


# In[ ]:


# Best estimator after GridSearchCV for typical SR
best_sr = grid_search_sr.best_estimator_


# In[ ]:


# Predict and calculate R2 score for typical SR model
y_pred_sr = best_sr.predict(X_test)
print(f'R2 Score (Typical SR): {r2_score(y_test, y_pred_sr)}')


# In[ ]:


from gplearn.functions import make_function
import numpy as np

# Define a safe exponential function that handles large values
def protected_exp(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.0)

# Create the custom function
exp_function = make_function(function=protected_exp, name='exp', arity=1)


# In[ ]:


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


# In[ ]:


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
    parsimony_coefficient=0.01,
    random_state=42,
    function_set=('add', 'sub', 'mul', protected_div, 'sqrt', 'log', 'sin', 'cos', exp_function, include_R),
    const_range=(-5.0, 5.0),
    n_jobs=-1
)


# In[ ]:


grid_search_dc = GridSearchCV(estimator=sr_dc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search_dc.fit(X_train, y_train)


# In[ ]:


# Best estimator after GridSearchCV with dimensional consistency (SR-DC)
best_sr_dc = grid_search_dc.best_estimator_


# In[ ]:


# Predict and calculate R2 score for SR-DC model
y_pred_dc = best_sr_dc.predict(X_test)
print(f'R2 Score (SR-DC): {r2_score(y_test, y_pred_dc)}')


# In[ ]:


# Get the best expression for Tg from both models
best_formula_sr = best_sr._program  # Typical SR model
best_formula_dc = best_sr_dc._program  # SR-DC model
print(f'Best Formula for Tg (Typical SR): {best_formula_sr}')
print(f'Best Formula for Tg (SR-DC): {best_formula_dc}')


# In[ ]:


# Visualization 1: Predicted vs Actual Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_sr, color='blue', label='Typical SR')
plt.scatter(y_test, y_pred_dc, color='red', label='SR-DC')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Actual Tg')
plt.ylabel('Predicted Tg')
plt.title('Predicted vs Actual Tg')
plt.legend()
plt.show()


# In[ ]:


# Visualization 2: Residuals Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_sr, y_test - y_pred_sr, color='blue', label='Typical SR')
plt.scatter(y_pred_dc, y_test - y_pred_dc, color='red', label='SR-DC')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Predicted Tg')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.legend()
plt.show()


# In[ ]:


# Visualization 3: Distribution of Errors
plt.figure(figsize=(10, 5))
plt.hist(y_test - y_pred_sr, bins=20, color='blue', alpha=0.5, label='Typical SR')
plt.hist(y_test - y_pred_dc, bins=20, color='red', alpha=0.5, label='SR-DC')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.legend()
plt.show()


# In[ ]:


# Visualization 4: Feature Importance (using symbolic regression counts or any importance metric you have)
# This requires counting how often each feature appears in the final expression or an alternative method
feature_importances_sr = {
    'Hmix': best_sr._program.count('Hmix'),
    'delta_H': best_sr._program.count('delta_H'),
    'Smix': best_sr._program.count('Smix'),
    'delta': best_sr._program.count('delta'),
    'delta_weight': best_sr._program.count('delta_weight'),
    'CN_avg': best_sr._program.count('CN_avg'),
    'ebya_avg': best_sr._program.count('ebya_avg'),
    'Elec': best_sr._program.count('Elec'),
    'EA_avg': best_sr._program.count('EA_avg')
}

plt.figure(figsize=(10, 5))
plt.bar(feature_importances_sr.keys(), feature_importances_sr.values(), color='blue')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance (Typical SR)')
plt.show()


# In[ ]:


# Visualization 5: Pairwise Plot of Features and Predictions
data['Predicted_Tg_SR'] = y_pred_sr
data['Predicted_Tg_DC'] = y_pred_dc

sns.pairplot(data, vars=['Hmix', 'delta_H', 'Smix', 'Predicted_Tg_SR'], kind='reg')
plt.suptitle('Pairwise Plot for Typical SR', y=1.02)
plt.show()

sns.pairplot(data, vars=['Hmix', 'delta_H', 'Smix', 'Predicted_Tg_DC'], kind='reg')
plt.suptitle('Pairwise Plot for SR-DC', y=1.02)
plt.show()


# In[ ]:


# Visualization 6: Cross-Validation Scores
cv_results_sr = grid_search_sr.cv_results_['mean_test_score']
cv_results_dc = grid_search_dc.cv_results_['mean_test_score']

plt.figure(figsize=(10, 5))
plt.boxplot([cv_results_sr, cv_results_dc], labels=['Typical SR', 'SR-DC'])
plt.ylabel('R² Score')
plt.title('Cross-Validation Scores')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




