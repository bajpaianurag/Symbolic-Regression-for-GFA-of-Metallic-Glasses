import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import collections
import sympy as sp
from sympy.physics.units import (
    joule, kelvin, mole, second, meter, kilogram
)
from sympy.physics.units import Unit, Dimension
from sympy.physics.units.util import check_dimensions
try:
    from sympy.physics.units.dimensions import Dimension, dimensionless
except ImportError:
    dimensionless = Dimension(1)

import warnings
warnings.filterwarnings('ignore')

## Load the Dataset
data = pd.read_csv('symbolic_data_dimensional_consistency_calculations.csv')
print(data.head())

## Define Units for Features and Targets
# Define sympy symbols for variables
variables_symbols = {
    'Heating_rate': sp.Symbol('Heating_rate'),
    'Hmix': sp.Symbol('Hmix'),
    'Smix': sp.Symbol('Smix'),
    'delta_r': sp.Symbol('delta_r'),
    'delta_elec': sp.Symbol('delta_elec'),
    'CN_avg': sp.Symbol('CN_avg'),
    'ebya_avg': sp.Symbol('ebya_avg'),
    'EA_avg': sp.Symbol('EA_avg'),
    'Tm_avg': sp.Symbol('Tm_avg'),
    'eta': sp.Symbol('eta'),
    'BP': sp.Symbol('BP'),
    'R': sp.Symbol('R'),
}

# Assign units to variables
variables_with_units = {
    'Heating_rate': variables_symbols['Heating_rate'] * kelvin / second,
    'Hmix': variables_symbols['Hmix'] * joule / mole,
    'Smix': variables_symbols['Smix'] * joule / (mole * kelvin),
    'delta_r': variables_symbols['delta_r'] * dimensionless,
    'delta_elec': variables_symbols['delta_elec'] * dimensionless,
    'CN_avg': variables_symbols['CN_avg'] * dimensionless,
    'ebya_avg': variables_symbols['ebya_avg'] * dimensionless,
    'EA_avg': variables_symbols['EA_avg'] * joule / mole,
    'Tm_avg': variables_symbols['Tm_avg'] * kelvin,
    'eta': variables_symbols['eta'] * dimensionless,
    'BP': variables_symbols['BP'] / meter,
    'R': variables_symbols['R'] * joule / (mole * kelvin),
}

# Define units for target variables
unit_mapping = {
    'Heating_rate': kelvin / second,
    'Hmix': joule / mole,
    'Smix': joule / (mole * kelvin),
    'delta_r': dimensionless,
    'delta_elec': dimensionless,
    'CN_avg': dimensionless,
    'ebya_avg': dimensionless,
    'EA_avg': joule / mole,
    'Tm_avg': kelvin,
    'eta': dimensionless,
    'BP': 1 / meter,
    'Tg': kelvin,
    'Tx': kelvin,
    'Tl': kelvin,
    'R': joule / (mole * kelvin)
}

## Attach Units to Data
data_with_units = data.copy()

for col in data.columns:
    if col in unit_mapping:
        unit = unit_mapping[col]
        data_with_units[col] = data[col].apply(lambda x: x * unit)


## Define Symbolic Variables with Units
variables = {}
for col in data.columns:
    if col in unit_mapping:
        symbol = sp.Symbol(col)
        variables[col] = symbol * unit_mapping[col]

## Extract Numerical Values for Regression
data_numeric = data.copy()

for col in data.columns:
    if col in unit_mapping:
        data_numeric[col] = data[col]

## Include the Universal Gas Constant R
R_value = 8.314462618  # J/(mol·K)
variables['R'] = sp.Symbol('R') * unit_mapping['R']
data_numeric['R'] = R_value

## Prepare Features and Targets
features = ['Heating_rate', 'Hmix', 'Smix', 'delta_r', 'delta_elec', 'CN_avg',
            'ebya_avg', 'EA_avg', 'Tm_avg', 'eta', 'BP', 'R']
targets = ['Tg', 'Tx', 'Tl']

X = data_numeric[features]
y_dict = {target: data_numeric[target] for target in targets}

train_test_data = {}

for target in targets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_dict[target], test_size=0.2, random_state=42)
    train_test_data[target] = (X_train, X_test, y_train, y_test)

## Define the Constraint Function for Dimensional Consistency
def equation_constraint(sympy_expr):
    try:
        expr_with_units = sympy_expr.subs(variables_with_units)

        def assign_units_to_constants(expr):
            if expr.is_Number:
                return expr
            elif expr.func == sp.Add:
                units_list = []
                for arg in expr.args:
                    if not arg.is_Number:
                        arg_units = units.dimension_system.get_dimensional_dependencies(arg)
                        if arg_units:
                            units_list.append(arg_units)
                if units_list:
                    units_set = set(frozenset(u.items()) for u in units_list)
                    if len(units_set) == 1:
                        common_units = units_list[0]
                        unit_expr = units.Quantity('unit_expr', dimension_dict=common_units)
                        new_args = []
                        for arg in expr.args:
                            if arg.is_Number:
                                new_args.append(arg * unit_expr)
                            else:
                                new_args.append(assign_units_to_constants(arg))
                        return expr.func(*new_args)
                    else:
                        raise ValueError('Inconsistent units in addition or subtraction')
                else:
                    new_args = [assign_units_to_constants(arg) for arg in expr.args]
                    return expr.func(*new_args)
            else:
                new_args = [assign_units_to_constants(arg) for arg in expr.args]
                return expr.func(*new_args)

        expr_with_units = assign_units_to_constants(expr_with_units)
        expr_simplified = sp.simplify(expr_with_units)
        expr_dimension = units.dimension_system.get_dimensional_dependencies(expr_simplified)

        target_unit = unit_mapping[equation_constraint.target_variable]
        target_dimension = units.dimension_system.get_dimensional_dependencies(target_unit)

        if expr_dimension != target_dimension:
            return False

        def check_dimensions_in_pow(expr):
            if isinstance(expr, sp.Pow):
                base, exponent = expr.args
                exponent_dimension = units.dimension_system.get_dimensional_dependencies(exponent)
                if exponent_dimension:
                    return False
                return check_dimensions_in_pow(base)
            elif expr.is_Atom:
                return True
            else:
                return all(check_dimensions_in_pow(arg) for arg in expr.args)

        if not check_dimensions_in_pow(expr_simplified):
            return False

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

## Define the Symbolic Regressor
def create_model(params, target_variable):
    equation_constraint.target_variable = target_variable

    model = PySRRegressor(
        niterations=params.get('niterations', 500),
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["sqrt", "exp", "log", "abs"],
        extra_sympy_mappings={"pow": sp.Pow},
        populations=params.get('populations', 50),
        population_size=params.get('population_size', 1000),
        maxsize=params.get('maxsize', 20),
        loss=params.get('loss', 'loss(x, y) = (x - y)^2'),
        verbosity=1,
        random_state=42,
        progress=True,
        procs=-1,
        constraints={
            'constraint_func': equation_constraint
        }
    )
    return model


## Perform Bayesian Hyperparameter Optimization
patience = 50 
min_delta = 1e-4

best_params = {}
for target in targets:
    print(f"\nOptimizing hyperparameters for {target}...")
    X_train, X_test, y_train, y_test = train_test_data[target]

    def objective(trial):
        params = {
            'niterations': trial.suggest_int('niterations', 100, 500, step=50),
            'populations': trial.suggest_int('populations', 10, 50, step=5),
            'population_size': trial.suggest_int('population_size', 100, 1000, step=100),
            'maxsize': trial.suggest_int('maxsize', 10, 25, step=5),
        }
        model = create_model(params, target)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            best_eqn = model.equations_[-1]["equation"]
            if not check_dimensional_consistency(best_eqn, target):
                penalty = 1e5  
                mse += penalty

            return mse
        except Exception as e:
            return np.inf

    study = optuna.create_study(direction='minimize')

    best_score = np.inf
    no_improvement_count = 0

    for iteration in range(50):
        study.optimize(objective, n_trials=1)
        current_score = study.best_value

        if current_score < best_score - min_delta:
            best_score = current_score
            no_improvement_count = 0 
            print(f"Iteration {iteration + 1}: Improvement found! Best Score: {best_score}")
        else:
            no_improvement_count += 1
            print(f"Iteration {iteration + 1}: No significant improvement. Count: {no_improvement_count}")
        
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {iteration + 1} iterations.")
            break

    best_params[target] = study.best_params
    print(f"Best parameters for {target}: {best_params[target]}")


## Perform Multi-objective Optimization
def objective_multi(trial, X_train, y_train, X_test, y_test, target_variable):
    params = {
        'niterations': trial.suggest_int('niterations', 50, 300, step=50),
        'populations': trial.suggest_int('populations', 10, 50, step=10),
        'population_size': trial.suggest_int('population_size', 50, 500, step=50),
        'maxsize': trial.suggest_int('maxsize', 10, 25, step=5),
    }
    model = create_model(params, target_variable)
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        complexity = model.get_best()['complexity']
    except Exception:
        mae = np.inf
        complexity = np.inf
    return mae, complexity

patience = 50
min_delta_mae = 0.001  
min_delta_complexity = 2

best_params = {}

for target in targets:
    print(f"\nPerforming multi-objective optimization for {target}...")
    target_variable = target
    X_train, X_test, y_train, y_test = train_test_data[target]

    study = optuna.create_study(directions=['minimize', 'minimize'])
    func = lambda trial: objective_multi(trial, X_train, y_train, X_test, y_test, target_variable)
    
    best_mae = np.inf
    best_complexity = np.inf
    no_improvement_count = 0

    for i in range(50): 
        study.optimize(func, n_trials=1)
        best_trial = study.best_trials[0]
        current_mae = best_trial.values[0]
        current_complexity = best_trial.values[1]

        mae_improvement = best_mae - current_mae > min_delta_mae
        complexity_improvement = best_complexity - current_complexity > min_delta_complexity

        if mae_improvement or complexity_improvement:
            best_mae = min(best_mae, current_mae)
            best_complexity = min(best_complexity, current_complexity)
            no_improvement_count = 0
            print(f"Improvement detected! New Best MAE: {best_mae}, New Best Complexity: {best_complexity}")
        else:
            no_improvement_count += 1
            print(f"No significant improvement. No improvement count: {no_improvement_count}")

        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {i + 1} trials.")
            break

    best_params[target] = best_trial.params
    print(f"Best parameters for {target} after multi-objective optimization: {best_params[target]}")


## Train Final Models
models = {}
results = {}

for target in targets:
    print(f"\nTraining final model for {target}...")
    target_variable = target
    X_train, X_test, y_train, y_test = train_test_data[target]
    params = best_params[target]
    model = create_model(params, target_variable)
    model.fit(X_train, y_train)
    models[target] = model

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print(f"Performance for {target}: MSE = {mse:.4f}, R2 = {r2:.4f}")

    train_results = pd.DataFrame({
        'Actual': y_train.reset_index(drop=True),
        'Predicted': y_train_pred,
        'Dataset': 'Train'
    })

    test_results = pd.DataFrame({
        'Actual': y_test.reset_index(drop=True),
        'Predicted': y_test_pred,
        'Dataset': 'Test'
    })

    combined_results = pd.concat([train_results, test_results], ignore_index=True)
    results[target] = combined_results


## Generate Plots
for target in targets:
    model = models[target]
    X_train, X_test, y_train, y_test = train_test_data[target]
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    plt.figure(figsize=(16, 10))
    
    train_scatter = plt.scatter(y_train, y_pred_train, label=f'Train (R²={r2_train:.2f}, MSE={mse_train:.2f})', 
                                c=y_pred_train, cmap='Blues', s=150, alpha=0.6, edgecolor='k')

    test_scatter = plt.scatter(y_test, y_pred_test, label=f'Test (R²={r2_test:.2f}, MSE={mse_test:.2f})', 
                               c=y_pred_test, cmap='Reds', s=150, alpha=0.6, edgecolor='k')

    plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], color='black', linestyle='--', lw=2, label='Perfect Prediction')
    
    cbar_train = plt.colorbar(train_scatter, ax=plt.gca(), label='Train Predictions')
    cbar_test = plt.colorbar(test_scatter, ax=plt.gca(), label='Test Predictions')

    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'Predicted vs Actual for {target}', fontsize=16, weight='bold')
    plt.grid(False)
    
    min_value = min(y_train.min(), y_test.min())
    max_value = max(y_train.max(), y_test.max())
    plt.xlim([min_value, max_value])
    plt.ylim([min_value, max_value])

    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"parity_plot_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    X_train, X_test, y_train, y_test = train_test_data[target]

    y_pred_test = model.predict(X_test)
    residuals = y_test - y_pred_test
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals, bins=20, kde=True, color='dodgerblue', alpha=0.6, edgecolor='k')

    plt.axvline(residual_mean, color='red', linestyle='--', lw=2, label=f'Mean: {residual_mean:.2f}')
    plt.axvline(residual_mean + residual_std, color='green', linestyle='--', lw=2, label=f'+1 Std Dev: {residual_mean + residual_std:.2f}')
    plt.axvline(residual_mean - residual_std, color='green', linestyle='--', lw=2, label=f'-1 Std Dev: {residual_mean - residual_std:.2f}')

    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], 0, plt.gca().get_xlim()[1], color='green', alpha=0.1, label='Positive Residuals')
    plt.fill_betweenx([0, plt.gca().get_ylim()[1]], plt.gca().get_xlim()[0], 0, color='red', alpha=0.1, label='Negative Residuals')

    plt.xlabel('Residuals (Actual - Predicted)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Residuals Distribution for {target}', fontsize=16, weight='bold')
    plt.grid(False, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"residuls_plot_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    history = model.equations_
    
    feature_count = collections.defaultdict(int)

    for eqn in history['equation']:
        for feature in features:
            if feature in str(eqn):
                feature_count[feature] += 1

    features_sorted = sorted(feature_count.keys(), key=lambda x: feature_count[x], reverse=True)
    counts_sorted = [feature_count[feature] for feature in features_sorted]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=features_sorted, y=counts_sorted, palette='viridis')
    
    for i, count in enumerate(counts_sorted):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=12)

    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Count of Appearances', fontsize=14)
    plt.title(f'Feature Importance for {target}', fontsize=16, weight='bold')
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"feature importance_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    history = model.equations_
    generations = history.index
    losses = history['loss']
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("coolwarm", as_cmap=True)
    
    plt.plot(generations, losses, marker='o', markersize=12, linewidth=3, linestyle='-', color='blue', label='Loss per Generation')

    min_loss_idx = np.argmin(losses)
    plt.scatter(generations[min_loss_idx], losses[min_loss_idx], color='red', s=180, zorder=5, label=f'Min Loss: {losses[min_loss_idx]:.4f}')

    plt.xlabel('Generation (Iteration Index)', fontsize=20)
    plt.ylabel('Loss (Error)', fontsize=20)

    plt.title(f'Loss Reduction Over Generations for {target}', fontsize=16, weight='bold')
    plt.grid(False)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"loss_numberofgenerations_plot_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    equations = model.equations_
    complexities = equations['complexity']
    losses = equations['loss']
    
    plt.figure(figsize=(10, 8))
    plt.plot(complexities, losses, marker='o', markersize=8, linewidth=2, linestyle='-', color='darkorange', label='Loss vs Complexity')

    min_loss_idx = np.argmin(losses)
    plt.scatter(complexities[min_loss_idx], losses[min_loss_idx], color='red', s=100, zorder=5, label=f'Min Loss: {losses[min_loss_idx]:.4f}')

    plt.text(complexities[min_loss_idx], losses[min_loss_idx] + 0.05, f"Min Loss\n{losses[min_loss_idx]:.4f}", 
             fontsize=12, color='red', ha='center', va='bottom')

    plt.xlabel('Model Complexity (Number of Terms)', fontsize=20)
    plt.ylabel('Loss (Error)', fontsize=20)
    plt.title(f'Loss vs. Model Complexity for {target}', fontsize=16, weight='bold')
    plt.grid(False)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"pareto_plot_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    history = model.equations_

    complexities = history['complexity']
    num_features = [len(set(feature for feature in features if feature in str(eqn))) for eqn in history['equation']]

    avg_features = np.mean(num_features)
    avg_complexity = np.mean(complexities)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=num_features, y=complexities, size=complexities, hue=complexities, palette='coolwarm', sizes=(50, 200), legend=False)
    sns.regplot(x=num_features, y=complexities, scatter=False, color='black', line_kws={'linewidth': 2, 'linestyle': '--'})
    max_complexity_idx = np.argmax(complexities)
    plt.scatter(num_features[max_complexity_idx], complexities[max_complexity_idx], color='red', s=200, zorder=5, label=f'Highest Complexity: {complexities[max_complexity_idx]}')

    plt.axvline(avg_features, color='green', linestyle='--', lw=2, label=f'Avg Features: {avg_features:.2f}')
    plt.axhline(avg_complexity, color='blue', linestyle='--', lw=2, label=f'Avg Complexity: {avg_complexity:.2f}')
    plt.xlabel('Number of Features in Equation', fontsize=14)
    plt.ylabel('Complexity (Number of Terms)', fontsize=14)
    plt.title(f'Complexity vs Number of Features for {target}', fontsize=16, weight='bold')
    plt.grid(False)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Complexity_NumberofFeatures_plot_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    history = model.equations_
    complexities = history['complexity']
    scores = history['score']
    avg_score = np.mean(scores)
    avg_complexity = np.mean(complexities)
    
    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=complexities, y=scores, hue=scores, palette='coolwarm', size=scores, sizes=(100, 300), legend=False)
    sns.regplot(x=complexities, y=scores, scatter=False, color='black', line_kws={'linewidth': 2, 'linestyle': '--'})
    best_score_idx = np.argmin(scores)
    plt.scatter(complexities[best_score_idx], scores[best_score_idx], color='red', s=200, zorder=5, label=f'Best Score: {scores[best_score_idx]:.4f}')

    plt.text(complexities[best_score_idx], scores[best_score_idx] + 0.05, f"Best Score\n{scores[best_score_idx]:.4f}", 
             fontsize=12, color='red', ha='center', va='bottom')
    plt.axvline(avg_complexity, color='green', linestyle='--', lw=2, label=f'Avg Complexity: {avg_complexity:.2f}')
    plt.axhline(avg_score, color='blue', linestyle='--', lw=2, label=f'Avg Score: {avg_score:.2f}')
    plt.xlabel('Model Complexity (Number of Terms)', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(f'Score vs. Complexity for {target}', fontsize=16, weight='bold')
    plt.grid(False)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Score_vs_Complexity_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    history = model.equations_
    
    complexities = history['complexity']
    
    mean_complexity = np.mean(complexities)
    std_complexity = np.std(complexities)
    
    plt.figure(figsize=(10, 8))
    
    sns.histplot(complexities, bins=10, color='mediumpurple', kde=False, alpha=0.7)
    plt.axvline(mean_complexity, color='red', linestyle='--', lw=2, label=f'Mean: {mean_complexity:.2f}')
    plt.axvline(mean_complexity + std_complexity, color='green', linestyle='--', lw=2, label=f'+1 Std Dev: {mean_complexity + std_complexity:.2f}')
    plt.axvline(mean_complexity - std_complexity, color='green', linestyle='--', lw=2, label=f'-1 Std Dev: {mean_complexity - std_complexity:.2f}')
    plt.text(mean_complexity, plt.gca().get_ylim()[1]*0.8, f"Mean: {mean_complexity:.2f}", color='red', fontsize=12, ha='center')
    plt.xlabel('Model Complexity (Number of Terms)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Complexity Distribution Over Time for {target}', fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Complexity_Distribution_Over_Time_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    model = models[target]
    history = model.equations_
    
    equation_lengths = [len(str(eqn).split()) for eqn in history['equation']]
    complexities = history['complexity']
    
    max_length = max(equation_lengths)
    max_complexity = max(complexities)
    max_length_idx = np.argmax(equation_lengths)
    max_complexity_idx = np.argmax(complexities)
    
    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=equation_lengths, y=complexities, hue=complexities, palette='coolwarm', size=complexities, sizes=(100, 300), legend=False)
    sns.regplot(x=equation_lengths, y=complexities, scatter=False, color='black', line_kws={'linewidth': 2, 'linestyle': '--'})
    plt.xlabel('Equation Length (Number of Terms)', fontsize=14)
    plt.ylabel('Complexity (Number of Terms)', fontsize=14)
    plt.title(f'Equation Length vs. Complexity for {target}', fontsize=16, weight='bold')
    plt.grid(False)
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"EquationLength_vs_Complexity_DESyR_{target}.jpg", dpi=600, format='jpg')
    plt.show()


for target in targets:
    print(f"\nBest equations for {target}:")
    print(models[target].sympy())
