import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import utils.model_utils as mu
import utils.shap_py as sp
import joblib
from typing import Dict

# === 1. LEARN ===
def learn_model(X, y, params=None, search=True, cv=5, scoring='neg_root_mean_squared_error', random_state=42):
    """Train Lasso Regression with optional hyperparameter tuning"""
    model = Lasso(random_state=random_state, max_iter=10000)

    if params:
        if search:
            best_params, best_model = mu.grid_search(model, X, y, params, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            model.set_params(**params)
            best_model = model.fit(X, y)
            best_params = params
    else:
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']
        }
        best_params, best_model = mu.grid_search(model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        'model': best_model,
        'model_params': best_params
    }

# === 2. APPLY ===
def apply_model(X, model_info: Dict) -> pd.DataFrame:
    """Apply lasso regression model and return predictions"""
    model = model_info['model']
    predictions = model.predict(X)

    results = pd.DataFrame({
        'predictions': predictions
    }, index=pd.DataFrame(X).index)

    return results

# === 3. EVALUATE ===
def evaluate_model(y_pred_df: pd.DataFrame, y_true: pd.Series):
    """Evaluate regression predictions with standard metrics"""
    y_pred = y_pred_df['predictions']

    mu.regression_evaluation(y_true, y_pred)

# === 4. EXPLAIN ===
def explain_model(model, X_train, X_test, top_n_features=10, sample_index=None, index_feature=False, save_path=None):
    """Use SHAP to explain a lasso regression model"""
    shap_vals = sp.shap_values(model, X_train, X_test, model_type='linear')

    sp.global_analysis(shap_vals, X_test, top_n_features=top_n_features, save_path=save_path)

    if sample_index is not None:
        sp.index_charts(shap_vals, sample_index=sample_index, top_n_features=top_n_features, save_path=save_path)

    if index_feature:
        sp.index_feature(shap_vals, X_test, save_path=save_path)

# === 5. SAVE ===
def save_model(model_info: Dict, filepath: str):
    """Save trained model and metadata"""
    joblib.dump(model_info, filepath)
    print(f"Model saved to {filepath}")

# === 6. LOAD ===
def load_model(filepath: str) -> Dict:
    """Load model and metadata from file"""
    model_info = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_info
