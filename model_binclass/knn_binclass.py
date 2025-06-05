import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import utils.model_utils as mu
import utils.shap_py as sp
import joblib
from typing import Dict

# === 1. LEARN ===
def learn_model(df, target_col, params=None, search=True, cv=5, scoring='recall', random_state=42):
    """Train K-Nearest Neighbors with optional hyperparameter tuning"""
    model = KNeighborsClassifier()

    X = df.drop(columns=[target_col, 'dataset'])
    y = df[target_col]

    if params:
        if search:
            best_params, best_model = mu.grid_search(model, X, y, params, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            model.set_params(**params)
            best_model = model.fit(X, y)
            best_params = params
    else:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # p=1: Manhattan, p=2: Euclidean
        }
        best_params, best_model = mu.grid_search(model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        'model': best_model,
        'model_params': best_params,
    }

# === 2. APPLY ===
def apply_model(df, target_col, model_info: Dict) -> pd.DataFrame:
    """Apply KNN model to data and return predictions + probabilities (if available)"""
    model = model_info['model']
    X = df.drop(columns=[target_col, 'dataset'])
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = df.copy()
    results['predictions'] = predictions

    # Only include probabilities if model supports it
    if hasattr(model, "predict_proba"):
        results['probabilities'] = probabilities

    return results

# === 3. EVALUATE ===
def evaluate_model(df: pd.DataFrame, target_col: str):
    """Evaluate predictions with classification metrics and probability-based plots"""
    y_true = df[target_col]
    y_pred = df['predictions']
    pred_proba = df['probabilities'] if 'probabilities' in df else None

    print("═══ Classification Report ═══")
    print(classification_report(y_true, y_pred))

    mu.plot_confusion_matrix(y_true, y_pred)

    if pred_proba is not None:
        mu.plot_probability_metrics(y_true, pred_proba)
        mu.plot_lift_curve(y_true, pred_proba)
        mu.plot_cumulative_gains(y_true, pred_proba)
    else:
        print("Note: Probability-based plots skipped (predict_proba not available).")

# === 4. EXPLAIN ===
def explain_model(model, X_train, X_test, top_n_features=10, sample_index=None, index_feature=False, save_path=None):
    """
    Generate SHAP-based model explanations (uses kernel SHAP for KNN)
    """
    shap_vals = sp.shap_values(model, X_train, X_test, model_type='kernel')  # SHAP için 'kernel' uygun

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
