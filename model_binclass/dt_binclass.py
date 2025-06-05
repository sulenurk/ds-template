import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import utils.model_utils as mu
import utils.shap_py as sp
import joblib
from typing import Dict, List, Optional

# === 1. LEARN ===
def learn_model(df, target_col, params=None, search=True, cv=5, scoring='recall', random_state=42):
    """Train Decision Tree with optional hyperparameter tuning"""
 
    model = DecisionTreeClassifier(random_state=random_state)

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
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        best_params, best_model = mu.grid_search(model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        'model': best_model,
        'model_params': best_params,
    }

# === 2. APPLY ===
def apply_model(df, model_info: Dict, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply Decision Tree model to data and return predictions + probabilities"""
    model = model_info['model']

    # Drop specified columns temporarily
    if columns:
        dropped_df = df[columns].copy()
        df_model_input = df.drop(columns=columns)
    else:
        dropped_df = None
        df_model_input = df

    if "predictions" in df.columns:
        df_model_input.drop("predictions", axis=1, inplace=True)

    if "probabilities" in df.columns:
        df_model_input.drop("probabilities", axis=1, inplace=True)
    
    predictions = model.predict(df_model_input)
    probabilities = model.predict_proba(df_model_input)[:, 1]

    results = df_model_input.copy()
    results['predictions'] = predictions
    results['probabilities'] = probabilities

    # Reinsert dropped columns
    if dropped_df is not None:
        results = pd.concat([results, dropped_df], axis=1)

    return results

# === 3. EVALUATE ===
def evaluate_model(df: pd.DataFrame, target_col: str):
    """Evaluate classification predictions with report and confusion matrix from a single DataFrame."""

    y_true = df[target_col]
    y_pred = df['predictions']
    
    print("═══ Classification Report ═══")
    print(classification_report(y_true, y_pred))
    
    mu.plot_confusion_matrix(y_true, y_pred)

# === 4. EXPLAIN ===
def explain_model(model_info, df, top_n_features=10, sample_index=None, index_feature=False, save_path=None):
    """Use SHAP to explain a decision tree model"""

    X_train = df[df.dataset == 1].drop(columns=["dataset"])
    X_test = df[df.dataset == 0].drop(columns=["dataset"])

    shap_vals = sp.shap_values(model_info, X_train, X_test, model_type='tree')

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
