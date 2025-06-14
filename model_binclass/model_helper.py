import utils.shap_py as sp
import joblib
from typing import Dict

# === 4. EXPLAIN ===
def explain_model(model, df, top_n_features=10, sample_index=None, index_feature=False, save_path=None):
    df = df[df.dataset == 0].drop(columns=["dataset"]) 
    #Comments take place here
    shap_vals = sp.shap_values(model, df)

    sp.global_analysis(shap_vals, df, top_n_features=top_n_features, save_path=save_path)

    if sample_index is not None:
        sp.index_charts(shap_vals, sample_index=sample_index, top_n_features=top_n_features, save_path=save_path)

    if index_feature:
        sp.index_feature(shap_vals, df, save_path=save_path)

    return shap_vals

# === 5. SAVE ===
def save_model(model_info: Dict, filepath: str):
    """Save trained XGBoost model and its metadata"""
    joblib.dump(model_info, filepath)
    print(f"Model saved to {filepath}")

# === 6. LOAD ===
def load_model(filepath: str) -> Dict:
    """Load trained XGBoost model and its metadata"""
    model_info = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_info
