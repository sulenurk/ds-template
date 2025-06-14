import utils.shap_py as sp
import joblib
from typing import Dict

# === 4. EXPLAIN ===
def explain_model(model, df, top_n_features=10, sample_index=None, index_feature=False, save_path=None):
    """
    Generate SHAP-based model explanations including global and instance-level visualizations.

    Filters the input DataFrame to `dataset == 0` (inference set),
    computes SHAP values, and creates:
    - Global feature importance plots
    - Per-sample SHAP explanation (if `sample_index` is set)
    - SHAP feature plots for each instance (if `index_feature=True`)

    Parameters
    ----------
    model : fitted model object
        Trained model compatible with SHAP explainers.
    df : pd.DataFrame
        Input dataset containing a 'dataset' column. Only rows where
        `dataset == 0` are used.
    top_n_features : int, default=10
        Number of top features to display in plots.
    sample_index : int, optional
        Index of a specific sample for which SHAP explanations will be drawn.
    index_feature : bool, default=False
        If True, creates SHAP feature-wise plots for each observation.
    save_path : str, optional
        Directory path to save generated plots as images.

    Returns
    -------
    np.ndarray or list of np.ndarray
        SHAP values computed for the filtered DataFrame.

    Notes
    -----
    - Requires `sp` module to define `shap_values`, `global_analysis`,
      `index_charts`, and `index_feature` functions.
    - Assumes binary or multiclass classification.
    """
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
    """
    Save a trained model and its metadata to disk using joblib.

    Parameters
    ----------
    model_info : dict
        Dictionary containing the trained model and related metadata such
        as best hyperparameters, feature names, etc.
    filepath : str
        Full file path (including `.pkl` or `.joblib`) where the model will be saved.

    Returns
    -------
    None
        Prints confirmation to stdout upon successful save.
    """
    joblib.dump(model_info, filepath)
    print(f"Model saved to {filepath}")

# === 6. LOAD ===
def load_model(filepath: str) -> Dict:
    """
    Load a previously saved model and its metadata from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model file (produced by `save_model`).

    Returns
    -------
    dict
        A dictionary containing the loaded model object and its associated metadata.

    Notes
    -----
    - Prints a confirmation message when the model is successfully loaded.
    """
    model_info = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_info
