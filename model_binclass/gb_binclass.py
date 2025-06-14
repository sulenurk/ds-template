import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import utils.model_utils as mu
import utils.shap_py as sp
import joblib
from typing import Dict, List, Optional

# === 1. LEARN ===
def learn_model(df, target_col, params=None, search=True, cv=5, scoring='recall', random_state=42):
    """
    Train a Gradient Boosting model with optional hyperparameter tuning.

    This function trains a `GradientBoostingClassifier` on the given DataFrame.
    It supports custom parameter input or automatic grid search for tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features, the target column, and a 'dataset' column that will be excluded.
    target_col : str
        Name of the target column to be predicted.
    params : dict, optional
        Dictionary of hyperparameters to use for training. If None, default parameter grid will be used.
    search : bool, default=True
        If True and `params` is provided, performs grid search using the provided `params`.
        If False, fits the model directly using `params`.
    cv : int, default=5
        Number of cross-validation folds used during hyperparameter search.
    scoring : str, default='recall'
        Scoring metric used for selecting the best model during grid search.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary with:
        - 'model' (GradientBoostingClassifier): The trained model.
        - 'model_params' (dict): The best parameter set used for training.

    Notes
    -----
    - The column 'dataset' is excluded before training.
    - If no `params` are provided, a default grid is used for hyperparameter search.
    - Assumes binary or multiclass classification.
    """
    model = GradientBoostingClassifier(random_state=random_state)

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
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        best_params, best_model = mu.grid_search(model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        'model': best_model,
        'model_params': best_params,
    }

# === 2. APPLY ===
def apply_model(df, model_info: Dict, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Apply Gradient Boosting model to data and return predictions + probabilities"""
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
    """
    Apply a trained Gradient Boosting model to a DataFrame and append predictions.

    This function uses a fitted `GradientBoostingClassifier` to generate predictions
    and class probabilities from the given DataFrame. Optionally, specified columns 
    can be excluded during prediction but retained in the final output.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to apply the model on. Must include the same features used during training.
    model_info : dict
        A dictionary containing at least the key 'model' mapped to a trained GradientBoostingClassifier.
    columns : list of str, optional
        Columns to exclude during prediction (e.g., IDs or non-feature columns).
        These columns are added back to the returned DataFrame after prediction.

    Returns
    -------
    pd.DataFrame
        A copy of the input data (excluding dropped columns during prediction),
        with two added columns:
        - 'predictions': predicted class labels
        - 'probabilities': predicted probabilities for the positive class (class 1)

    Notes
    -----
    - Existing 'predictions' or 'probabilities' columns in `df` will be removed before generating new ones.
    - The positive class is assumed to be at index 1 in `predict_proba`.
    """
    y_true = df[target_col]
    y_pred = df['predictions']
    pred_proba = df['probabilities']

    print("═══ Classification Report ═══")
    print(classification_report(y_true, y_pred))

    mu.plot_confusion_matrix(y_true, y_pred)
    mu.plot_probability_metrics(y_true, pred_proba)
    mu.plot_lift_curve(y_true, pred_proba)
    mu.plot_cumulative_gains(y_true, pred_proba)