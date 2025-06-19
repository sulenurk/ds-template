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
    Evaluate classification results separately for training and test datasets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - true labels in *target_col*
        - a 'predictions' column (required)
        - a 'probabilities' column (optional) for probability-based plots
        - a 'dataset' column with values 1 for training and 0 for test
    target_col : str
        Name of the column holding true class labels.

    Returns
    -------
    None
    """
    train_set = df[df['dataset'] == 1]
    test_set = df[df['dataset'] == 0]

    #Train Set Evaluation
    if train_set.empty:
        print("No data found for the training set.")
        return

    y_true = train_set[target_col]
    y_pred = train_set['predictions']
    pred_proba = train_set['probabilities'] if 'probabilities' in train_set.columns else None

    print("\n═══ Train Set Classification Report ═══")
    print(classification_report(y_true, y_pred))
    mu.plot_confusion_matrix(y_true, y_pred)

    if pred_proba is not None:
        mu.plot_probability_metrics(y_true, pred_proba)
        mu.plot_lift_curve(y_true, pred_proba)
        mu.plot_cumulative_gains(y_true, pred_proba)
        mu.plot_calibration_curve(y_true, pred_proba)
    else:
        print("Note: Train set probability-based plots skipped (no probabilities found).")

    #Test Set Evaluation
    if test_set.empty:
        print("No data found for the test set.")
        return

    y_true = test_set[target_col]
    y_pred = test_set['predictions']
    pred_proba = test_set['probabilities'] if 'probabilities' in test_set.columns else None

    print("\n═══ Test Set Classification Report ═══")
    print(classification_report(y_true, y_pred))
    mu.plot_confusion_matrix(y_true, y_pred)

    if pred_proba is not None:
        mu.plot_probability_metrics(y_true, pred_proba)
        mu.plot_lift_curve(y_true, pred_proba)
        mu.plot_cumulative_gains(y_true, pred_proba)
        mu.plot_calibration_curve(y_true, pred_proba)
    else:
        print("Note: Test set probability-based plots skipped (no probabilities found).")    