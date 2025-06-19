import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import utils.model_utils as mu
import utils.shap_py as sp
import joblib
from typing import Dict, List, Optional

# === 1. LEARN ===
def learn_model(df, target_col, params=None, search=True, cv=5, scoring='recall', random_state=42):
    """
    Train an XGBoost classifier with optional hyper-parameter tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature columns, the target column, and a
        ``'dataset'`` indicator column (excluded from training).
    target_col : str
        Name of the target column to predict.
    params : dict, optional
        Hyper-parameters to evaluate or set. If ``None``, a default grid is
        used for tuning.
    search : bool, default=True
        If ``True`` and ``params`` is provided, performs grid search over
        ``params``; if ``False``, fits directly with the supplied ``params``.
    cv : int, default=5
        Number of cross-validation folds used during grid search.
    scoring : str, default='recall'
        Metric used for model selection in grid search.
    random_state : int, default=42
        Seed passed to ``XGBClassifier`` for reproducibility.

    Returns
    -------
    dict
        {
            'model'        : fitted ``XGBClassifier``,
            'model_params' : best or fixed parameter set as a ``dict``
        }

    Notes
    -----
    - Column ``'dataset'`` is always dropped before training.
    - Grid search is delegated to ``mu.grid_search``.
    """
    model = XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False)

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
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        best_params, best_model = mu.grid_search(model, X, y, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        'model': best_model,
        'model_params': best_params,
    }

# === 2. APPLY ===
def apply_model(df: pd.DataFrame, model_info: Dict, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply a trained XGBoost model to a DataFrame and append predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Data on which predictions are required.
    model_info : dict
        Dictionary with key ``'model'`` mapping to a fitted ``XGBClassifier``.
    columns : list[str], optional
        Column names to exclude during prediction (e.g., IDs). They are
        re-attached to the returned DataFrame unchanged.

    Returns
    -------
    pd.DataFrame
        Copy of the input data (excluding any temporarily dropped columns
        during inference) with two new columns:  
        - ``'predictions'``   : predicted class labels  
        - ``'probabilities'`` : probability of the positive class (index 1)

    Notes
    -----
    - Existing ``'predictions'`` or ``'probabilities'`` columns in *df* are
      removed before new values are added.
    - Positive-class probability is extracted as
      ``model.predict_proba(... )[:, 1]``.
    """
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

    # Predict
    predictions = model.predict(df_model_input)
    probabilities = model.predict_proba(df_model_input)[:, 1]

    # Construct result
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
