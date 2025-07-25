import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import utils.model_utils as mu
import utils.shap_py as sp
import joblib
from typing import Dict, List, Optional

# === 1. LEARN ===
def learn_model(df, target_col, params=None, search=True, cv=5, scoring='recall', random_state=42):
    """
    Train a Decision Tree model with optional hyperparameter tuning.

    This function trains a `DecisionTreeClassifier` on the provided DataFrame.
    It supports manual parameter input or automatic grid search for hyperparameter optimization.

    Parameters:
        df (pd.DataFrame): Input dataset containing features, target column, and a 'dataset' indicator column.
        target_col (str): Name of the target column to be predicted.
        params (dict, optional): Dictionary of hyperparameters to set for the model. If None, default grid search is used.
        search (bool, default=True): If True and `params` is provided, performs grid search over the given `params`. 
                                     If False, trains directly with provided `params`.
        cv (int, default=5): Number of cross-validation folds used in grid search.
        scoring (str, default='recall'): Scoring metric used for model selection during hyperparameter tuning.
        random_state (int, default=42): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing:
            - 'model' (DecisionTreeClassifier): The trained decision tree model.
            - 'model_params' (dict): The best parameters used during training.

    Notes:
        - The column named 'dataset' is dropped from `df` before training.
        - If `params` is not provided, a predefined hyperparameter grid is used for tuning.
    
    """ 
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
    """
    Apply a trained Decision Tree model to a DataFrame and append predictions.

    The function:
    1. Drops the specified `columns` (if any) **only** for inference,  
       but re-inserts them into the output to preserve the original schema.  
    2. Removes existing `"predictions"` or `"probabilities"` columns to avoid
       accidental overwriting conflicts.  
    3. Generates class predictions and positive-class probabilities, and
       returns the DataFrame with two new columns:  
       - **predictions**  (int or str)  
       - **probabilities** (float in \[0, 1\])

    Parameters
    ----------
    df : pd.DataFrame
        Input data on which predictions will be made.
    model_info : Dict
        Dictionary returned by `learn_model`; must contain at least the key
        `'model'` mapped to a fitted `DecisionTreeClassifier`.
    columns : list[str], optional
        Column names to exclude from the model’s feature set during inference.
        These columns are concatenated back to the result unchanged.

    Returns
    -------
    pd.DataFrame
        A copy of the input (excluding any temporarily dropped columns during
        prediction) with two additional columns:
        `'predictions'` and `'probabilities'`.

    Notes
    -----
    - The model is assumed to have been trained on
      `df.drop(columns=columns)` (plus any other preprocessing).
    - If `'predictions'` or `'probabilities'` already exist in `df`, they are
      removed before new values are added.
    - The positive-class probability is extracted as `model.predict_proba(... )[:, 1]`.

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

    print("\n═══ Train Set Classification Report ═══")
    print(classification_report(y_true, y_pred))
    mu.plot_confusion_matrix(y_true, y_pred)
    
    #Test Set Evaluation
    if test_set.empty:
        print("No data found for the test set.")
        return

    y_true = test_set[target_col]
    y_pred = test_set['predictions']

    print("\n═══ Test Set Classification Report ═══")
    print(classification_report(y_true, y_pred))
    mu.plot_confusion_matrix(y_true, y_pred)
