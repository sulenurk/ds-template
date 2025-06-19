import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, 
                            roc_auc_score, average_precision_score,
                            RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import cross_val_score, StratifiedKFold , GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve


def cros_val(model, X, y, cv=5):
    """
    Print cross-validated classification metrics.

    Parameters
    ----------
    model : estimator
        Scikit-learn–compatible classifier implementing `.fit`.
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y : array-like or pd.Series, shape (n_samples,)
        True labels.
    cv : int, default=5
        Number of stratified folds for cross-validation.

    Returns
    -------
    None
        Averages of Accuracy, Precision, Recall and F1 are printed to stdout.
    """

    cv_accuracy = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv), scoring='accuracy').mean()
    cv_precision = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv), scoring='precision').mean()
    cv_recall = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv), scoring='recall').mean()
    cv_f1 = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv), scoring='f1').mean()

    print("\nCross-validated Metrics:")
    print(f"Accuracy: {cv_accuracy:.4f}")
    print(f"Precision: {cv_precision:.4f}")
    print(f"Recall: {cv_recall:.4f}")
    print(f"F1 Score: {cv_f1:.4f}")

def grid_search(model, X, y, param_grid, cv=5, scoring='precision', n_jobs=-1):
  """
    Perform an exhaustive grid search and return the best model.

    Parameters
    ----------
    model : estimator
        Base estimator.
    X : array-like or pd.DataFrame, shape (n_samples, n_features)
        Feature matrix.
    y : array-like or pd.Series, shape (n_samples,)
        True labels.
    param_grid : dict
        Dictionary with parameter names (str) as keys and lists of settings.
    cv : int, default=5
        Number of folds for cross-validation.
    scoring : str or callable, default='precision'
        Metric used to evaluate parameter combinations.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 => use all processors).

    Returns
    -------
    tuple (dict, estimator)
        Best parameter set and the refitted best estimator.
    """
  grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
  grid_search.fit(X, y)
  best_params = grid_search.best_params_
  best_model = grid_search.best_estimator_

  return best_params, best_model

def feature_importance(model, X):
  """
    Return a sorted DataFrame of feature importances.

    Parameters
    ----------
    model : estimator
        Fitted model exposing the attribute ``feature_importances_``.
    X : pd.DataFrame
        Training features; column names are used as feature labels.

    Returns
    -------
    pd.DataFrame
        Columns: ``'Feature'`` and ``'Importance'``, sorted descending.
    """

  importances = model.feature_importances_
  feature_names = X.columns if hasattr(X, 'columns') else range(X.shape[1])
  importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

  return importance_df

def plot_confusion_matrix(y_true, y_pred, title=None):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    if title:
        plt.title(title)
    plt.show()

def plot_probability_metrics(y_true, probabilities):
    """
    Print probability-based scores and plot ROC & PR curves.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth labels.
    probabilities : array-like, shape (n_samples,)
        Predicted positive-class probabilities.

    Returns
    -------
    None
        Prints ROC-AUC and Average Precision; displays ROC and PR curves.
    """
    print(f"\n═══ Probability Metrics ═══")
    print(f"ROC AUC: {roc_auc_score(y_true, probabilities):.4f}")
    print(f"Average Precision: {average_precision_score(y_true, probabilities):.4f}")

    # ROC Curve
    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title('ROC Curve')
    plt.show()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title('Precision-Recall Curve')
    plt.show()

def plot_lift_curve(y_true, probabilities, n_bins=10):
    """
    Plot a lift chart based on decile analysis.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth labels.
    probabilities : array-like, shape (n_samples,)
        Predicted positive-class probabilities.
    n_bins : int, default=10
        Number of equal-frequency bins (deciles).

    Returns
    -------
    None
        Shows a bar chart of lift values by decile.
    """
    df = pd.DataFrame({'prob': probabilities, 'actual': y_true})
    df['decile'] = pd.qcut(df['prob'], n_bins, labels=False, duplicates='drop')

    lift_stats = df.groupby('decile').agg(
        avg_prob=('prob', 'mean'),
        response_rate=('actual', 'mean'),
        count=('actual', 'count')
    ).sort_index(ascending=False)

    lift_stats['lift'] = lift_stats['response_rate'] / df['actual'].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(lift_stats.index+1, lift_stats['lift'], color='dodgerblue')
    plt.axhline(y=1, color='red', linestyle='--')
    plt.xlabel('Decile (1=Highest Risk)')
    plt.ylabel('Lift (vs Random)')
    plt.title(f'Lift Chart (Top Decile Lift: {lift_stats.iloc[0]["lift"]:.1f}x)')
    plt.xticks(range(1, n_bins+1))
    plt.grid(True)
    plt.show()

def plot_cumulative_gains(y_true, probabilities):
    """
    Plot a cumulative gains curve.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth labels.
    probabilities : array-like, shape (n_samples,)
        Predicted positive-class probabilities.

    Returns
    -------
    None
        Displays the cumulative gains plot versus the random baseline.
    """
    df = pd.DataFrame({'prob': probabilities, 'actual': y_true})
    df = df.sort_values('prob', ascending=False)
    df['cum_capture'] = df['actual'].cumsum() / df['actual'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, 1, len(df)), df['cum_capture'], label='Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.xlabel('Percentage of Population')
    plt.ylabel('Percentage of Positive Cases')
    plt.title('Cumulative Gains Chart')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_calibration_curve(y_true, probabilities):
    """
    Plot a reliability (calibration) curve.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth labels.
    probabilities : array-like, shape (n_samples,)
        Predicted positive-class probabilities.

    Returns
    -------
    None
        Displays the calibration curve together with a perfect-calibration line.
    """
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10)

    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(y_true, y_pred, save_path=None):
    """
    Scatter plot of residuals for regression models.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like, shape (n_samples,)
        Predicted target values.
    save_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
        Shows (and optionally saves) the residual plot.
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def regression_evaluation(y_true, y_pred):
  """
    Print regression error metrics and display residual plot.

    Metrics reported:
    - RMSE (root mean squared error)
    - MAE  (mean absolute error)
    - R²   (coefficient of determination)

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth target values.
    y_pred : array-like, shape (n_samples,)
        Predicted target values.

    Returns
    -------
    None
        Prints numeric metrics and shows a residual scatter plot.
    """
  rmse = root_mean_squared_error(y_true, y_pred, squared=False)
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)

  print("═══ Regression Evaluation ═══")
  print(f"RMSE: {rmse:.4f}")
  print(f"MAE : {mae:.4f}")
  print(f"R²  : {r2:.4f}")

  plot_residuals(y_true, y_pred)