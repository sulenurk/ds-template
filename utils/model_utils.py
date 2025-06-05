import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                            roc_auc_score, average_precision_score,
                            RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import cross_val_score, StratifiedKFold , GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def cros_val(model, X, y, cv=5):
  # Cross-validated metrics
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

  grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
  grid_search.fit(X, y)
  best_params = grid_search.best_params_
  best_model = grid_search.best_estimator_

  return best_params, best_model

def feature_importance(model, X):

  importances = model.feature_importances_
  feature_names = X.columns if hasattr(X, 'columns') else range(X.shape[1])
  importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

  return importance_df

def plot_confusion_matrix(y_true, y_pred):
    """Confusion matrix visualization (All classifiers)"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    title = 'Confusion Matrix'
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()

def plot_probability_metrics(y_true, probabilities):
    """ROC and Precision-Recall plots (Probability-based classifiers)"""
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
    """Lift curve (Best for marketing/models with good probability calibration)"""
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
    """Cumulative gains chart (All probability-based classifiers)"""
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
    """Calibration curve (Best for logistic regression and well-calibrated models)"""
    from sklearn.calibration import calibration_curve
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
  rmse = root_mean_squared_error(y_true, y_pred, squared=False)
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)

  print("═══ Regression Evaluation ═══")
  print(f"RMSE: {rmse:.4f}")
  print(f"MAE : {mae:.4f}")
  print(f"R²  : {r2:.4f}")

  plot_residuals(y_true, y_pred)