# src/evaluate.py
```python
import pandas as pd
import numpy as np
from openpyxl import Workbook
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss, f1_score, accuracy_score, recall_score
)


def calibration_plot_percentiles(y_true, y_prob, n_bins=10):
    """
    Calculate bin centers and true positive rates for a calibration plot using percentiles.
    """
    percentiles = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(y_prob, percentiles, right=True) - 1
    bin_indices[bin_indices == n_bins] = n_bins - 1
    bin_centers = np.array([y_prob[bin_indices == i].mean() for i in range(n_bins)])
    bin_true_means = np.array([y_true[bin_indices == i].mean() for i in range(n_bins)])
    # Fill NaN values with the mean
    bin_centers = np.nan_to_num(bin_centers, nan=np.nanmean(bin_centers))
    bin_true_means = np.nan_to_num(bin_true_means, nan=np.nanmean(bin_true_means))
    return bin_centers, bin_true_means


def bootstrap_ci(y_true, y_pred_or_prob, metric_func, n_bootstraps=2000, alpha=0.95):
    """
    Compute confidence intervals for a performance metric via bootstrap sampling.
    """
    rng = np.random.RandomState(42)
    scores = []
    y_true = np.array(y_true)
    y_vals = np.array(y_pred_or_prob)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_func(y_true[idx], y_vals[idx]))
    scores = np.array(scores)
    lower = np.percentile(scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(scores, (alpha + (1 - alpha) / 2) * 100)
    return lower, upper


def save_model_metrics(best_models, selected_features_dict, X_test, y_test, best_params_dict, file_name='model_evaluation.xlsx'):
    """
    Save evaluation metrics and their confidence intervals for each model to an Excel file.
    """
    wb = Workbook()
    ws = wb.active
    # Header row
    ws.append([
        'Model', 'ROC AUC', 'ROC AUC Lower', 'ROC AUC Upper',
        'PR AUC', 'PR AUC Lower', 'PR AUC Upper',
        'Brier Score', 'Log Loss', 'F1 Score', 'Accuracy', 'Recall',
        'Calibration Slope', 'Calibration Slope Lower', 'Calibration Slope Upper',
        'Best Params'
    ])
    # Evaluate each model
    for model_name, model in best_models.items():
        features = selected_features_dict[model_name]
        X_sel = X_test[features]
        y_prob = model.predict_proba(X_sel)[:, 1]
        y_pred = model.predict(X_sel)
        # Base metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        loss = log_loss(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        # Confidence intervals
        roc_low, roc_high = bootstrap_ci(y_test, y_prob, lambda yt, yp: auc(*roc_curve(yt, yp)[:2]))
        pr_low, pr_high = bootstrap_ci(y_test, y_prob, average_precision_score)
        # Calibration slope CI
        centers, truths = calibration_plot_percentiles(y_test, y_prob, n_bins=10)
        try:
            slope = np.polyfit(centers, truths, 1)[0]
            slope_low, slope_high = bootstrap_ci(
                y_test, y_prob,
                lambda yt, yp: np.polyfit(*calibration_plot_percentiles(yt, yp), 1)[0]
            )
        except Exception:
            slope, slope_low, slope_high = 0, 0, 0
        # Write to Excel
        ws.append([
            model_name,
            roc_auc, roc_low, roc_high,
            pr_auc, pr_low, pr_high,
            brier, loss, f1, acc, rec,
            slope, slope_low, slope_high,
            str(best_params_dict.get(model_name, ''))
        ])
    wb.save(file_name)
    print(f"Saved evaluation metrics to {file_name}")
```
