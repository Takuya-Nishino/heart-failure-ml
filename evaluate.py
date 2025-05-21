import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss, f1_score, accuracy_score, recall_score
)
from openpyxl import Workbook

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def load_results(model_path: str):
    data = joblib.load(model_path)
    return data['model'], data['features']

def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    loss = log_loss(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier': brier,
        'log_loss': loss,
        'f1': f1,
        'accuracy': acc,
        'recall': rec
    }

def main(args):
    df = pd.read_excel(args.input)
    X = df.drop(columns=[args.target, 'INDEX', 'Day'])
    y = df[args.target]
    wb = Workbook()
    ws = wb.active
    headers = ['Model', 'ROC AUC', 'PR AUC', 'Brier', 'Log Loss', 'F1', 'Accuracy', 'Recall']
    ws.append(headers)

    for model_name in args.models.split(','):
        model, features = load_results(f"models/{model_name}_best.pkl")
        res = evaluate(model, X[features], y)
        ws.append([model_name] + [res[h.lower()] for h in headers[1:]])

    wb.save(args.output)
    logging.info(f"Saved evaluation results to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to Excel data file')
    parser.add_argument('--models', default='LogisticRegression,RandomForest,XGBoost,LightGBM', help='Comma-separated model names')
    parser.add_argument('--target', default='Event180', help='Target column name')
    parser.add_argument('--output', default='evaluation.xlsx', help='Excel output path')
    args = parser.parse_args()
    main(args)
