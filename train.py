import argparse
import logging
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb

# 定数
SEED = 42
CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
MODEL_DIR = 'models'

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading data from {path}")
    return pd.read_excel(path)


def preprocess(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target, 'INDEX', 'Day'])
    y = df[target]
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return X, y, numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols, estimator):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def select_and_tune(X, y, pipeline, param_grid):
    # RFE + CV
    selector = RFECV(
        estimator=pipeline.named_steps['classifier'],
        step=1,
        cv=CV,
        scoring='roc_auc',
        n_jobs=-1,
        min_features_to_select=20
    )
    selector.fit(pipeline.named_steps['preprocessor'].fit_transform(X), y)
    mask = selector.support_
    sel_cols = X.columns[mask]
    logging.info(f"Selected {len(sel_cols)} features: {list(sel_cols)}")

    # Pipeline with selected features
    X_sel = X.loc[:, sel_cols]
    tuned = GridSearchCV(
        estimator=pipeline.set_params(**{"classifier": pipeline.named_steps['classifier']}),
        param_grid={f'classifier__{k}': v for k, v in param_grid.items()},
        cv=CV,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    tuned.fit(X_sel, y)
    logging.info(f"Best params: {tuned.best_params_}")
    return tuned.best_estimator_, sel_cols


def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data(args.input)
    X, y, num_cols, cat_cols = preprocess(df, args.target)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=SEED
    )
    models = {
        'LogisticRegression': (
            LogisticRegression(max_iter=10000, solver='liblinear', class_weight='balanced'),
            {'C': [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=SEED),
            {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        ),
        'XGBoost': (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED),
            {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 5, 7, 10],
                'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
                'subsample': [0.5, 0.7, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2],
                'min_child_weight': [1, 3, 5]
            }
        ),
        'LightGBM': (
            lgb.LGBMClassifier(random_state=SEED),
            {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 5, 7, 10],
                'num_leaves': [31, 64, 128],
                'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
                'subsample': [0.5, 0.7, 0.9, 1.0],
                'min_child_samples': [5, 10, 20]
            }
        )
    }
    for name, (estimator, param_grid) in models.items():
        logging.info(f"Processing {name}")
        pipe = build_pipeline(num_cols, cat_cols, estimator)
        best_model, features = select_and_tune(X_train, y_train, pipe, param_grid)
        joblib.dump((best_model, list(features)), os.path.join(MODEL_DIR, f"{name}_best.pkl"))

    logging.info("Training and tuning completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to Excel data file')
    parser.add_argument('--target', default='Event180', help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation set fraction')
    args = parser.parse_args()
    main(args)
