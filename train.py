# src/train.py
```python
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

# Constants
SEED = 42
CV = RepeatedStratifiedKFold(n_splits=10, random_state=SEED)
MODEL_DIR = 'models'

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from an Excel file.
    """
    logging.info(f"Loading data from {path}")
    return pd.read_excel(path)


def preprocess(df: pd.DataFrame, target: str):
    """
    Split the DataFrame into features and target, and identify numeric and categorical columns.
    """
    X = df.drop(columns=[target, 'INDEX', 'Day'])
    y = df[target]
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return X, y, numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols, estimator):
    """
    Create a preprocessing and modeling pipeline.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    return pipeline


def select_and_tune(X, y, pipeline, param_grid):
    """
    Perform RFE with cross-validation and grid search for hyperparameter tuning.
    """
    # Recursive Feature Elimination with Cross-Validation
    selector = RFECV(
        estimator=pipeline.named_steps['classifier'],
        step=1,
        cv=CV,
        scoring='roc_auc',
        n_jobs=-1,
        min_features_to_select=20
    )
    selector.fit(pipeline.named_steps['preprocessor'].fit_transform(X), y)
    selected_mask = selector.support_
    selected_cols = X.columns[selected_mask]
    logging.info(f"Selected {len(selected_cols)} features: {list(selected_cols)}")

    # Grid search on the reduced feature set
    X_reduced = X[selected_cols]
    grid_search = GridSearchCV(
        estimator=pipeline.set_params(**{"classifier": pipeline.named_steps['classifier']}),
        param_grid={f'classifier__{k}': v for k, v in param_grid.items()},
        cv=CV,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_reduced, y)
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_, selected_cols


def main(args):
    """
    Main function: load data, preprocess, train and tune models, and save best pipelines.
    """
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and preprocess data
    df = load_data(args.input)
    X, y, numeric_cols, categorical_cols = preprocess(df, args.target)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=SEED
    )

    # Define models and hyperparameter grids
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

    # Train and tune each model
    for name, (estimator, param_grid) in models.items():
        logging.info(f"Training and tuning {name}")
        pipeline = build_pipeline(numeric_cols, categorical_cols, estimator)
        best_pipeline, selected_features = select_and_tune(X_train, y_train, pipeline, param_grid)
        # Save the best pipeline and selected features
        joblib.dump((best_pipeline, list(selected_features)), os.path.join(MODEL_DIR, f"{name}_best.pkl"))

    logging.info("All models trained and tuned successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and tune heart failure prognosis models")
    parser.add_argument('--input', required=True, help='Path to the input Excel data file')
    parser.add_argument('--target', default='Event180', help='Name of the target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data for validation')
    args = parser.parse_args()
    main(args)
```
