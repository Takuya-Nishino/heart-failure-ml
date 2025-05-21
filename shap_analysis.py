import argparse
import logging
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading data from {path}")
    return pd.read_excel(path)

def main(args):
    pipeline, features = joblib.load(args.model)
    df = load_data(args.input)
    X = df[features]

    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(pipeline.named_steps['preprocessor'].transform(X))

    shap.summary_plot(
        shap_values,
        X,
        feature_names=features,
        show=False
    )
    plt.tight_layout()
    plt.savefig(args.output)
    logging.info(f"Saved SHAP plot to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model pickle file')
    parser.add_argument('--input', required=True, help='Path to Excel data file')
    parser.add_argument('--output', default='shap_summary.png', help='SHAP plot output path')
    args = parser.parse_args()
    main(args)
