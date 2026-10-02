"""
train_model.py
---------------
End-to-end training script for the Heart Disease Prediction project.

Run with:
    python src/train_model.py

This script:
1. Loads and cleans the dataset (src/preprocessing.py).
2. Splits data into stratified train/test sets (80/20, random_state=42).
3. Builds two full pipelines (preprocessing + model):
       - Logistic Regression
       - Decision Tree Classifier
4. Evaluates both models on the held-out test set using Accuracy, Precision,
   Recall, F1-score, ROC-AUC, a classification report and a confusion matrix.
5. Saves confusion-matrix and ROC-curve plots to outputs/.
6. Saves a model comparison table (CSV) to outputs/.
7. Selects the better model (by ROC-AUC, with Recall as a tie-breaker,
   since missing an actual heart-disease case is the costlier error here)
   and saves the FULL fitted pipeline (preprocessing + model) to
   models/heart_disease_model.pkl using joblib.
8. Saves feature-importance / coefficient plots for interpretation.

No results in this script are hard-coded - every number is produced by
actually fitting and evaluating the models on this run.
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    get_preprocessor,
    load_and_prepare,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "heart_disease_uci.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "confusion_matrix"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "roc_curve"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "eda"), exist_ok=True)

RANDOM_STATE = 42


def main():
    print("=" * 70)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------------
    df, X, y = load_and_prepare(DATA_PATH)
    print(f"\nDataset shape after cleaning: {df.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # ------------------------------------------------------------------
    # 2. Stratified train/test split (80/20)
    # ------------------------------------------------------------------
    # Stratification keeps the same proportion of heart-disease-positive
    # cases in both the training and test sets as in the full dataset.
    # Without it, a random split could (by chance) under- or over-represent
    # the positive class in the test set, making evaluation metrics
    # unreliable, especially for a moderately imbalanced target like this.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 3. Build pipelines (preprocessing + model) - avoids data leakage
    #    because the preprocessor is fit only on X_train inside .fit()
    # ------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
    }

    fitted_pipelines = {}
    results = []

    plt.figure(figsize=(7, 6))
    ax_roc = plt.gca()

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", model),
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        })

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred,
                                      target_names=["No Disease", "Disease"]))
        print(f"ROC-AUC: {auc:.4f}")

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Disease", "Disease"],
        )
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title(f"Confusion Matrix - {name}")
        fig_cm.tight_layout()
        fname = name.lower().replace(" ", "_")
        fig_cm.savefig(
            os.path.join(OUTPUTS_DIR, "confusion_matrix", f"cm_{fname}.png"),
            dpi=150,
        )
        plt.close(fig_cm)

        # Add this model's ROC curve to the shared ROC figure
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax_roc)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_title("ROC Curve Comparison - Logistic Regression vs Decision Tree")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "roc_curve", "roc_comparison.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 4. Comparison table
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (actual test-set results)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)

    # ------------------------------------------------------------------
    # 5. Select final model
    #    Primary criterion: ROC-AUC (overall discriminative ability).
    #    Tie-breaker: Recall (minimising missed positive/disease cases
    #    is clinically more important than a false alarm in this context).
    # ------------------------------------------------------------------
    results_df_sorted = results_df.sort_values(
        by=["ROC-AUC", "Recall"], ascending=False
    ).reset_index(drop=True)
    best_model_name = results_df_sorted.loc[0, "Model"]
    print(f"\nSelected final model: {best_model_name}")
    best_pipeline = fitted_pipelines[best_model_name]

    # ------------------------------------------------------------------
    # 6. Feature importance / interpretation
    # ------------------------------------------------------------------
    preprocessor = best_pipeline.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    plt.figure(figsize=(8, 6))
    if best_model_name == "Decision Tree":
        importances = best_pipeline.named_steps["classifier"].feature_importances_
        imp_series = pd.Series(importances, index=all_feature_names).sort_values(
            ascending=False
        ).head(15)
        sns.barplot(x=imp_series.values, y=imp_series.index, color="#3b7dd8")
        plt.title(f"Top 15 Feature Importances - {best_model_name}")
        plt.xlabel("Importance")
    else:
        coefs = best_pipeline.named_steps["classifier"].coef_[0]
        coef_series = pd.Series(coefs, index=all_feature_names)
        coef_series = coef_series.reindex(
            coef_series.abs().sort_values(ascending=False).head(15).index
        )
        colors = ["#d84343" if v < 0 else "#3b7dd8" for v in coef_series.values]
        bars_ax = sns.barplot(x=coef_series.values, y=coef_series.index, hue=coef_series.index, palette=colors, legend=False)
        plt.title(f"Top 15 Feature Coefficients - {best_model_name}")
        plt.xlabel("Coefficient (log-odds impact)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "eda", "feature_importance.png"), dpi=150)
    plt.close()

    # Also save Decision Tree importances and Logistic Regression coefficients
    # for BOTH models regardless of which was selected, for the report/app.
    for name, pipe in fitted_pipelines.items():
        clf = pipe.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
        else:
            vals = clf.coef_[0]
        pd.Series(vals, index=all_feature_names).sort_values(
            ascending=False
        ).to_csv(os.path.join(OUTPUTS_DIR, f"importance_{name.lower().replace(' ', '_')}.csv"))

    # ------------------------------------------------------------------
    # 7. Save the full pipeline (preprocessing + best model)
    # ------------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "heart_disease_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved final model pipeline to: {model_path}")

    # Save metadata needed by the web app to build the input form
    metadata = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_options": {
            col: sorted([v for v in X[col].dropna().unique().tolist()])
            for col in CATEGORICAL_FEATURES
        },
        "numerical_ranges": {
            col: {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "median": float(X[col].median()),
            }
            for col in NUMERICAL_FEATURES
        },
        "best_model_name": best_model_name,
        "all_results": results,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved model metadata to: models/model_metadata.json")

    # Save all fitted pipelines too (used by the Streamlit "Model Performance"
    # page to show both models' metrics/plots without retraining).
    joblib.dump(fitted_pipelines, os.path.join(MODELS_DIR, "all_pipelines.pkl"))

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
