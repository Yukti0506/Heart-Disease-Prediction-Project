"""
eda.py
------
Exploratory Data Analysis script. Generates all EDA plots used in the
notebook, the report, the presentation, and the Streamlit "Dataset
Insights" page. Run with:

    python src/eda.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import clean_data, add_binary_target, load_raw_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "heart_disease_uci.csv")
EDA_DIR = os.path.join(PROJECT_ROOT, "outputs", "eda")
os.makedirs(EDA_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = {"No Disease": "#3b7dd8", "Disease": "#d84343"}


def label(df):
    d = df.copy()
    d["Diagnosis"] = d["heart_disease"].map({0: "No Disease", 1: "Disease"})
    return d


def main():
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    df = add_binary_target(df)
    df = label(df)

    print("Shape:", df.shape)
    print("\nFirst rows:\n", df.head())
    print("\nColumns:", list(df.columns))
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())

    # 1. Target distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="Diagnosis", hue="Diagnosis",
                        palette=PALETTE, legend=False)
    ax.set_title("Heart Disease vs No Heart Disease Distribution")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of Patients")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "target_distribution.png"), dpi=150)
    plt.close()

    # 2. Age distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="age", hue="Diagnosis", kde=True, palette=PALETTE,
                 element="step", bins=25)
    plt.title("Age Distribution by Diagnosis")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "age_distribution.png"), dpi=150)
    plt.close()

    # 3. Resting blood pressure
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Diagnosis", y="trestbps", hue="Diagnosis",
                palette=PALETTE, legend=False)
    plt.title("Resting Blood Pressure by Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Resting Blood Pressure (mm Hg)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "resting_bp.png"), dpi=150)
    plt.close()

    # 4. Cholesterol
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Diagnosis", y="chol", hue="Diagnosis",
                palette=PALETTE, legend=False)
    plt.title("Cholesterol Level by Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Serum Cholesterol (mg/dl)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "cholesterol_distribution.png"), dpi=150)
    plt.close()

    # 5. Maximum heart rate
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="thalch", hue="Diagnosis", kde=True, palette=PALETTE,
                 element="step", bins=25)
    plt.title("Maximum Heart Rate Achieved Distribution")
    plt.xlabel("Maximum Heart Rate (bpm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "max_heart_rate.png"), dpi=150)
    plt.close()

    # 6. Oldpeak
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Diagnosis", y="oldpeak", hue="Diagnosis",
                palette=PALETTE, legend=False)
    plt.title("ST Depression (Oldpeak) by Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Oldpeak")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "oldpeak_distribution.png"), dpi=150)
    plt.close()

    # 7. Sex vs target
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="sex", hue="Diagnosis", palette=PALETTE)
    ax.set_title("Heart Disease by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "sex_vs_target.png"), dpi=150)
    plt.close()

    # 8. Chest pain type vs target
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, y="cp", hue="Diagnosis", palette=PALETTE)
    ax.set_title("Chest Pain Type vs Diagnosis")
    ax.set_xlabel("Number of Patients")
    ax.set_ylabel("Chest Pain Type")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "chest_pain_vs_target.png"), dpi=150)
    plt.close()

    # 9. Fasting blood sugar vs target
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="fbs", hue="Diagnosis", palette=PALETTE)
    ax.set_title("Fasting Blood Sugar (>120 mg/dl) vs Diagnosis")
    ax.set_xlabel("Fasting Blood Sugar > 120 mg/dl")
    ax.set_ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "fbs_vs_target.png"), dpi=150)
    plt.close()

    # 10. Resting ECG vs target
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, y="restecg", hue="Diagnosis", palette=PALETTE)
    ax.set_title("Resting ECG Result vs Diagnosis")
    ax.set_xlabel("Number of Patients")
    ax.set_ylabel("Resting ECG Result")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "restecg_vs_target.png"), dpi=150)
    plt.close()

    # 11. Exercise induced angina vs target
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="exang", hue="Diagnosis", palette=PALETTE)
    ax.set_title("Exercise-Induced Angina vs Diagnosis")
    ax.set_xlabel("Exercise-Induced Angina")
    ax.set_ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "exang_vs_target.png"), dpi=150)
    plt.close()

    # 12. Slope vs target
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="slope", hue="Diagnosis", palette=PALETTE)
    ax.set_title("Slope of Peak Exercise ST Segment vs Diagnosis")
    ax.set_xlabel("Slope")
    ax.set_ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "slope_vs_target.png"), dpi=150)
    plt.close()

    # 13. Thalassemia vs target
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="thal", hue="Diagnosis", palette=PALETTE)
    ax.set_title("Thalassemia Result vs Diagnosis")
    ax.set_xlabel("Thalassemia")
    ax.set_ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "thal_vs_target.png"), dpi=150)
    plt.close()

    # 14. Correlation heatmap of numeric features
    plt.figure(figsize=(7, 6))
    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "heart_disease"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    print("\nAll EDA plots saved to outputs/eda/")


if __name__ == "__main__":
    main()
