"""
preprocessing.py
-----------------
Data cleaning and preprocessing pipeline for the Heart Disease Prediction project.

This module is responsible for:
1. Loading the raw UCI Heart Disease dataset.
2. Cleaning it (fixing dtypes, standardising missing-value markers).
3. Creating the binary target column `heart_disease` from the multi-class `num` column.
4. Building a scikit-learn ColumnTransformer that imputes + encodes/scales
   features consistently, so the exact same transformations are applied
   at training time and at prediction time (no data leakage).
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
# Numerical (continuous) clinical measurements
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

# Categorical clinical attributes
CATEGORICAL_FEATURES = [
    "sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal",
]

TARGET_RAW = "num"
TARGET = "heart_disease"

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Heart Disease UCI dataframe.

    Decisions made here (and why):
    - The dataset has no literal '?' strings (pandas already parsed missing
      cells as NaN), but we standardise any stray '?' just in case the file
      is regenerated from the original UCI source, which does use '?'.
    - `fbs` and `exang` are boolean (True/False) with some missing values;
      they are cast to a nullable categorical (object) dtype so the
      downstream imputer/encoder can treat them uniformly with the other
      categorical columns.
    - `id` is dropped: it is a row identifier with no predictive meaning and
      would leak record order into the model.
    - We do NOT drop rows/columns just because they contain missing values.
      `ca`, `thal` and `slope` have a large proportion of missing values in
      this multi-site dataset (they were not collected consistently at every
      site), so dropping them would discard most of the dataset. Instead,
      missing values are handled later by imputation inside the modelling
      pipeline (median for numeric columns, most-frequent for categorical
      columns), which is fit only on the training split to avoid leakage.
    - Duplicate rows (if any) are removed.
    """
    df = df.copy()

    # Standardise any '?' markers (defensive - source file for this project
    # already encodes missing values as empty/NaN, not '?').
    df.replace("?", np.nan, inplace=True)

    # Drop the row-identifier column - not a predictive feature.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Remove exact duplicate records.
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate row(s).")

    # Cast boolean-like columns to object/categorical so they are handled
    # consistently by the categorical branch of the preprocessing pipeline.
    for col in ["fbs", "exang"]:
        if col in df.columns:
            df[col] = df[col].map({True: "True", False: "False"}).where(
                df[col].notna(), np.nan
            )

    # Ensure numeric columns are numeric dtype (coerce any stray strings).
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the multi-class `num` column (0-4, representing increasing
    disease severity/number of major vessels affected) into a binary
    classification target:
        0                -> 0 (No heart disease)
        1, 2, 3, 4        -> 1 (Heart disease present)
    """
    df = df.copy()
    df[TARGET] = (df[TARGET_RAW] > 0).astype(int)
    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer used inside the full modelling Pipeline.

    Numerical features: median imputation + standard scaling.
    Categorical features: most-frequent imputation + one-hot encoding.

    Using a ColumnTransformer wrapped inside a Pipeline (see train_model.py)
    guarantees that:
      - Imputation statistics (medians / modes) and scaling parameters are
        learned ONLY on the training fold and re-used, unchanged, on the
        test fold and on any new patient data submitted through the web app.
      - The exact same transformation is applied at prediction time, so the
        saved .pkl file can accept raw, human-readable input directly.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def load_and_prepare(csv_path: str):
    """Convenience function: load -> clean -> add target -> split X/y."""
    df = load_raw_data(csv_path)
    df = clean_data(df)
    df = add_binary_target(df)

    X = df[ALL_FEATURES]
    y = df[TARGET]
    return df, X, y
