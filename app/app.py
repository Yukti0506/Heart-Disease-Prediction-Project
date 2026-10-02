"""
app.py
------
Streamlit web application for the Heart Disease Prediction System.

Run with:
    streamlit run app/app.py

Pages:
    - Home                : project introduction, objective, disclaimer
    - Predict              : interactive patient input form + prediction
    - Model Performance   : accuracy/precision/recall/F1/ROC-AUC, confusion
                             matrices, ROC curves, model comparison table
    - Dataset Insights     : EDA visualizations
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Make src/ importable
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from predict import load_model, load_metadata, predict_single  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ---------------------------------------------------------------------------
# Page config & light healthcare-style theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="\U0001FAC0",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 1100px;}
    h1, h2, h3 {color: #16324f;}
    .metric-card {
        background-color: #f4f8fb;
        border: 1px solid #dbe6ee;
        border-radius: 10px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .result-positive {
        background-color: #fdecea;
        border-left: 5px solid #c0392b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
    }
    .result-negative {
        background-color: #eafaf1;
        border-left: 5px solid #1e8449;
        padding: 1rem 1.2rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_all_pipelines():
    return joblib.load(os.path.join(MODELS_DIR, "all_pipelines.pkl"))


@st.cache_data
def get_metadata():
    return load_metadata()


@st.cache_data
def get_comparison_table():
    return pd.read_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("\U00002764 Heart Disease Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict", "Model Performance", "Dataset Insights"],
)

# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("Heart Disease Prediction System")
    st.subheader("Machine Learning Based Heart Disease Risk Prediction")

    st.write(
        "This project is an end-to-end Machine Learning system that "
        "estimates the likelihood of a patient having heart disease based "
        "on clinical measurements such as age, blood pressure, cholesterol, "
        "chest pain type and other diagnostic indicators from the UCI Heart "
        "Disease dataset."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Project Objective")
        st.write(
            "Build and compare Machine Learning classification models "
            "(Logistic Regression and Decision Tree) that predict the "
            "presence of heart disease, evaluate them with standard "
            "classification metrics, and deploy the best-performing model "
            "through an interactive web application."
        )
    with col2:
        st.markdown("### Technologies Used")
        st.markdown(
            "- **Python** - core language\n"
            "- **Pandas / NumPy** - data handling\n"
            "- **Matplotlib / Seaborn** - visualization\n"
            "- **Scikit-learn** - preprocessing & ML models\n"
            "- **Joblib** - model persistence\n"
            "- **Streamlit** - web application"
        )

    st.markdown("### How to use this application")
    st.markdown(
        "1. Go to the **Predict** page and fill in the patient's clinical details.\n"
        "2. Click **Predict** to see the model's output and confidence score.\n"
        "3. Visit **Model Performance** to review how the models were evaluated.\n"
        "4. Visit **Dataset Insights** to explore the underlying dataset."
    )

# ---------------------------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------------------------
elif page == "Predict":
    st.title("Patient Risk Prediction")
    st.write("Enter the patient's clinical details below, then click **Predict**.")

    meta = get_metadata()
    ranges = meta["numerical_ranges"]

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Age (years)", 18, 100, int(ranges["age"]["median"]))
            sex = st.selectbox("Sex", ["Male", "Female"])
            dataset_site = st.selectbox(
                "Data Source / Site", meta["categorical_options"]["dataset"],
                help="The clinical site where the original data was recorded.",
            )
            cp_label_map = {
                "Typical Angina": "typical angina",
                "Atypical Angina": "atypical angina",
                "Non-anginal Pain": "non-anginal",
                "Asymptomatic": "asymptomatic",
            }
            cp_choice = st.selectbox("Chest Pain Type", list(cp_label_map.keys()))
            cp = cp_label_map[cp_choice]

        with c2:
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)", min_value=60.0, max_value=250.0,
                value=round(ranges["trestbps"]["median"], 1), step=1.0,
            )
            chol = st.number_input(
                "Serum Cholesterol (mg/dl)", min_value=0.0, max_value=650.0,
                value=round(ranges["chol"]["median"], 1), step=1.0,
            )
            fbs_choice = st.radio(
                "Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], horizontal=True,
            )
            fbs = "True" if fbs_choice == "Yes" else "False"
            restecg_label_map = {
                "Normal": "normal",
                "ST-T Wave Abnormality": "st-t abnormality",
                "Left Ventricular Hypertrophy": "lv hypertrophy",
            }
            restecg_choice = st.selectbox("Resting ECG Result", list(restecg_label_map.keys()))
            restecg = restecg_label_map[restecg_choice]

        with c3:
            thalch = st.slider(
                "Maximum Heart Rate Achieved (bpm)", 60, 220,
                int(ranges["thalch"]["median"]),
            )
            exang_choice = st.radio(
                "Exercise-Induced Angina?", ["No", "Yes"], horizontal=True,
            )
            exang = "True" if exang_choice == "Yes" else "False"
            oldpeak = st.slider(
                "Oldpeak (ST depression induced by exercise)", -3.0, 7.0,
                round(ranges["oldpeak"]["median"], 1), step=0.1,
            )
            slope_label_map = {
                "Upsloping": "upsloping",
                "Flat": "flat",
                "Downsloping": "downsloping",
            }
            slope_choice = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_label_map.keys()))
            slope = slope_label_map[slope_choice]

        c4, c5 = st.columns(2)
        with c4:
            ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
        with c5:
            thal_label_map = {
                "Normal": "normal",
                "Fixed Defect": "fixed defect",
                "Reversible Defect": "reversable defect",
            }
            thal_choice = st.selectbox("Thalassemia", list(thal_label_map.keys()))
            thal = thal_label_map[thal_choice]

        submitted = st.form_submit_button("\U0001F50D Predict", use_container_width=True)

    if submitted:
        patient = {
            "age": age,
            "sex": sex,
            "dataset": dataset_site,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalch": thalch,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": float(ca),
            "thal": thal,
        }

        pipeline = get_model()
        prediction, probability = predict_single(pipeline, patient)

        st.markdown("### Prediction Result")
        if prediction == 1:
            st.markdown(
                f'<div class="result-positive">'
                f'<h4>Heart Disease Prediction: Positive</h4>'
                f'<p>Predicted probability: <b>{probability * 100:.1f}%</b></p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-negative">'
                f'<h4>No Heart Disease Detected by the Model</h4>'
                f'<p>Predicted probability of disease: <b>{probability * 100:.1f}%</b></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    comp = get_comparison_table()
    meta = get_metadata()
    st.write(
        f"Final selected model: **{meta['best_model_name']}** "
        "(selected by ROC-AUC, with Recall as a tie-breaker)."
    )

    st.markdown("### Evaluation Metrics (actual test-set results)")
    st.dataframe(comp.set_index("Model"), use_container_width=True)

    metric_cols = st.columns(5)
    best_row = comp[comp["Model"] == meta["best_model_name"]].iloc[0]
    for col, metric in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]):
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{best_row[metric]*100:.1f}%</h3>'
                f'<p>{metric}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Confusion Matrices")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.image(os.path.join(OUTPUTS_DIR, "confusion_matrix", "cm_logistic_regression.png"),
                  caption="Logistic Regression", use_container_width=True)
    with cm2:
        st.image(os.path.join(OUTPUTS_DIR, "confusion_matrix", "cm_decision_tree.png"),
                  caption="Decision Tree", use_container_width=True)

    st.markdown("### ROC Curve Comparison")
    st.image(os.path.join(OUTPUTS_DIR, "roc_curve", "roc_comparison.png"),
              use_container_width=True)

    st.markdown("### Feature Importance / Interpretation")
    st.image(os.path.join(OUTPUTS_DIR, "eda", "feature_importance.png"),
              use_container_width=True)
    st.caption(
        "These values show which patient attributes the selected model relies on "
        "most when making predictions. They reflect statistical association "
        "learned from this dataset, not medical causation."
    )

# ---------------------------------------------------------------------------
# DATASET INSIGHTS PAGE
# ---------------------------------------------------------------------------
elif page == "Dataset Insights":
    st.title("Dataset Insights")
    st.write(
        "Exploratory analysis of the UCI Heart Disease dataset used to train "
        "these models (920 patient records from four clinical sites)."
    )

    eda_dir = os.path.join(OUTPUTS_DIR, "eda")

    st.markdown("### Target Distribution")
    st.image(os.path.join(eda_dir, "target_distribution.png"), use_container_width=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.image(os.path.join(eda_dir, "age_distribution.png"), caption="Age Distribution",
                  use_container_width=True)
    with r1c2:
        st.image(os.path.join(eda_dir, "sex_vs_target.png"), caption="Heart Disease by Sex",
                  use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.image(os.path.join(eda_dir, "chest_pain_vs_target.png"),
                  caption="Chest Pain Type vs Diagnosis", use_container_width=True)
    with r2c2:
        st.image(os.path.join(eda_dir, "cholesterol_distribution.png"),
                  caption="Cholesterol by Diagnosis", use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.image(os.path.join(eda_dir, "max_heart_rate.png"),
                  caption="Maximum Heart Rate Distribution", use_container_width=True)
    with r3c2:
        st.image(os.path.join(eda_dir, "resting_bp.png"),
                  caption="Resting Blood Pressure by Diagnosis", use_container_width=True)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.image(os.path.join(eda_dir, "oldpeak_distribution.png"),
                  caption="Oldpeak by Diagnosis", use_container_width=True)
    with r4c2:
        st.image(os.path.join(eda_dir, "correlation_heatmap.png"),
                  caption="Correlation Heatmap", use_container_width=True)

    with st.expander("More categorical relationships"):
        r5c1, r5c2 = st.columns(2)
        with r5c1:
            st.image(os.path.join(eda_dir, "fbs_vs_target.png"), use_container_width=True)
            st.image(os.path.join(eda_dir, "exang_vs_target.png"), use_container_width=True)
        with r5c2:
            st.image(os.path.join(eda_dir, "slope_vs_target.png"), use_container_width=True)
            st.image(os.path.join(eda_dir, "thal_vs_target.png"), use_container_width=True)
        st.image(os.path.join(eda_dir, "restecg_vs_target.png"), use_container_width=True)
