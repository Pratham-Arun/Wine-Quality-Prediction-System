import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# --- 1. System Configuration ---
st.set_page_config(page_title="Wine Quality Research Dashboard", page_icon="🍷", layout="wide")

st.markdown("""
    <style>
    /* Dark Modern UI Theme */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f1a;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    h1, h2, h3, h4, p, span, div {
        color: white !important;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: #0b0f1a !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0ea5e9;
        color: white !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #38bdf8 !important;
    }
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        background-color: transparent !important;
    }
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. Data Pipeline ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'winequalityN.csv')
    df = pd.read_csv(file_path)
    
    df['type'] = df['type'].map({'white': 1, 'red': 0})
    features = df.drop(['quality', 'best quality'], axis=1, errors='ignore')
    target = df['best quality'] if 'best quality' in df.columns else (df['quality'] > 5).astype(int)
        
    return df, features, target

df, features, target = load_data()

# --- 3. Model Engine ---
@st.cache_resource
def train_models(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=40)
    
    imputer = SimpleImputer(strategy='median')
    xtrain_imp = imputer.fit_transform(xtrain)
    xtest_imp = imputer.transform(xtest)
    
    # Standard linear baseline
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(xtrain_imp, ytrain)
    
    # Advanced ensemble
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(xtrain_imp, ytrain)
    
    return lr_model, xgb_model, xtest_imp, ytest, xtrain_imp, imputer

lr_model, xgb_model, xtest_imp, ytest, xtrain_imp, imputer = train_models(features, target)

# --- SIDEBAR NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3014/3014502.png", width=50) # Wine icon
    st.title("🍷 Wine Quality")
    st.markdown("---")
    
    st.markdown("### Navigation")
    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"
    if st.button("About", use_container_width=True):
        st.session_state.page = "About"
    if st.button("Model", use_container_width=True):
        st.session_state.page = "Model"
    if st.button("Results", use_container_width=True):
        st.session_state.page = "Results"
    if st.button("Explorer", use_container_width=True):
        st.session_state.page = "Explorer"
        
    page = st.session_state.page
    
    st.markdown("---")
    st.markdown("### System KPI")
    st.metric("Dataset Size", f"{len(df):,}")
    st.metric("Features", f"{len(features.columns)}")

# --- PAGE ROUTING ---

if page == "Home":
    st.title("Wine Quality Prediction System")
    st.markdown("### Machine Learning Framework for Wine Classification")
    st.markdown("---")
    st.markdown("""
        Welcome to the **Wine Quality Prediction System**. 
        
        This platform leverages advanced machine learning algorithms to predict the quality of wine based on its physicochemical properties.
        Transitioning from traditional sensory evaluation, this system provides an objective, data-driven approach to wine assessment.
    """)
    st.info("👈 Use the sidebar navigation to explore the dataset, understand the model, view performance metrics, or try the interactive Explorer.")

elif page == "About":
    st.title("About the Project")
    st.markdown("---")
    st.markdown("""
    ### Problem Context
    Determining wine quality has traditionally relied heavily on sensory testing by human experts. However, this process is subjective, time-consuming, and expensive. 
    Can we use the objective chemical properties of wine to predict its quality score?

    ### Industry Importance
    Predictive modeling for wine quality offers significant benefits:
    - **Quality Control:** Automated screening during the production process.
    - **Cost Reduction:** Minimizing the reliance on extensive sensory panels.
    - **Targeted Winemaking:** Understanding which chemical factors most strongly influence high quality ratings.

    ### Dataset Overview
    The engine uses the *Wine Quality* dataset, encompassing both red and white variants. The target variable is binary: 
    - **High Quality (1)** 
    - **Standard Quality (0)**
    """)

elif page == "Model":
    st.title("Machine Learning Pipeline")
    st.markdown("---")
    
    st.markdown("### Architectures Deployed")
    st.markdown("""
    1. **Logistic Regression:** Serves as the standard linear baseline model.
    2. **Extreme Gradient Boosting (XGBoost):** An advanced, non-linear ensemble architecture that typically yields higher predictive performance on tabular data.
    """)
    
    st.markdown("### Data Preprocessing")
    st.markdown("""
    - **Missing Value Handling:** Missing data points are imputed using the `median` strategy to maintain robust central tendencies without being skewed by outliers.
    - **Feature Selection:** Target variables (`quality` and `best quality`) are isolated, while all remaining physicochemical attributes are utilized as predictive features.
    """)
    
    st.markdown("### Evaluation Metrics")
    st.markdown("""
    - **Accuracy:** The overall proportion of correct predictions.
    - **Precision:** The proportion of predicted 'High Quality' wines that were actually high quality (minimizing false positives).
    - **Recall:** The proportion of actual 'High Quality' wines that the model successfully identified (minimizing false negatives).
    - **ROC-AUC:** A comprehensive metric charting the model's ability to distinguish between classes across different thresholds.
    """)

elif page == "Results":
    st.title("Performance Validation")
    st.markdown("Comparing the baseline Logistic Regression against the Gradient Boosting (XGBoost) architecture.")
    st.markdown("---")

    st.header("Dataset Diagnostics")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.subheader("Target Variable Imbalance")
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        sb.countplot(x=target, palette="mako", ax=ax_dist)
        ax_dist.set_xticklabels(['Standard Quality (0)', 'High Quality (1)'])
        ax_dist.set_xlabel("")
        st.pyplot(fig_dist)
        
    with col_e2:
        st.subheader("Chemical Collinearity Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        sb.heatmap(df.corr(numeric_only=True), annot=False, cmap="vlag", linewidths=.5, ax=ax_corr)
        st.pyplot(fig_corr)

# --- TAB 2: METRICS ---
    st.header("Performance Validation")
    st.markdown("Comparing the baseline Logistic Regression against the Gradient Boosting (XGBoost) architecture.")
    
    xgb_preds = xgb_model.predict(xtest_imp)
    xgb_probs = xgb_model.predict_proba(xtest_imp)[:, 1]
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("XGB Accuracy", f"{metrics.accuracy_score(ytest, xgb_preds) * 100:.1f}%")
    col_m2.metric("XGB Precision", f"{metrics.precision_score(ytest, xgb_preds) * 100:.1f}%")
    col_m3.metric("XGB Recall", f"{metrics.recall_score(ytest, xgb_preds) * 100:.1f}%")
    col_m4.metric("ROC-AUC Score", f"{metrics.roc_auc_score(ytest, xgb_probs):.3f}")
    
    st.markdown("---")
    
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.subheader("Confusion Matrix (XGBoost)")
        st.markdown("Visualizing True Positives, False Positives, True Negatives, and False Negatives.")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        cm = metrics.confusion_matrix(ytest, xgb_preds)
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                   xticklabels=['Standard', 'High'], yticklabels=['Standard', 'High'])
        plt.ylabel('Actual Quality')
        plt.xlabel('Predicted Quality')
        st.pyplot(fig_cm)
        

    with col_cm2:
        st.subheader("Receiver Operating Characteristic (ROC)")
        st.markdown("Mapping the diagnostic ability of the binary classifier as its discrimination threshold varies.")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        fpr, tpr, _ = metrics.roc_curve(ytest, xgb_probs)
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='XGBoost')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

# --- TAB 3: EXPLAINABLE AI --- (Moving to Results page for now)
    st.markdown("---")
    st.header("Model Interpretability")

    st.markdown("Using SHapley Additive exPlanations (SHAP) to deconstruct the black-box nature of the XGBoost ensemble. This reveals exactly how each chemical property drives the final quality classification.")
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(xtest_imp)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Feature Importance Profile")
        st.markdown("Relative impact of features based on the XGBoost model's learned weights.")
        importances = pd.Series(xgb_model.feature_importances_, index=features.columns)
        importances = importances.sort_values(ascending=False).head(10)
        st.bar_chart(importances)
        
    with col_s2:
        st.subheader("Global Feature Impact (SHAP)")
        fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
        shap.summary_plot(shap_values, xtest_imp, feature_names=features.columns, show=False)
        st.pyplot(fig_shap)
        
    st.markdown("### Research Insights")
    st.info("**How to read the SHAP plot:**\n\n* **Vertical Axis:** Features ordered by their impact on the model's prediction.\n* **Horizontal Axis:** The SHAP value. Points to the right push the prediction toward 'High Quality', points to the left push toward 'Standard'.\n* **Color:** The actual value of the feature (Red = High value, Blue = Low value).")
    
    st.success("**Key Findings:**\n\nAlcohol is typically the strongest driver of high quality, while high volatile acidity damages the wine's quality score.")

elif page == "Explorer":
    st.title("Interactive Prediction Explorer")
    st.markdown("Adjust the physicochemical properties below to simulate a wine profile and predict its quality.")
    st.markdown("---")
    
    wine_type = st.radio("Wine Type", options=["White", "Red"], horizontal=True)
    type_val = 1 if wine_type == "White" else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 3.8, 15.9, 7.2)
        residual_sugar = st.slider("Residual Sugar", 0.6, 65.8, 5.4)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 440.0, 115.7)
        sulphates = st.slider("Sulphates", 0.22, 2.0, 0.53)
        
    with col2:
        volatile_acidity = st.slider("Volatile Acidity", 0.08, 1.58, 0.34)
        chlorides = st.slider("Chlorides", 0.009, 0.611, 0.056)
        density = st.slider("Density", 0.987, 1.039, 0.995)
        alcohol = st.slider("Alcohol", 8.0, 14.9, 10.5)
        
    with col3:
        citric_acid = st.slider("Citric Acid", 0.0, 1.66, 0.32)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 289.0, 30.5)
        pH = st.slider("pH", 2.72, 4.01, 3.22)
        
    st.markdown("---")
    st.markdown("### Model Prediction")
    
    if st.button("Calculate Quality Score", use_container_width=True):
        with st.spinner("Analyzing physicochemical profile..."):
            # Prediction pipeline
            input_data = pd.DataFrame([[
                type_val, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                pH, sulphates, alcohol
            ]], columns=[
                'type', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol'
            ])
            
            # Apply imputer exactly as during training
            input_imp = imputer.transform(input_data)
            
            # Predict
            pred = xgb_model.predict(input_imp)[0]
            prob = xgb_model.predict_proba(input_imp)[0][1]
            st.markdown("---")
            if pred == 1:
                st.success("🌟 **High Quality Wine**")
            else:
                st.error("📉 **Standard Quality**")
                
            st.metric("Confidence Score", f"{prob * 100:.1f}%")