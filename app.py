import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

# --- 1. System Configuration ---
st.set_page_config(page_title="Wine Quality Prediction System", page_icon="🍷", layout="wide")

# Custom CSS for a Premium "Wine Intelligence" look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glassmorphism Cards */
    .stMetric, .css-1r6slb0, .e1tzpsn23 {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }

    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    .main-title {
        font-size: 3rem;
        background: linear-gradient(to right, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .sub-title {
        color: #94a3b8 !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        background: rgba(255, 255, 255, 0.05);
        color: #f8fafc;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }

    /* Sidebar Active State Indicator (Simulated) */
    .nav-active {
        background: linear-gradient(to right, #3b82f6, #8b5cf6) !important;
        color: white !important;
    }

    /* Info/Success Boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
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
    # Basic cleanup: remove duplicates
    df = df.drop_duplicates()
    
    # Feature Selection: Removing 'total sulfur dioxide' due to high correlation with 'free sulfur dioxide'
    if 'total sulfur dioxide' in df.columns:
        df = df.drop('total sulfur dioxide', axis=1)
    
    features = df.drop(['quality', 'best quality'], axis=1, errors='ignore')
    target = df['best quality'] if 'best quality' in df.columns else (df['quality'] > 5).astype(int)
        
    return df, features, target

df, features, target = load_data()

# --- 3. Model Engine ---
@st.cache_resource
def train_models(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=40)
    
    imputer = SimpleImputer(strategy='mean') # Switched to mean as per latest documentation
    xtrain_imp = imputer.fit_transform(xtrain)
    xtest_imp = imputer.transform(xtest)
    
    # Normalization using MinMaxScaler
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain_imp)
    xtest_scaled = scaler.transform(xtest_imp)
    
    # Standard linear baseline
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(xtrain_scaled, ytrain)
    
    # Advanced ensemble
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(xtrain_scaled, ytrain)
    
    # Support Vector Machine
    svc_model = SVC(kernel='rbf', probability=True)
    svc_model.fit(xtrain_scaled, ytrain)
    
    return lr_model, xgb_model, svc_model, xtest_scaled, ytest, xtrain_scaled, imputer, scaler

lr_model, xgb_model, svc_model, xtest_imp, ytest, xtrain_imp, imputer, scaler = train_models(features, target)

# --- SIDEBAR NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #60a5fa;'>Wine Quality Prediction System</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation Buttons
    if st.button("🏠 Home", use_container_width=True): st.session_state.page = "Home"
    if st.button("📖 About", use_container_width=True): st.session_state.page = "About"
    if st.button("🧠 Model", use_container_width=True): st.session_state.page = "Model"
    if st.button("📈 Results", use_container_width=True): st.session_state.page = "Results"
    
    st.markdown("---")

page = st.session_state.page

# --- 4. PAGE ROUTING ---
if page == "Home":
    st.markdown("<h1 class='main-title'>Wine Quality Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Predictive Analytics for Wine Quality</p>", unsafe_allow_html=True)
    
    st.markdown("""
        ### Elevating Wine Assessment through Data Science
        Traditional wine evaluation relies on subjective sensory testing. Our system transforms this process into an **objective, data-driven methodology**.
        
        By analyzing the intricate physicochemical properties of wine, we provide a high-precision classification engine that distinguishes between standard and exceptional vintages.
    """)
    
    # Dataset Overview
    st.markdown("#### 📊 Dataset Statistics")
    c1, c2 = st.columns(2)
    c1.metric("Total Samples", "5,329")
    c2.metric("Chemical Features", "11")
    
    st.info("💡 **Pro Tip:** Start with the 'Results' page to see how the model understands the dataset.")

elif page == "About":
    st.markdown("<h1 class='main-title'>Wine Quality Prediction System: The Project</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Bridging the gap between chemistry and quality</p>", unsafe_allow_html=True)
    
    st.markdown("""
        ### Project Overview
        Predicting the quality of wine based on its physicochemical properties using open-source datasets. 
        This system replaces subjective expert tasting with objective, scalable Machine Learning.
    """)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6;'>
            <h4>The Problem Statement</h4>
            <p>Manual wine evaluation is <b>subjective</b>, <b>expensive</b>, and <b>non-scalable</b>. Human perception varies, but chemical data remains constant.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(" ")
        
        st.markdown("""
        <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border-left: 4px solid #8b5cf6;'>
            <h4>Computational Solution</h4>
            <p>Using <b>eXtreme Gradient Boosting (XGBoost)</b> and <b>Support Vector Machines (SVM)</b> to achieve high-precision classification DNA.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_a2:
        st.markdown("### Libraries & Frameworks")
        st.info("**Pandas & NumPy:** Data handling and array structures.\n\n"
                "**Seaborn & Matplotlib:** Advanced data visualization.\n\n"
                "**Scikit-Learn:** Preprocessing, model development, and validation.\n\n"
                "**XGBoost:** High-accuracy gradient boosting ensemble.")

    st.markdown("---")
    st.markdown("### 🍷 Understanding the Chemistry")
    st.write("The system analyzes 11 key physicochemical parameters to determine wine quality:")
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        st.markdown("""
        **1. Fixed Acidity**  
        Non-volatile acids (primarily tartaric) that provide the foundational tartness and structural backbone.
        
        **2. Volatile Acidity**  
        A measure of acetic acid; critical for aroma at low levels but leads to a vinegar-like defect if high.
        
        **3. Citric Acid**  
        Acts as a natural preservative and enhances the wine's "freshness" and flavor profile.
        
        **4. Residual Sugar**  
        The sugar remaining after fermentation; essential for balancing acidity and body.
        
        **5. Chlorides**  
        The amount of salt in the wine, influencing the perceived "savouriness" and mouthfeel.
        
        **6. Alcohol Content**  
        The percentage of alcohol by volume, directly impacting the wine's body and warmth.
        """)
        
    with param_col2:
        st.markdown("""
        **7. Free Sulfur Dioxide**  
        The active form of SO₂ that protects the wine from oxidation and microbial spoilage.
        
        **8. Density**  
        Closely tied to alcohol and sugar levels; reflects the overall viscosity of the wine.
        
        **9. pH Level**  
        The scale of acidity (0-14); influences the wine's color, stability, and sensory balance.
        
        **10. Sulphates**  
        A wine additive (Potassium Sulphate) that acts as an antioxidant and antimicrobial agent.
        
        **11. Wine Variant (Type)**  
        The categorical distinction between Red and White variants, which significantly influences the chemical baseline.
        """)

elif page == "Model":
    st.markdown("<h1 class='main-title'>Intelligence Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Decoding the Wine Quality Classifier</p>", unsafe_allow_html=True)
    
    st.markdown("### 🛠 Processing Pipeline")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("#### 1. Preprocessing")
        st.write("Missing values are imputed using the **Mean** strategy, as physicochemical features are continuous.")
    
    with col_p2:
        st.markdown("#### 2. Normalization")
        st.write("Using **MinMaxScaler** to scale features between 0 and 1, ensuring stable and fast model convergence.")
        
    with col_p3:
        st.markdown("#### 3. Feature Selection")
        st.write("Redundant features like **Total Sulfur Dioxide** are removed to reduce collinearity and improve metrics.")

    st.markdown("---")
    
    st.markdown("### 📊 Ensemble Architecture")
    st.markdown("""
    The system benchmarks three state-of-the-art architectures:
    
    1. **Logistic Regression:** Linear baseline for binary classification.
    2. **XGBoost:** Highly efficient gradient boosting for tabular data.
    3. **SVC (Support Vector Classifier):** Effective in high-dimensional spaces using the RBF kernel.
    """)
    
    st.success("**Performance Note:** XGBoost typically provides the highest validation accuracy (~80%) for this dataset.")

elif page == "Results":
    st.markdown("<h1 class='main-title'>Intelligence Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Performance metrics and dataset forensics</p>", unsafe_allow_html=True)

    st.header("1. Dataset Diagnostics")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.subheader("Target Variable Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        fig_dist.patch.set_facecolor('#0f172a')
        ax_dist.set_facecolor('#0f172a')
        sb.countplot(x=target, palette="viridis", ax=ax_dist)
        ax_dist.set_xticklabels(['Standard (0)', 'High Quality (1)'], color='white')
        ax_dist.set_xlabel("")
        ax_dist.tick_params(colors='white')
        for spine in ax_dist.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig_dist)
        
    with col_e2:
        st.subheader("Chemical Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        fig_corr.patch.set_facecolor('#0f172a')
        numeric_df = df.select_dtypes(include=['float', 'int'])
        sb.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", linewidths=.5, ax=ax_corr)
        ax_corr.tick_params(colors='white')
        st.pyplot(fig_corr)

    st.markdown("---")
    st.header("2. Comparative Model Performance")
    
    # Calculate predictions for all models
    models_list = [
        ("Logistic Regression", lr_model),
        ("XGBoost", xgb_model),
        ("Support Vector Machine", svc_model)
    ]
    
    for name, model in models_list:
        with st.expander(f"📊 {name} Performance Details", expanded=(name=="XGBoost")):
            preds = model.predict(xtest_imp)
            probs = model.predict_proba(xtest_imp)[:, 1]
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics.accuracy_score(ytest, preds) * 100:.1f}%")
            m2.metric("Precision", f"{metrics.precision_score(ytest, preds) * 100:.1f}%")
            m3.metric("Recall", f"{metrics.recall_score(ytest, preds) * 100:.1f}%")
            m4.metric("ROC-AUC", f"{metrics.roc_auc_score(ytest, probs):.3f}")
            
            c_cm1, c_cm2 = st.columns(2)
            with c_cm1:
                fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
                fig_cm.patch.set_facecolor('#0f172a')
                cm = metrics.confusion_matrix(ytest, preds)
                sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                           xticklabels=['Std', 'High'], yticklabels=['Std', 'High'])
                ax_cm.tick_params(colors='white')
                plt.ylabel('Actual', color='white')
                plt.xlabel('Predicted', color='white')
                st.pyplot(fig_cm)
            
            with c_cm2:
                fig_roc, ax_roc = plt.subplots(figsize=(5, 3))
                fig_roc.patch.set_facecolor('#0f172a')
                ax_roc.set_facecolor('#0f172a')
                fpr, tpr, _ = metrics.roc_curve(ytest, probs)
                ax_roc.plot(fpr, tpr, color='#3b82f6', lw=2, label=f'ROC {name}')
                ax_roc.plot([0, 1], [0, 1], color='#94a3b8', lw=1, linestyle='--')
                ax_roc.tick_params(colors='white')
                ax_roc.legend()
                st.pyplot(fig_roc)

    st.markdown("---")
    st.header("3. Explainable AI (SHAP - XGBoost)")
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(xtest_imp)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Top Feature Importances")
        importances = pd.Series(xgb_model.feature_importances_, index=features.columns)
        importances = importances.sort_values(ascending=False).head(10)
        st.bar_chart(importances, color="#8b5cf6")
        
    with col_s2:
        st.subheader("Global Impact Analysis")
        fig_shap, ax_shap = plt.subplots(figsize=(6, 4))
        fig_shap.patch.set_facecolor('#0f172a')
        shap.summary_plot(shap_values, xtest_imp, feature_names=features.columns, show=False)
        plt.gcf().axes[-1].tick_params(colors='white') 
        plt.gca().tick_params(colors='white')
        st.pyplot(fig_shap)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8;'>Wine Quality Prediction System</p>", unsafe_allow_html=True)