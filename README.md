# 🍷 Wine Quality Prediction System

An interactive, machine-learning-powered web application built with Streamlit that predicts wine quality based on physicochemical properties. 

This project explores the *Wine Quality* dataset, replacing subjective sensory testing with an objective, data-driven approach using an Extreme Gradient Boosting (XGBoost) architecture.

## 🌟 Features

- **Interactive Explorer:** Adjust 11 chemical properties (like Alcohol, pH, and Sulphates) and the Wine Type (Red/White) via intuitive sliders to simulate a wine profile and instantly calculate its predicted quality score.
- **Dark Modern UI:** A fully styled, responsive, and professional dashboard interface.
- **Performance Validation:** Side-by-side metric comparison against a baseline Logistic Regression model.
- **Model Interpretability (Explainable AI):** Includes dataset diagnostics and SHAP (SHapley Additive exPlanations) summary plots to deconstruct the XGBoost ensemble, revealing exactly how each chemical property drives the final classification.

## 🚀 Live Demo
You can deploy this instantly on Streamlit Community Cloud:
1. Connect this repository to your Streamlit Cloud account.
2. Set the main file path to `app.py`.
3. Click "Deploy".

## 💻 Local Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. 

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pratham-Arun/Wine-Quality-Prediction-System.git
   cd Wine-Quality-Prediction-System
   ```

2. **Install dependencies:**
   *(Ensure you have `streamlit`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, and `shap` installed)*
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `requirements.txt` is missing, manually install the packages mentioned above.*

3. **Run the Application:**
   ```bash
   python -m streamlit run app.py
   ```

4. **Access the Web App:**
   Open your browser and navigate to `http://localhost:8501`.

## 🧠 Machine Learning Pipeline

- **Dataset:** Contains physicochemical tests of red and white variants of Portuguese "Vinho Verde" wine. 
- **Target Variable:** Binary classification mapping scores to either **Standard Quality (0)** or **High Quality (1)**.
- **Preprocessing:** Missing values imputed using the median strategy.
- **Model:** Primary architecture is `XGBClassifier` evaluated against `LogisticRegression` for standard baselining.

## 📂 Project Structure

- `app.py`: The core Streamlit application containing the UI logic, data pipeline, and model inference.
- `winequalityN.csv`: The dataset powering the models and EDA diagnostics.

---
*Developed by Pratham Arun*
