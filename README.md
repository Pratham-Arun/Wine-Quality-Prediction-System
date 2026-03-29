# 🍷 Wine Quality Prediction System

An interactive, machine-learning-powered web application built with Streamlit that predicts wine quality based on physicochemical properties. 

This project explores the *Wine Quality* dataset, replacing subjective sensory testing with an objective, data-driven approach using an Extreme Gradient Boosting (XGBoost) architecture.

## 🌟 Features

- **Multi-Model Analytics:** Real-time performance benchmarking of **Logistic Regression**, **XGBoost**, and **Support Vector Machines (SVC)**.
- **Dark Modern UI:** A fully styled, responsive, and professional dashboard interface built for high-contrast data visualization.
- **Deep Insights:** Comprehensive dataset diagnostics, including correlation matrices and target distribution analysis.
- **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) to deconstruct model decisions and identify the key chemical properties driving wine quality.

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
   Open your browser and navigate to (https://wine-quality-prediction-system.streamlit.app/).

## 🧠 Machine Learning Pipeline

- **Dataset:** Contains physicochemical tests of red and white variants of "Vinho Verde" wine from northern Portugal.
- **Target Variable:** Binary classification (scores > 5 are mapped to **High Quality (1)**, others to **Standard (0)**).
- **Preprocessing:** Missing values imputed using the **Mean** strategy; features normalized using **MinMaxScaler**.
- **Model Ensemble:** A comparative architecture featuring **Logistic Regression**, **XGBoost (XGBClassifier)**, and **Support Vector Classifier (SVC)**.

## 📂 Project Structure

- `app.py`: The core Streamlit application containing the UI logic, data pipeline, and model inference.
- `winequalityN.csv`: The dataset powering the models and EDA diagnostics.

---
## 👥 Team Members

This project was brought to life by the following contributors:
- **Sparsh Mishra** - [GitHub Profile](https://github.com/maj0rkeen)
- **Ajeet Singh Rawat** - [GitHub Profile](https://github.com/ajeetsinghrawat)
- **Pratham Arun** - [GitHub Profile](https://github.com/Pratham-Arun)
