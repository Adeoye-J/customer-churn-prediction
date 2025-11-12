# Customer Churn Prediction: A Comprehensive Scikit-Learn Pipeline

## 🌟 Overview

This project provides an end-to-end Machine Learning solution for predicting **customer churn** in a telecommunications-like environment. Utilizing **Python** and the **Scikit-Learn** ecosystem, it focuses on building a robust, maintainable, and highly reproducible predictive pipeline. The core of the project involves data cleaning, advanced feature preprocessing (using `ColumnTransformer`), model training with established algorithms (Logistic Regression and Random Forest), rigorous evaluation, and saving the final, optimized model.

This project serves as a strong foundation for moving a predictive model from experimentation into a production environment.

***

## 🎯 Project Goals

1.  **Develop a Robust Pipeline:** Implement a full Scikit-Learn `Pipeline` using `ColumnTransformer` to handle both categorical and numerical features efficiently and prevent data leakage.
2.  **Establish Baseline Performance:** Train and evaluate standard classification models, **Logistic Regression** and **Random Forest**, to set performance benchmarks.
3.  **Optimize Model Performance:** Use techniques like **GridSearchCV** or **RandomizedSearchCV** for hyperparameter tuning to maximize the model's predictive power.
4.  **Deliver a Production-Ready Asset:** Serialize and save the best-performing model (including the preprocessing steps) as a `.pkl` file for immediate deployment.

***

## 🛠️ Technical Stack

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python (3.x) | Core programming language |
| **Core ML** | Scikit-Learn | Pipelines, Models (`LogisticRegression`, `RandomForestClassifier`), Evaluation, Tuning |
| **Data Handling** | Pandas, NumPy | Data manipulation and numerical operations |
| **Visualization** | Matplotlib, Seaborn | Plotting results, ROC curves, and feature importance |
| **Environment** | Jupyter Notebook / Google Colab | Interactive development and documentation |
| **Persistence** | Pickle | Serializing and saving the trained ML model |

***

## 📊 Dataset Description

The project uses the **`churn_data.csv`** file, which is a synthetic dataset commonly used for telco churn prediction.

| Feature Type | Description |
| :--- | :--- |
| **Target Variable** | `churn` (Binary: 1 for churn, 0 for retained) |
| **Key Demographic** | `gender`, `senior_citizen`, `partner`, `dependents` |
| **Service Data** | `phone_service`, `multiple_lines`, `internet_service`, `online_security`, `tech_support`, `streaming_tv`, etc. |
| **Billing Data** | `tenure`, `contract`, `paperless_billing`, `payment_method`, `monthly_charges`, `total_charges` |

***

## ⚙️ Methodology: The ML Pipeline

The entire process is encapsulated within a Scikit-Learn `Pipeline`, ensuring that all preprocessing steps are applied consistently during training, testing, and deployment.

### 1. Data Preprocessing

A **`ColumnTransformer`** is employed to handle different data types simultaneously:
* **Numerical Features:** Scaled using `StandardScaler` to normalize the data (e.g., `monthly_charges`, `tenure`).
* **Categorical Features:** Encoded using **`OneHotEncoder`** with `handle_unknown='ignore'` to convert non-numeric features into a format suitable for ML models.

### 2. Model Selection and Training

* **Baseline Model 1: Logistic Regression** - Excellent for interpretability and a solid, fast baseline.
* **Baseline Model 2: Random Forest Classifier** - A powerful ensemble method for non-linear relationships, typically offering higher predictive accuracy.

### 3. Hyperparameter Tuning

The pipeline is passed to **`GridSearchCV`** (or similar search strategy) to systematically explore the optimal combination of hyperparameters for the best-performing model, significantly enhancing performance metrics.

***

## 📁 Project Structure and Files

| File Name | MIME Type | Description |
| :--- | :--- | :--- |
| **`01_Customer_Churn_Prediction.ipynb`** | `application/vnd.jupyter` | The primary Jupyter Notebook containing all the code, analysis, visualizations, pipeline definition, model training, tuning, and evaluation. **This is the recommended starting point.** |
| **`churn_data.csv`** | `text/csv` | The raw, synthetic dataset used for training and evaluation. |
| **`best_churn_model.pkl`** | `application/octet-stream` | The serialized Python object containing the final, tuned Scikit-Learn `Pipeline` and the best-trained model (e.g., the tuned Random Forest Classifier). This file is ready for deployment. |

***

## 🚀 How to Run the Project

The most straightforward way to execute the project is by using Google Colab or a local Jupyter environment.

1.  **Open the Notebook:** Upload and open **`01_Customer_Churn_Prediction.ipynb`** in Google Colab or a local Jupyter server.
2.  **Upload Data:** Ensure **`churn_data.csv`** and **`best_churn_model.pkl`** are in the same working directory or uploaded to the notebook environment.
3.  **Execute Cells:** Run the notebook cells sequentially from top to bottom.
4.  **Review Output:** View the data exploration plots, model training logs, evaluation metrics, ROC curves, and the final best model saved to disk.

***

## 📈 Results Summary

The models were evaluated using standard classification metrics, focusing on **Accuracy** and **ROC AUC** (Area Under the Receiver Operating Characteristic Curve), a key metric for imbalanced classification problems like churn.

| Model | Accuracy (Baseline) | ROC AUC (Baseline) |
| :--- | :--- | :--- |
| **Logistic Regression** | **≈ 0.72** | **≈ 0.75** |
| **Random Forest** | **≈ 0.71** | **≈ 0.74** |
| **Tuned Final Model** | *Significantly Improved* | *Targeting > 0.80* |

### Key Takeaways:
* The initial baseline models show reasonable, but moderate, predictive power.
* A major focus was on optimizing the **Random Forest** model, which, post-tuning, provided the highest performance, as saved in `best_churn_model.pkl`.

***

## ⏭️ Future Enhancements

The following steps are recommended for taking this project to the next level:

1.  **Handling Class Imbalance:** Implement techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or adjusting class weights to improve the model's ability to predict the minority class (Churn=1).
2.  **Model Interpretability:** Apply **SHAP (SHapley Additive exPlanations)** or **LIME** to explain individual customer predictions and provide global feature importance, which is crucial for business decision-making.
3.  **Deployment:** Deploy the `best_churn_model.pkl` file using a web framework like **Streamlit** or **Flask/FastAPI** to create a live-prediction API or interactive dashboard.
4.  **Advanced Modeling:** Explore other high-performance models such as **XGBoost** or **LightGBM** to potentially capture more complex patterns in the data.
