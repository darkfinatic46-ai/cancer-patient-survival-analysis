# Cancer Patient Survival Analysis & Outcome Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Project-Portfolio-success)

## 📌 Project Overview

This project analyzes cancer patient data from India to identify patterns in patient outcomes, cancer types, stages, treatments, survival duration, and geographical distribution.

The project combines **Exploratory Data Analysis (EDA)** with **machine learning classification** to compare different models for predicting patient status.

> **Note:** This is an educational and portfolio project for data-analytics purposes only. The predictions must not be used for clinical diagnosis, treatment decisions, or medical advice.

---

## 🎯 Objectives

- Understand the structure and quality of the cancer patient dataset.
- Analyze patient demographics and cancer characteristics.
- Explore the relationship between cancer stage, treatment, cancer type, and patient status.
- Analyze survival duration across patient groups.
- Detect potential outliers using the IQR method.
- Examine correlations between numerical variables.
- Analyze the geographical distribution of patients across Indian states.
- Compare multiple machine learning classification models.
- Tune a Random Forest model using `RandomizedSearchCV`.
- Evaluate model performance using accuracy, classification reports, and a confusion matrix.

---

## 📊 Dataset

The dataset contains **100,000 patient records and 12 columns**.

### Dataset columns

| Column | Description |
|---|---|
| `Patient_ID` | Unique patient identifier |
| `Age` | Patient age |
| `Gender` | Patient gender |
| `State` | Indian state |
| `City` | Patient city |
| `Hospital_Name` | Hospital associated with the record |
| `Cancer_Type` | Type of cancer |
| `Stage` | Cancer stage |
| `Treatment_Type` | Treatment category |
| `Diagnosis_Date` | Date of diagnosis |
| `Survival_Months` | Recorded survival duration in months |
| `Status` | Patient outcome/status |

---

## 🛠️ Technologies & Tools

### Data Analysis
- Python
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### Other Techniques
- Data cleaning
- Exploratory Data Analysis (EDA)
- Missing-value analysis
- Duplicate detection
- IQR-based outlier analysis
- Correlation analysis
- Feature preprocessing
- One-hot encoding
- Standardization
- Model evaluation
- Hyperparameter tuning

---

## 🔍 Project Workflow

```text
Data Loading
     ↓
Data Quality Check
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Survival Analysis
     ↓
Outlier Analysis
     ↓
Correlation Analysis
     ↓
Geographical Analysis
     ↓
Feature Preparation
     ↓
Train/Test Split
     ↓
Model Comparison
     ↓
Random Forest Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Export Results
```

---

## 📈 Exploratory Data Analysis

The project investigates:

- Patient status distribution
- Age distribution
- Cancer type distribution
- Cancer stage distribution
- Treatment type distribution
- Gender vs. patient status
- Cancer type vs. patient status
- Cancer stage vs. patient status
- Treatment type vs. patient status
- Average and median survival duration
- Average survival by cancer type
- Age and survival-month outliers
- Numerical feature correlations
- Patient distribution by state

---

## 🤖 Machine Learning

The target variable is:

```text
Status
```

The predictive features include:

```text
Age
Gender
State
City
Cancer_Type
Stage
Treatment_Type
```

`Survival_Months` is intentionally excluded from the predictive model because it may contain information that becomes available after or alongside the patient outcome, which could introduce **target leakage**.

### Models Compared

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Support Vector Machine

The models are evaluated using:

- Accuracy
- Classification report
- Test-set predictions

---

## 🌲 Random Forest Tuning

After comparing the baseline models, Random Forest is tuned using:

```python
RandomizedSearchCV
```

The tuning process explores:

- Number of estimators
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples required at a leaf

The tuned model is evaluated using cross-validation and a held-out test set.

---

## 📁 Project Structure

```text
cancer-patient-survival-analysis/
│
├── cancer_patient_analysis.py
├── india_cancer_patients_2022_2025 (2)(1).csv
├── model_comparison.csv
├── survival_by_status.csv
└── README.md
```

> The exported CSV files are generated after running the Python analysis script.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/cancer-patient-survival-analysis.git
cd cancer-patient-survival-analysis
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Keep the dataset in the project folder

Make sure the CSV filename matches the path used in the Python script:

```text
india_cancer_patients_2022_2025 (2).csv
```

### 4. Run the analysis

```bash
python cancer_patient_analysis.py
```

The script generates analysis outputs including:

```text
model_comparison.csv
survival_by_status.csv
```

---

## 📌 Key Portfolio Skills Demonstrated

This project demonstrates practical experience with:

- Python-based data analysis
- Pandas and NumPy
- Data cleaning and quality checks
- Exploratory Data Analysis
- Data visualization
- Statistical analysis
- Outlier detection
- Feature preprocessing
- Classification algorithms
- Model comparison
- Hyperparameter tuning
- Model evaluation
- Data-driven interpretation

---

## 👤 Author

**Inderjeet Singh**

Junior Data Analyst | Data Analyst

**Skills:** Python • SQL • Excel • Power BI • Tableau • Pandas • NumPy • Data Visualization • Machine Learning

---

## ⚠️ Disclaimer

This project is intended strictly for **educational and portfolio purposes**.

It is not a medical research study and should not be used to diagnose patients, recommend treatments, or make clinical decisions.
