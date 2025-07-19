## Employee Attrition Analysis

This project explores employee attrition by analyzing HR data to understand the key features influencing whether an employee stays or leaves a company. It includes exploratory data analysis (EDA), feature engineering, and classification modeling to predict employee attrition.

---

## Problem Statement

Predict whether an employee is likely to leave the organization based on HR-related factors. The goal is to help HR teams proactively identify at-risk employees and reduce attrition rates.

---

## Tools & Technologies

- Python (Pandas, NumPy)
- Data Visualization: Matplotlib, Seaborn
- Classification Models: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN)
- Google Colab (for code execution)

---

## Dataset

- **Source**: [Kaggle – HR Analytics and Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction)
- **Records**: 14,999
- **Features**:
  - Satisfaction Level
  - Last Evaluation
  - Number of Projects
  - Average Monthly Hours
  - Time Spent at Company
  - Work Accident
  - Promotion in Last 5 Years
  - Department
  - Salary
- **Target Variable**: `Target` (1 = Left, 0 = Stayed)

*Note: The original target column `left` was renamed to `Target` for clarity.*

---

## Project Workflow

1. **Data Loading & Cleaning**
   - Checked for null values and data types
   - Identified numerical vs categorical features

2. **Exploratory Data Analysis**
   - Univariate & bivariate analysis
   - Visualizations: bar plots, box plots, correlation matrix

3. **Feature Engineering**
   - Converted categorical columns to numerical using One-Hot Encoding
   - Checked for class imbalance

4. **Model Building**
   - Trained 4 classification models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - K-Nearest Neighbors (KNN)
   - Compared performance using accuracy, precision, recall, F1-score

5. **Results & Insights**
   - Identified important predictors: satisfaction level, time at company, and salary
   - Random Forest achieved the best balance of accuracy and generalization

---

## Evaluation Metrics
Four classification models were trained and evaluated. Random Forest achieved the highest accuracy and generalization, followed by Decision Tree and KNN.

### 1. Logistic Regression
- **Train Accuracy**: 79.23%
- **Test Accuracy**: 78.51%
- **Overfitting Check**: 0.71%
- **AUC Score**: 0.81
<pre> Confusion Matrix:
[[3174  254]
 [ 713  359]] </pre>


### Classification Report
| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| 0 (Stayed)  | 0.82      | 0.93   | 0.87     | 3428    |
| 1 (Left)    | 0.59      | 0.33   | 0.43     | 1072    |
| **Accuracy**|           |        | **0.79** | 4500    |
| **Macro avg** | 0.70    | 0.63   | 0.65     | 4500    |
| **Weighted avg** | 0.76 | 0.79   | 0.76     | 4500    |

### 2. Decision Tree (Max Depth = 10)
- **Train Accuracy**: 98.38%
- **Test Accuracy**: 97.71%
- **Overfitting Check**: 1.13%
- **AUC Score**: 0.97
<pre> Confusion Matrix:
[[3391  37]
 [ 66  1006]] </pre>
### Classification Report
| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0 (Stayed)       | 0.98      | 0.99   | 0.99     | 3428    |
| 1 (Left)         | 0.96      | 0.94   | 0.95     | 1072    |
| **Accuracy**     |           |        | **0.98** | 4500    |
| **Macro avg**    | 0.97      | 0.96   | 0.97     | 4500    |
| **Weighted avg** | 0.98      | 0.98   | 0.98     | 4500    |


### 3. Random Forest (Max Depth = 10, Estimators = 200)
- **Train Accuracy**: 98.33%
- **Test Accuracy**: 97.57%
- **Overfitting Check**: 0.75%
- **AUC Score**: 0.99 
<pre> Confusion Matrix:
[[3415  13]
 [ 96  976]] </pre>

### Classification Report
| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0 (Stayed)       | 0.97      | 1.00   | 0.98     | 3428    |
| 1 (Left)         | 0.99      | 0.91   | 0.95     | 1072    |
| **Accuracy**     |           |        | **0.98** | 4500    |
| **Macro avg**    | 0.98      | 0.95   | 0.97     | 4500    |
| **Weighted avg** | 0.98      | 0.98   | 0.98     | 4500    |


### 4. K-Nearest Neighbors (K = 5)
- **Train Accuracy**: 96.03%
- **Test Accuracy**: 93.29%
- **Overfitting Check**: 2.73%
- **AUC Score**: 0.96
<pre> Confusion Matrix:
[[3256  172]
 [ 130  942]] </pre>

### Classification Report:
| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0 (Stayed)       | 0.96      | 0.95   | 0.96     | 3428    |
| 1 (Left)         | 0.85      | 0.88   | 0.86     | 1072    |
| **Accuracy**     |           |        | **0.93** | 4500    |
| **Macro avg**    | 0.90      | 0.91   | 0.91     | 4500    |
| **Weighted avg** | 0.93      | 0.93   | 0.93     | 4500    |

---

## ROC Curve Comparison

All models were evaluated using the ROC curve for binary classification. Below is the AUC (Area Under the Curve) comparison:

- **Logistic Regression**: AUC = 0.81  
- **Decision Tree**: AUC = 0.97  
- **Random Forest**: AUC = 0.99 
- **KNN**: AUC = 0.96

---

## Summary

- **Random Forest** achieved the **highest recall and AUC**, making it the most reliable model.
- **KNN and Decision Tree** performed comparably well.
- **Logistic Regression** underperformed, likely due to class imbalance or linear limitations.

---

## Team Members:

- Keyur Parkhi  
- Gourish Salgaonkar  
- Dev Vatnani  

