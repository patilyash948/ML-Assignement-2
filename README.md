# Machine Learning Assignment 2  
M.Tech (AIML/DSE) – Work Integrated Learning Programmes Division

## a. Problem statement

The objective of this assignment is to implement and compare six classification algorithms on a selected public dataset, evaluate their performance using six standard metrics (Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient), and deploy an interactive Streamlit web application.  

The application allows users to:
- Upload test data in CSV format
- Select one of the trained models via dropdown
- View predictions
- See evaluation metrics (when true labels are provided)
- Display confusion matrix and classification report

This demonstrates a complete end-to-end machine learning workflow: data preparation, model training, evaluation, web interface creation, and cloud deployment.

## b. Dataset description

**Dataset name:** Bank Marketing (full dataset – bank-full.csv)  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing  
**Task:** Binary classification – predict whether a client will subscribe a term deposit (target: y = "yes" / "no")  
**Number of instances:** 45,211  
**Number of features:** 16 input features (7 numerical + 9 categorical)  
**Target variable:** y (highly imbalanced: ~88.3% "no", ~11.7% "yes")  
**Preprocessing applied:**  
- Categorical variables → one-hot encoded (drop_first=True)  
- Numerical variables → standardized using StandardScaler  
- Train/test split: 75/25, stratified to preserve class distribution  

This dataset meets the assignment requirements (>> 500 instances, >> 12 features) and is suitable for comparing linear, tree-based, distance-based, probabilistic, and ensemble methods.

## c. Models used

The following six classification models were implemented and evaluated on the same dataset:

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9016   | 0.9054 | 0.6474    | 0.3488 | 0.4533 | 0.4280 |
| Decision Tree          | 0.8777   | 0.7135 | 0.4783    | 0.4991 | 0.4884 | 0.4191 |
| kNN                    | 0.8961   | 0.8373 | 0.5931    | 0.3554 | 0.4444 | 0.4067 |
| Naive Bayes            | 0.8639   | 0.8088 | 0.4282    | 0.4877 | 0.4560 | 0.3797 |
| Random Forest (Ensemble) | 0.9045 | 0.9272 | 0.6554    | 0.3866 | 0.4863 | 0.4561 |
| XGBoost (Ensemble)     | 0.9080   | 0.9291 | 0.6348    | 0.5028 | 0.5612 | 0.5149 |

**Note:** All metrics were computed on the test set (25% hold-out). Values are rounded to 4 decimal places.

### Observations about model performance

| ML Model Name            | Observation about model performance                                                                 |
|--------------------------|-----------------------------------------------------------------------------------------------------|
| Logistic Regression      | Strong linear baseline with high accuracy and AUC. However, low recall due to class imbalance — tends to predict majority class ("no") more often. |
| Decision Tree            | Shows signs of overfitting (lowest AUC). Balanced precision/recall trade-off but overall weaker generalization compared to ensembles. |
| kNN                      | Reasonable accuracy, but suffers from high dimensionality after one-hot encoding → lower recall and MCC than tree-based ensembles. |
| Naive Bayes              | Fast training and surprisingly decent recall despite strong independence assumption. Lowest precision — over-predicts positive class. |
| Random Forest (Ensemble) | Very robust with high AUC and accuracy. Excellent precision but lower recall than XGBoost on this imbalanced problem. |
| XGBoost (Ensemble)       | Best overall performance: highest AUC, F1 score, and MCC. Effectively handles class imbalance and complex feature interactions through gradient boosting. |

**Deployment information**  
Live Streamlit App: https://[your-app-name].streamlit.app  
(Replace with your actual deployed link after successful deployment on Streamlit Community Cloud)

**Repository contents**  
- `app.py` – Streamlit application  
- `requirements.txt` – required Python packages  
- `model/` – saved trained models (.pkl files)  
- `model_training.ipynb` – training and evaluation notebook (executed on BITS Virtual Lab)