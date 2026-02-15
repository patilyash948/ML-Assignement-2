# Bank Marketing Subscription Prediction using Machine Learning

## a. Problem Statement
The objective of this project is to build and evaluate multiple machine learning models to classify whether a client will **subscribe to a term deposit** (yes/no) using the Bank Marketing dataset. The goal is to compare model performance using different evaluation metrics and identify the best performing model.

---

## b. Dataset Description
The dataset used in this project is the **Bank Marketing Dataset** (bank-full.csv) obtained from **UCI Machine Learning Repository**.

- Total Samples: 45,211
- Features: 16 input features (7 numerical + 9 categorical)
- Target Variable:
  - 0 → No
  - 1 → Yes
- Highly imbalanced (~88.3% No, ~11.7% Yes)
- No missing values present
- Preprocessing: One-hot encoding for categorical features, StandardScaler for numerical features, stratified train-test split (75/25)

---

## c. Models Used
Six machine learning models were trained and evaluated using the following metrics:  
**Accuracy, AUC, Precision, Recall, F1 Score, MCC**

### Model Comparison Table
| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.9016   | 0.9054 | 0.6474    | 0.3488 | 0.4533 | 0.4280 |
| Decision Tree            | 0.8777   | 0.7135 | 0.4783    | 0.4991 | 0.4884 | 0.4191 |
| kNN                      | 0.8961   | 0.8373 | 0.5931    | 0.3554 | 0.4444 | 0.4067 |
| Naive Bayes              | 0.8639   | 0.8088 | 0.4282    | 0.4877 | 0.4560 | 0.3797 |
| Random Forest (Ensemble) | 0.9045   | 0.9272 | 0.6554    | 0.3866 | 0.4863 | 0.4561 |
| XGBoost (Ensemble)       | **0.9080** | **0.9291** | 0.6348    | **0.5028** | **0.5612** | **0.5149** |

---

## Observations about Model Performance
| ML Model Name            | Observation about model performance                                                                 |
|--------------------------|-----------------------------------------------------------------------------------------------------|
| Logistic Regression      | Performs well as a simple linear baseline with high accuracy and AUC. Low recall due to class imbalance — tends to favor the majority class. |
| Decision Tree            | Shows signs of overfitting with the lowest AUC. Balanced precision/recall but weaker generalization compared to ensemble methods. |
| kNN                      | Reasonable accuracy but affected by high dimensionality after encoding. Moderate recall and MCC performance. |
| Naive Bayes              | Fast and surprisingly good recall despite independence assumption. Lowest precision — tends to over-predict subscriptions. |
| Random Forest (Ensemble) | Very robust with excellent precision and high AUC. Conservative recall but strong overall reliability. |
| XGBoost (Ensemble)       | **Best performing model overall** with highest AUC, F1 score, and MCC. Excellent handling of imbalance and feature interactions. |

---

## Conclusion
Among all the models tested, **XGBoost** performed the best in terms of AUC, F1 Score, and MCC, making it the most suitable model for predicting term deposit subscriptions in this project. Ensemble methods (Random Forest and XGBoost) clearly outperformed single models, especially on this imbalanced dataset.

**Deployment information**  
Live Streamlit App: https://[your-app-name].streamlit.app  
(Replace with your actual deployed link after successful deployment on Streamlit Community Cloud)

**Repository contents**  
- `app.py` – Streamlit application  
- `requirements.txt` – required Python packages  
- `model/` – saved trained models (.pkl files)  
- `model_training.ipynb` – training and evaluation notebook (executed on BITS Virtual Lab)
