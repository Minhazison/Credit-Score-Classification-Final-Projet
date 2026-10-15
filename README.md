# Credit Score Classification

A machine learning project that classifies customers into credit score brackets (**Poor / Standard / Good**) from their bank and credit-related information, to reduce manual effort in a finance company.

## Problem Statement

A global finance company has collected basic bank details and a large amount of credit-related information over the years. Management wants an intelligent system that segregates people into credit score brackets automatically. Given a person's credit-related information, the task is to build a model that predicts the credit score.

## Dataset

Two files, `train.csv` (100,000 rows, with labels) and `test.csv` (50,000 rows, no labels).

- 12,500 customers, each appearing in 12 monthly records (8 in train, 4 in test)
- Target: `Credit_Score` (Poor 29%, Standard 53%, Good 18% in train)
- Features include age, occupation, annual income, monthly salary, number of bank accounts and credit cards, interest rate, loans, delayed payments, credit mix, outstanding debt, credit utilization, credit history age, EMI, investments and monthly balance
- The data is deliberately dirty: placeholders such as `_`, `!@9#%8` and `NM`, values like `28_`, impossible ages (e.g. -500) and extreme outliers

> The dataset is not included in this repository. Place `train.csv` and `test.csv` in the project folder before running.

## Approach

1. **Exploratory analysis:** class balance, missing values, dirty-value examples
2. **Data cleaning:**
   - Remove junk placeholders and trailing underscores
   - Convert numeric-looking text to numbers and parse credit history age into months
   - Mark impossible values and outliers as missing
   - Repair static fields (age, income, accounts) with each customer's most common value, and monthly fields with each customer's median
3. **Feature engineering:**
   - One flag per loan type (multi-hot encoding)
   - Ratios: debt-to-income, EMI-to-salary, balance-to-salary, invest-to-salary, cards-per-account
   - Ordinal encoding of credit mix and minimum-payment behaviour
   - Payment behaviour split into spending level and payment size
   - One-hot encoding of occupation
   - Identifiers (`ID`, `Customer_ID`, `Name`, `SSN`) removed from the features
4. **Modelling:** Logistic Regression, Random Forest and Hist Gradient Boosting, all with balanced class weights
5. **Validation:** 5-fold `GroupKFold` grouped by customer, so the same person never appears in both training and validation
6. **Evaluation:** accuracy, macro-F1, per-class report, confusion matrix and permutation importance
7. **Prediction:** the best model is trained on all training data and predicts the test set

## Results

Cross-validated, grouped by customer:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression | 0.657 | 0.651 |
| Random Forest (30 trees) | 0.684 | 0.679 |
| **Hist Gradient Boosting** | **0.691** | **0.682** |

The most important features were credit mix, interest rate, outstanding debt, payment delay, credit history length and number of credit cards.

### Why about 69%?

Each customer appears in many rows. A random row-wise split would put the same person in both train and validation and inflate the score, so this project validates on unseen customers. The result is a more honest estimate of performance on new customers.

## Repository Structure

```
.
├── Credit_Score_Classification.ipynb   # full pipeline
├── README.md
├── train.csv                           # (not included) add locally
├── test.csv                            # (not included) add locally
└── outputs/
    ├── model_comparison.csv
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── test_predictions.csv
```

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install the requirements
   ```bash
   pip install numpy pandas matplotlib "scikit-learn>=1.2" jupyter
   ```
3. Put `train.csv` and `test.csv` in the project folder
4. Open `Credit_Score_Classification.ipynb` and run all cells in order

Running the full notebook takes a few minutes, mostly for the cross-validation.

## Output

`outputs/test_predictions.csv` has one row per test record, with columns `ID`, `Customer_ID` and `Predicted_Credit_Score`.

## Limitations and Future Work

- Outlier thresholds were chosen by inspecting the data and not tuned
- Model settings were kept modest and no hyperparameter search was done. `RandomizedSearchCV`, LightGBM or XGBoost may add a few points
- The test file has no labels, so cross-validation is the only performance estimate
- Using each customer's full history (for example, aggregating all 12 months) could improve results, but it was left out to keep the model usable for one record at a time

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter
